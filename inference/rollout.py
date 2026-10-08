#!/usr/bin/env python3
"""Shared multi-turn chemistry-agent rollout logic for AR and dLLM models."""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from tool_registry import ToolExecutor, ToolSpec, unified_system_prompt


QUESTION_FIELD = "question"
ANSWER_START = "<|box_start|>"
ANSWER_END = "<|box_end|>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


class TextGenerator(Protocol):
    tokenizer: Any

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


@dataclass
class ParsedTurn:
    kind: str
    text: str
    thought: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    answer: str | None = None
    parse_error: str | None = None


@dataclass
class RolloutState:
    question: str
    answer: Any
    sample_id: str
    context: str
    messages: list[dict[str, Any]]
    tool_use_chain: list[dict[str, Any]] = field(default_factory=list)
    raw_generations: list[str] = field(default_factory=list)
    prediction: str | None = None
    termination_reason: str = ""
    num_turns: int = 0


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL)
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)
ANSWER_RE = re.compile(r"<\|box_start\|>\s*(.*?)\s*<\|box_end\|>", flags=re.DOTALL)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(make_json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def _parse_payload(raw: str) -> tuple[str, dict[str, Any]]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except Exception:
        try:
            payload = ast.literal_eval(text)
        except Exception as original_exc:
            # A recurring generation in the recovered chemistry checkpoints is
            # ``{"arguments":{"input": 1.0, 2.0}}``.  Its intended contract,
            # confirmed by the recorded calling chain, is input=[1.0, 2.0].
            # Repair only when input is the final argument and the values form
            # a valid JSON/Python list; arbitrary malformed payloads still fail.
            match = re.search(
                r"(?P<prefix>['\"]input['\"]\s*:\s*)"
                r"(?P<values>.*?)"
                r"(?P<suffix>}\s*}\s*)$",
                text,
                flags=re.DOTALL,
            )
            repaired_payload = None
            if match and "," in match.group("values"):
                values_text = f"[{match.group('values')}]"
                try:
                    values = json.loads(values_text)
                except Exception:
                    try:
                        values = ast.literal_eval(values_text)
                    except Exception:
                        values = None
                if isinstance(values, list) and len(values) > 1:
                    repaired = (
                        text[: match.start()]
                        + match.group("prefix")
                        + json.dumps(values, ensure_ascii=False)
                        + match.group("suffix")
                    )
                    try:
                        repaired_payload = json.loads(repaired)
                    except Exception:
                        try:
                            repaired_payload = ast.literal_eval(repaired)
                        except Exception:
                            repaired_payload = None
            if repaired_payload is None:
                raise ValueError(
                    f"tool payload is neither JSON nor a Python literal: {original_exc}"
                ) from original_exc
            payload = repaired_payload
    if not isinstance(payload, dict):
        raise ValueError("tool payload must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool payload has no valid name")
    arguments = make_json_safe(payload.get("arguments", {}))
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}
    return name.strip(), arguments


def _last_thought_before(text: str, end: int) -> str | None:
    matches = THINK_RE.findall(text[:end])
    return matches[-1].strip() if matches else None


def clean_generation(text: str) -> str:
    text = (
        text.replace("‹", "<")
        .replace("›", ">")
        .replace("＜", "<")
        .replace("＞", ">")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    text = re.sub(r"^\s*<\|im_start\|>assistant\s*", "", text)
    if "<tool_response>" in text:
        text = text.split("<tool_response>", 1)[0]
    for marker in ("<|im_start|>user", "<|im_start|>assistant"):
        if marker in text:
            text = text.split(marker, 1)[0]
    if ANSWER_END in text:
        return text[: text.rfind(ANSWER_END) + len(ANSWER_END)].strip()
    if TOOL_CALL_END in text:
        return text[: text.rfind(TOOL_CALL_END) + len(TOOL_CALL_END)].strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def parse_turn(text: str) -> ParsedTurn:
    cleaned = clean_generation(text)
    actions: list[tuple[int, str, re.Match[str]]] = []
    actions.extend((match.end(), "answer", match) for match in ANSWER_RE.finditer(cleaned))
    actions.extend((match.end(), "tool", match) for match in TOOL_RE.finditer(cleaned))
    if not actions:
        thought_matches = THINK_RE.findall(cleaned)
        thought = thought_matches[-1].strip() if thought_matches else None
        # Some recovered checkpoints emit <|im_end|> immediately after the
        # answer body and omit <|box_end|>. ``clean_generation`` has already
        # removed that turn delimiter, so safely close a non-empty final box.
        if ANSWER_START in cleaned:
            answer = cleaned.rsplit(ANSWER_START, 1)[1].strip()
            if answer:
                thought_block = f"<think>\n{thought}\n</think>\n" if thought else ""
                return ParsedTurn(
                    kind="answer",
                    thought=thought,
                    answer=answer,
                    text=f"{thought_block}{ANSWER_START}{answer}{ANSWER_END}",
                )
        return ParsedTurn(
            kind="invalid",
            text=cleaned,
            thought=thought,
            parse_error="No complete <tool_call> or <|box_start|> action found",
        )

    _, kind, match = max(actions, key=lambda item: item[0])
    thought = _last_thought_before(cleaned, match.end())
    thought_block = f"<think>\n{thought}\n</think>\n" if thought else ""
    if kind == "answer":
        answer = match.group(1).strip()
        return ParsedTurn(
            kind="answer",
            thought=thought,
            answer=answer,
            text=f"{thought_block}{ANSWER_START}{answer}{ANSWER_END}",
        )

    try:
        name, arguments = _parse_payload(match.group(1))
    except ValueError as exc:
        return ParsedTurn(
            kind="invalid",
            thought=thought,
            text=cleaned,
            parse_error=str(exc),
        )
    payload = json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ": ")
    )
    return ParsedTurn(
        kind="tool",
        thought=thought,
        tool_name=name,
        tool_arguments=arguments,
        text=f"{thought_block}{TOOL_CALL_START}\n{payload}\n{TOOL_CALL_END}",
    )


def build_initial_context(question: str, system_prompt: str | None = None) -> str:
    system_prompt = system_prompt or unified_system_prompt()
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question}\n\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_tool_response(tool_result: str) -> str:
    return (
        f"<|im_start|>user\n<tool_response>\n{tool_result}\n</tool_response>"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )


def _prompt_token_count(generator: TextGenerator, text: str) -> int:
    return len(generator.tokenizer.encode(text, add_special_tokens=False))


def run_one(
    item: dict[str, Any],
    generator: TextGenerator,
    tools: list[ToolSpec],
    *,
    max_turns: int,
    max_new_tokens: int,
    max_context_tokens: int,
    tool_timeout_seconds: int,
    print_turns: bool = False,
) -> dict[str, Any]:
    question = str(item[QUESTION_FIELD])
    sample_id = str(item.get("sample_id") or "sample")
    system_prompt = unified_system_prompt()
    state = RolloutState(
        question=question,
        answer=item.get("answer", ""),
        sample_id=sample_id,
        context=build_initial_context(question, system_prompt),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n"},
        ],
    )
    executor = ToolExecutor(tools, timeout_seconds=tool_timeout_seconds)
    started_at = time.perf_counter()

    for turn_index in range(1, max_turns + 1):
        state.num_turns = turn_index
        if _prompt_token_count(generator, state.context) >= max_context_tokens:
            state.termination_reason = "max_context_tokens"
            break

        raw = generator.generate(state.context, max_new_tokens=max_new_tokens)
        state.raw_generations.append(raw)
        parsed = parse_turn(raw)
        if print_turns:
            print("=" * 80, flush=True)
            print(f"[{sample_id} turn={turn_index} kind={parsed.kind}]", flush=True)
            print(raw, flush=True)
            if parsed.text != clean_generation(raw):
                print("[normalized]", parsed.text, flush=True)

        if parsed.kind == "invalid":
            state.messages.append(
                {
                    "role": "assistant",
                    "content": parsed.text,
                    "parse_error": parsed.parse_error,
                }
            )
            state.termination_reason = "invalid_generation"
            break

        state.context += parsed.text + "<|im_end|>\n"
        state.messages.append({"role": "assistant", "content": parsed.text})

        if parsed.kind == "answer":
            state.prediction = parsed.answer
            state.termination_reason = "answer"
            break

        assert parsed.tool_name is not None and parsed.tool_arguments is not None
        tool_started_at = time.perf_counter()
        result, success = executor.execute(parsed.tool_name, parsed.tool_arguments)
        tool_seconds = time.perf_counter() - tool_started_at
        state.tool_use_chain.append(
            {
                "turn": turn_index,
                "thought": parsed.thought,
                "tool": parsed.tool_name,
                "input": parsed.tool_arguments.get("input", parsed.tool_arguments),
                "arguments": parsed.tool_arguments,
                "output": result,
                "success": success,
                "seconds": round(tool_seconds, 6),
            }
        )
        state.messages.append(
            {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}
        )
        state.context += build_tool_response(result)
        if print_turns:
            print(f"[tool response success={success}] {result}", flush=True)
    else:
        state.termination_reason = "max_turns"

    if not state.termination_reason:
        state.termination_reason = "max_turns"

    expected_tools = list(item.get("expected_tools") or [])
    observed_tools = [step["tool"] for step in state.tool_use_chain]
    row = {
        "question": state.question,
        "answer": state.answer,
        "prediction": state.prediction,
        "sample_id": state.sample_id,
        "num_turns": state.num_turns,
        "termination_reason": state.termination_reason,
        "messages": state.messages,
        "raw_generations": state.raw_generations,
        "tool_use_chain": state.tool_use_chain,
        "evaluation": {
            "expected_tools": expected_tools,
            "observed_tools": observed_tools,
            "tool_sequence_match": observed_tools == expected_tools if expected_tools else None,
            "all_tool_calls_succeeded": all(step["success"] for step in state.tool_use_chain),
        },
        "timing": {"total_seconds": round(time.perf_counter() - started_at, 6)},
        "original_data": item,
    }
    return make_json_safe(row)


def read_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if QUESTION_FIELD not in row:
                    raise ValueError(f"Missing {QUESTION_FIELD!r} in {path}:{line_number}")
                rows.append(row)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"JSON dataset must be a list: {path}")
    return [row for row in data if QUESTION_FIELD in row]


def completed_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                keys.add((str(row["sample_id"]), int(row.get("roll_idx", 0))))
            except Exception:
                continue
    return keys


def run_dataset(
    *,
    input_path: str | Path,
    output_path: str | Path,
    generator: TextGenerator,
    tools: list[ToolSpec],
    max_samples: int,
    sample_id: str | None,
    num_rolls: int,
    resume: bool,
    max_turns: int,
    max_new_tokens: int,
    max_context_tokens: int,
    tool_timeout_seconds: int,
    print_turns: bool,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    rows = read_dataset(input_path)
    if sample_id:
        rows = [row for row in rows if str(row.get("sample_id")) == sample_id]
        if not rows:
            raise ValueError(f"sample_id not found in {input_path}: {sample_id}")
    if max_samples > 0:
        rows = rows[:max_samples]
    rows = rows[shard_index::num_shards]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output_path) if resume else set()
    mode = "a" if resume else "w"
    total_pending = sum(
        (str(row.get("sample_id") or "sample"), roll_idx) not in done
        for row in rows
        for roll_idx in range(num_rolls)
    )
    counts = {"total": 0, "answered": 0, "tool_errors": 0, "invalid_generation": 0}
    with output_path.open(mode, encoding="utf-8") as output_handle:
        progress = tqdm(total=total_pending, desc=output_path.stem, unit="roll")
        for row in rows:
            for roll_idx in range(num_rolls):
                key = (str(row.get("sample_id") or "sample"), roll_idx)
                if key in done:
                    continue
                result = run_one(
                    row,
                    generator,
                    tools,
                    max_turns=max_turns,
                    max_new_tokens=max_new_tokens,
                    max_context_tokens=max_context_tokens,
                    tool_timeout_seconds=tool_timeout_seconds,
                    print_turns=print_turns,
                )
                result["roll_idx"] = roll_idx
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                counts["total"] += 1
                counts["answered"] += result["termination_reason"] == "answer"
                counts["tool_errors"] += not result["evaluation"]["all_tool_calls_succeeded"]
                counts["invalid_generation"] += result["termination_reason"] == "invalid_generation"
                progress.update(1)
                progress.set_postfix(counts)
        progress.close()
    return {
        "input": str(input_path),
        "output": str(output_path),
        "num_shards": num_shards,
        "shard_index": shard_index,
        **counts,
    }
