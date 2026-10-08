#!/usr/bin/env python3
"""Memory-conscious block-diffusion generation for SDAR models."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers.cache_utils import DynamicCache


def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, -torch.inf)


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
    return logits.masked_fill(mask, -torch.inf)


def sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one candidate and its probability for every masked position."""
    logits = logits.float()
    if temperature <= 0:
        probabilities = torch.softmax(logits, dim=-1)
        tokens = torch.argmax(logits, dim=-1)
        confidence = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, confidence
    logits = logits / temperature
    logits = _top_k_logits(logits, top_k)
    logits = _top_p_logits(logits, top_p)
    probabilities = torch.softmax(logits, dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    tokens = torch.multinomial(flat, num_samples=1).reshape(probabilities.shape[:-1])
    confidence = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, confidence


class BlockDiffusionGenerator:
    """Generate one assistant turn using SDAR block diffusion and KV caching.

    The complete prompt blocks are prefetched into KV cache.  Any prompt tail
    shares the first denoising block with newly generated masks, matching
    ``sdar_generate.py`` and JetEngine in the recovered DLLM-Searcher repo.
    The implementation intentionally supports batch size one: on a 24 GiB
    RTX 3090 this leaves enough memory for an 8B bf16 model and a useful cache.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        *,
        block_length: int = 64,
        denoising_steps: int = 64,
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
        remasking_strategy: str = "low_confidence_static",
        confidence_threshold: float = 0.9,
        max_context_tokens: int = 32768,
    ):
        if block_length < 1:
            raise ValueError("block_length must be positive")
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")
        if remasking_strategy not in {
            "low_confidence_static", "low_confidence_dynamic", "sequential"
        }:
            raise ValueError(f"Unsupported remasking strategy: {remasking_strategy}")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_length = int(block_length)
        self.denoising_steps = int(denoising_steps)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.remasking_strategy = remasking_strategy
        self.confidence_threshold = float(confidence_threshold)
        self.max_context_tokens = int(max_context_tokens)
        self.mask_id = self._mask_id()
        self.pad_id = int(tokenizer.pad_token_id)
        self.stop_ids = self._single_token_ids(
            ["</tool_call>", "<|box_end|>", "<|im_end|>"]
        )

    def _mask_id(self) -> int:
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            ids = self.tokenizer.encode("<|MASK|>", add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("Tokenizer has no single mask token")
            mask_id = ids[0]
        return int(mask_id)

    def _single_token_ids(self, tokens: list[str]) -> list[int]:
        ids = []
        for token in tokens:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
        return list(dict.fromkeys(ids))

    def _prefill_mask(self, length: int) -> torch.Tensor:
        positions = torch.arange(length, device=self.device)
        block_ids = positions // self.block_length
        visible = block_ids[:, None] >= block_ids[None, :]
        return visible[None, None, :, :]

    def _decode_mask(self, cached_length: int, block_length: int) -> torch.Tensor:
        return torch.ones(
            (1, 1, block_length, cached_length + block_length),
            dtype=torch.bool,
            device=self.device,
        )

    def _select_transfer(
        self,
        mask: torch.Tensor,
        confidence: torch.Tensor,
        *,
        steps_left: int,
    ) -> torch.Tensor:
        remaining = int(mask.sum().item())
        count = max(1, math.ceil(remaining / max(1, steps_left)))
        transfer = torch.zeros_like(mask)
        if self.remasking_strategy == "sequential":
            indices = mask.nonzero(as_tuple=False)[:count]
            transfer[indices[:, 0], indices[:, 1]] = True
            return transfer

        eligible_confidence = torch.where(mask, confidence, -torch.inf)
        if self.remasking_strategy == "low_confidence_dynamic":
            high = mask & (confidence >= self.confidence_threshold)
            if int(high.sum().item()) >= count:
                return high
        for batch_index in range(mask.shape[0]):
            batch_count = min(count, int(mask[batch_index].sum().item()))
            if batch_count:
                indices = torch.topk(eligible_confidence[batch_index], batch_count).indices
                transfer[batch_index, indices] = True
        return transfer

    def _resolved_stop(self, block: torch.Tensor, start_index: int = 0) -> int | None:
        row = block[0]
        unresolved = row.eq(self.mask_id)
        for index, token in enumerate(row.tolist()[start_index:], start=start_index):
            if token in self.stop_ids and not bool(unresolved[: index + 1].any().item()):
                return index
        return None

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        requested = int(max_new_tokens)
        total_length = math.ceil(
            (len(prompt_ids) + requested) / self.block_length
        ) * self.block_length
        if total_length > self.max_context_tokens:
            raise ValueError(
                f"Prompt/generation needs {total_length} block-aligned tokens, "
                f"limit is {self.max_context_tokens}"
            )

        prefill_length = (len(prompt_ids) // self.block_length) * self.block_length
        prompt_tail = prompt_ids[prefill_length:]
        cache = DynamicCache()

        if prefill_length:
            input_ids = torch.tensor(
                [prompt_ids[:prefill_length]], dtype=torch.long, device=self.device
            )
            position_ids = torch.arange(
                prefill_length, device=self.device
            ).unsqueeze(0)
            prefill_mask = self._prefill_mask(prefill_length)
            # Calling the base model avoids materializing prompt-length
            # vocabulary logits (over 1 GiB for a long prompt).
            self.model.model(
                input_ids=input_ids,
                attention_mask=prefill_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                store_kv=True,
            )

        generated: list[int] = []
        number_of_blocks = math.ceil(
            (len(prompt_tail) + requested) / self.block_length
        )
        cached_length = prefill_length

        for block_index in range(number_of_blocks):
            remaining_requested = requested - len(generated)
            fixed_prefix = prompt_tail if block_index == 0 else []
            generation_start = len(fixed_prefix)
            current_length = min(
                self.block_length - generation_start, remaining_requested
            )
            model_block_length = self.block_length
            current_values = fixed_prefix + [self.mask_id] * (
                model_block_length - generation_start
            )
            current = torch.tensor(
                [current_values], dtype=torch.long, device=self.device
            )
            decode_mask = self._decode_mask(cached_length, model_block_length)
            block_positions = torch.arange(
                cached_length,
                cached_length + model_block_length,
                device=self.device,
            ).unsqueeze(0)

            for step in range(self.denoising_steps):
                mask = current.eq(self.mask_id)
                if not bool(mask.any().item()):
                    break
                output = self.model(
                    input_ids=current,
                    attention_mask=decode_mask,
                    position_ids=block_positions,
                    past_key_values=cache,
                    use_cache=True,
                    store_kv=False,
                )
                candidates, confidence = sample_tokens(
                    output.logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
                transfer = self._select_transfer(
                    mask,
                    confidence,
                    steps_left=self.denoising_steps - step,
                )
                current[transfer] = candidates[transfer]
                stop_index = self._resolved_stop(current, generation_start)
                if (
                    stop_index is not None
                    and stop_index < generation_start + current_length
                ):
                    result_ids = generated + current[
                        0, generation_start : stop_index + 1
                    ].tolist()
                    return self.tokenizer.decode(result_ids, skip_special_tokens=False)

            # A dynamic threshold can leave positions unresolved only if a
            # custom strategy is introduced. Fill defensively with one greedy
            # pass so masks never leak into the transcript.
            remaining_mask = current.eq(self.mask_id)
            if bool(remaining_mask.any().item()):
                output = self.model(
                    input_ids=current,
                    attention_mask=decode_mask,
                    position_ids=block_positions,
                    past_key_values=cache,
                    use_cache=True,
                    store_kv=False,
                )
                candidates = torch.argmax(output.logits, dim=-1)
                current[remaining_mask] = candidates[remaining_mask]

            stop_index = self._resolved_stop(current, generation_start)
            if (
                stop_index is not None
                and stop_index < generation_start + current_length
            ):
                result_ids = generated + current[
                    0, generation_start : stop_index + 1
                ].tolist()
                return self.tokenizer.decode(result_ids, skip_special_tokens=False)

            generated.extend(
                current[0, generation_start : generation_start + current_length].tolist()
            )
            if current_length < model_block_length - generation_start:
                break

            # Commit the complete block to the KV cache for the next block.
            self.model.model(
                input_ids=current,
                attention_mask=decode_mask,
                position_ids=block_positions,
                past_key_values=cache,
                use_cache=True,
                store_kv=True,
            )
            cached_length += model_block_length

        return self.tokenizer.decode(generated[:requested], skip_special_tokens=False)
