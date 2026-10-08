# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse
import requests
from tqdm import tqdm

# ============================================================================
# ============================================================================

SFT = "your_path"
DPO_ONLINE = "your_path"
MODEL_PATH = DPO_ONLINE
DATASET_NAME = "bamboogle"
GOOGLE_SEARCH_KEY="your_key"


INPUT_JSONL = f"your_path"
OUTPUT_DIR = f"your_path"
os.makedirs(OUTPUT_DIR, exist_ok=True)
QUESTION_FIELD = "question"
NUM_ROLLS_PER_QUESTION = 1

MAX_TURNS = 5
MAX_COMPLETION_LENGTH = 512
MAX_TOTAL_LENGTH = 8192
TEMPERATURE = 1.0
BLOCK_LENGTH = 128
DENOISING_STEPS = 128
REMASKING_STRATEGY = "low_confidence_static"
DYNAMIC_THRESHOLD = 0.9

MAX_NUM_SEQS = 1
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.8

BATCH_SIZE = 8

SEARCH_API_URL = "your_api"

# ============================================================================
# ============================================================================

@dataclass
class RolloutResult:
    question: str
    answer: str  # ground truth answer
    prediction: Optional[str]
    num_turns: int
    termination_reason: str
    messages: List[Dict]  

@dataclass
class ActiveSample:
    idx: int
    question: str
    answer: str  # ground truth
    messages: List[Dict]  
    num_turns: int
    context: str = ""  

# ============================================================================
# ============================================================================


class SearchTool:
    name = "search"
    description = "Performs batched web searches"
    
    def google_search(self, query: str) -> str:
        headers = {
            'X-API-KEY': GOOGLE_SEARCH_KEY,
            'Content-Type': 'application/json',
        }
        data = {
            "q": query,
            "num": 10,
            "extendParams": {
                "country": "en",
                "page": 1,
            },
        }

        for i in range(5):
            try:
                response = requests.post(
                    SEARCH_API_URL,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=30
                )
                results = response.json()
                break
            except Exception as e:
                print(f"[search] Request failed (attempt {i + 1}/5): {e}")
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = []
            idx = 0
            
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content

        except Exception as e:
            print(f"[search] Parse error: {e}")
            return f"No results found for '{query}'. Try with a more general query, or remove the year filter."
    
    def __call__(self, args: Dict) -> str:
        try:
            query = args.get("query", [])
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = self.google_search(query)
        else:
            assert isinstance(query, list)
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(self.google_search, query))
            response = "\n=======\n".join(response)
        return response


# ============================================================================
# ReAct Rollout 
# ============================================================================

class SimpleReActRollout:
    ANSWER_START = "<|box_start|>"
    ANSWER_END = "<|box_end|>"
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_RESP_START = "<tool_response>"
    TOOL_RESP_END = "</tool_response>"
    
    def __init__(self, tokenizer, tool_executor):
        self.tokenizer = tokenizer
        self.tool_executor = tool_executor
        self.system_prompt = self._get_system_prompt()
        self.user_prompt_template = self._get_user_prompt()
    
    def _get_system_prompt(self) -> str:
        return '''You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.'''

    def _get_user_prompt(self) -> str:
        return """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Array of query strings. Include 3 or less search queries in a single call."
      }
    },
    "required": [
      "query"
    ]
    }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with answer of the question. The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think>
thinking process here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["query string 1", "query string 2", ...]}}   
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think>
thinking process here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["another query string"]}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<|box_start|>
answer here
<|box_end|>
The assistant must strictly abide by the above format. Tool calls should be placed between <tool_call> and </tool_call>, tool responses between <tool_response> and </tool_response>, thinking processes between <think> and </think>, and answers between <|box_start|> and <|box_end|>.
User: """

    def build_initial_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template + question}
        ]
        
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt_str
    
    def build_tool_response_segment(self, tool_result: str) -> str:
        return f"<|im_start|>user\n<tool_response>\n{tool_result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    
    def parse_tool_call(self, content: str) -> Optional[Tuple[str, Dict]]:
        if self.TOOL_CALL_START not in content or self.TOOL_CALL_END not in content:
            return None
        
        try:
            tool_call_str = content.split(self.TOOL_CALL_START)[1].split(self.TOOL_CALL_END)[0]
            tool_call = json.loads(tool_call_str.strip())
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})
            return tool_name, tool_args
        except:
            return None
    
    def has_answer(self, content: str) -> bool:
        return self.ANSWER_START in content and self.ANSWER_END in content
    
    def extract_answer(self, content: str) -> Optional[str]:
        if not self.has_answer(content):
            return None
        try:
            return content.split(self.ANSWER_START)[1].split(self.ANSWER_END)[0].strip()
        except IndexError:
            return None
    
    def _clean_generated_text(self, text: str) -> str:
        if self.TOOL_RESP_START in text:
            pos = text.find(self.TOOL_RESP_START)
            text = text[:pos]
        return text
    
    def _call_tool(self, tool_name: str, tool_args: Dict) -> str:
        try:
            result = self.tool_executor(tool_args)
        except Exception as e:
            result = f'Error: Tool call failed. {str(e)}'
        return result
    
    def _execute_tool_calls_batch(self, tool_calls: List[Tuple[int, str, Dict]]) -> Dict[int, str]:
        results = {}
        
        def execute_single(item):
            idx, tool_name, tool_args = item
            result = self._call_tool(tool_name, tool_args)
            return idx, result
        
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 10)) as executor:
            for idx, result in executor.map(execute_single, tool_calls):
                results[idx] = result
        return results
    
    def format_prompt(self, prompt: str) -> str:
        PADDING_PROMPT = "Note that I need you to keep the content as concise as possible, limit it to no more than 128 tokens, but near to 128 tokens." * 10
        prompt_len = len(self.tokenizer.encode(prompt))
        padding_len = BLOCK_LENGTH - (prompt_len % BLOCK_LENGTH)
        if padding_len == BLOCK_LENGTH:
            padding_len = 0
        if padding_len > 0:
            padding_tokens = self.tokenizer.decode(
                self.tokenizer.encode(PADDING_PROMPT)[:padding_len]
            )
            prompt = padding_tokens + prompt
        return prompt
    
    def rollout_batch(
        self,
        questions: List[str],
        answers: List[str],
        llm,
        sampling_params,
        rank: int = 0,
        pbar: Optional[tqdm] = None,
    ) -> List[RolloutResult]:
        
        active_samples: List[ActiveSample] = []
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            prompt_text = self.build_initial_prompt(question)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_template + question}
            ]
            
            sample = ActiveSample(
                idx=idx,
                question=question,
                answer=answer,
                messages=messages,
                num_turns=0,
                context=prompt_text,
            )
            active_samples.append(sample)
        
        completed_results: Dict[int, RolloutResult] = {}
        
        for turn in range(MAX_TURNS):
            if not active_samples:
                break
            
            if rank == 0 and pbar is not None:
                pbar.set_postfix({"turn": turn + 1, "active": len(active_samples)})
            
            prompts_for_generation = [self.format_prompt(s.context) for s in active_samples]
            
            try:
                if turn >= 3:
                    sampling_params.remasking_strategy = "answer_rl"
                else:
                    sampling_params.remasking_strategy = "toolcall_pre_rl"

                outputs = llm.generate_streaming(
                    prompts_for_generation,
                    sampling_params,
                    max_active=MAX_NUM_SEQS
                )
                outputs = list(outputs)
                
            except Exception as e:
                print(f"[Rank {rank}] Generation error: {e}")
                for sample in active_samples:
                    completed_results[sample.idx] = self._create_result(
                        sample, None, "error"
                    )
                break
            
            still_active = []
            tool_calls_to_execute = []
            
            for sample, output in zip(active_samples, outputs):
                sample.num_turns = turn + 1
                
                new_text = output['text']
                import rich
                if rank == 0:
                    rich.print(f"[Rank {rank}][Turn {turn}] {new_text}")
                
                if self.TOOL_RESP_START in new_text:
                    pos = new_text.find(self.TOOL_RESP_START)
                    new_text = new_text[:pos]
                
                sample.messages.append({"role": "assistant", "content": new_text.strip()})
                sample.context += new_text
                
                token_count = len(self.tokenizer.encode(sample.context))
                if token_count >= MAX_TOTAL_LENGTH:
                    sample.messages[-1]['content'] = 'Sorry, the number of tokens exceeds the limit.'
                    completed_results[sample.idx] = self._create_result(
                        sample, self.extract_answer(new_text), "max_length"
                    )
                    continue
                
                if self.has_answer(new_text):
                    completed_results[sample.idx] = self._create_result(
                        sample, self.extract_answer(new_text), "answer"
                    )
                    continue
                
                if self.TOOL_CALL_START in new_text and self.TOOL_CALL_END in new_text:
                    tool_info = self.parse_tool_call(new_text)
                    if tool_info is not None:
                        tool_name, tool_args = tool_info
                        tool_calls_to_execute.append((sample, tool_name, tool_args))
                    else:
                        error_msg = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                        resp_text = self.build_tool_response_segment(error_msg)
                        sample.messages.append({"role": "user", "content": f"<tool_response>\n{error_msg}\n</tool_response>"})
                        sample.context += resp_text
                        still_active.append(sample)
                else:
                    still_active.append(sample)
            
            if tool_calls_to_execute:
                tool_call_items = [
                    (sample.idx, tool_name, tool_args)
                    for sample, tool_name, tool_args in tool_calls_to_execute
                ]
                tool_results = self._execute_tool_calls_batch(tool_call_items)
                
                for sample, tool_name, tool_args in tool_calls_to_execute:
                    tool_result = tool_results.get(sample.idx, "Error: No result")
                    
                    resp_text = self.build_tool_response_segment(tool_result)
                    sample.messages.append({"role": "user", "content": f"<tool_response>\n{tool_result}\n</tool_response>"})
                    sample.context += resp_text
                    
                    token_count = len(self.tokenizer.encode(sample.context))
                    if token_count >= MAX_TOTAL_LENGTH:
                        completed_results[sample.idx] = self._create_result(
                            sample, None, "max_length"
                        )
                    else:
                        still_active.append(sample)
            
            active_samples = still_active
            
            if rank == 0 and pbar is not None:
                pbar.update(0)
        
        for sample in active_samples:
            if sample.idx not in completed_results:
                last_content = sample.messages[-1]['content'] if sample.messages else ''
                if self.has_answer(last_content):
                    prediction = self.extract_answer(last_content)
                    termination = 'answer'
                else:
                    prediction = None
                    termination = 'max_turns'
                
                completed_results[sample.idx] = self._create_result(
                    sample, prediction, termination
                )
        
        results = [completed_results[i] for i in range(len(questions))]
        return results
    
    def _create_result(
        self,
        sample: ActiveSample,
        prediction: Optional[str],
        termination_reason: str,
    ) -> RolloutResult:
        return RolloutResult(
            question=sample.question,
            answer=sample.answer,
            prediction=prediction,
            num_turns=sample.num_turns,
            termination_reason=termination_reason,
            messages=sample.messages,
        )


# ============================================================================
# ============================================================================

def load_jsonl(filepath: str, question_field: str = "question") -> List[Dict]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if question_field in item:
                    data.append(item)
    return data


def result_to_dict(result: RolloutResult, roll_idx: int) -> Dict:
    return {
        "question": result.question,
        "answer": result.answer,
        "prediction": result.prediction,
        "roll_idx": roll_idx,
        "num_turns": result.num_turns,
        "termination_reason": result.termination_reason,
        "messages": result.messages,
    }


# ============================================================================
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=INPUT_JSONL, help="Input jsonl file")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path")
    parser.add_argument("--num_rolls", type=int, default=NUM_ROLLS_PER_QUESTION, help="Rolls per question")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if rank == 0:
        print(f"=" * 60)
        print(f"ReAct Data Roller")
        print(f"=" * 60)
        print(f"World size: {world_size}")
        print(f"Input: {args.input}")
        print(f"Output dir: {args.output_dir}")
        print(f"Model: {args.model_path}")
        print(f"Rolls per question: {args.num_rolls}")
        print(f"Batch size: {args.batch_size}")
        print(f"=" * 60)
    
    all_data = load_jsonl(args.input, QUESTION_FIELD)
    if rank == 0:
        print(f"Loaded {len(all_data)} questions from {args.input}")
    
    shard_data = all_data[rank::world_size]
    if rank == 0:
        print(f"Each rank processing ~{len(shard_data)} questions")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    from jetengine import LLM, SamplingParams
    llm = LLM(
        model=args.model_path,
        mask_token_id=tokenizer.mask_token_id,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        block_length=BLOCK_LENGTH,
        max_num_seqs=MAX_NUM_SEQS,
    )
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_COMPLETION_LENGTH,
        block_length=BLOCK_LENGTH,
        denoising_steps=DENOISING_STEPS,
        remasking_strategy=REMASKING_STRATEGY,
        dynamic_threshold=DYNAMIC_THRESHOLD,
        stop_words=[151645, 151658], # <|im_end|> </tool_call>
        topk=1,
        topp=1.0
    )
    
    tool_executor = SearchTool()
    rollout_engine = SimpleReActRollout(tokenizer, tool_executor)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"rollout_results_rank{rank}_{timestamp}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_questions = len(shard_data)
    total_iterations = (total_questions + args.batch_size - 1) // args.batch_size * args.num_rolls
    
    with open(output_path, 'a', encoding='utf-8') as output_file:
        total_saved = 0
        
        if rank == 0:
            pbar = tqdm(total=total_iterations, desc="Processing", unit="batch")
        else:
            pbar = None
        
        for batch_start in range(0, total_questions, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_questions)
            batch_data = shard_data[batch_start:batch_end]
            
            if rank == 0:
                print(f"\n[Rank {rank}] Batch {batch_start//args.batch_size + 1}: questions {batch_start+1}-{batch_end}/{total_questions}")
            
            for roll_idx in range(args.num_rolls):
                if rank == 0 and pbar is not None:
                    pbar.set_description(f"Batch {batch_start//args.batch_size + 1}, Roll {roll_idx + 1}/{args.num_rolls}")
                
                questions = [item[QUESTION_FIELD] for item in batch_data]
                answers = [item.get("answer", "") for item in batch_data]
                
                results = rollout_engine.rollout_batch(
                    questions=questions,
                    answers=answers,
                    llm=llm,
                    sampling_params=sampling_params,
                    rank=rank,
                    pbar=pbar,
                )
                
                for item, result in zip(batch_data, results):
                    result_dict = result_to_dict(result, roll_idx)
                    result_dict["original_data"] = {k: v for k, v in item.items() if k not in [QUESTION_FIELD, "answer"]}
                    
                    output_file.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                    output_file.flush()
                    total_saved += 1
                
                if rank == 0:
                    term_reasons = {}
                    for r in results:
                        term_reasons[r.termination_reason] = term_reasons.get(r.termination_reason, 0) + 1
                    pbar.write(f"    Saved {len(results)} results. Total: {total_saved}")
                    pbar.write(f"    Termination: {term_reasons}")
                    pbar.update(1)
        
        if rank == 0 and pbar is not None:
            pbar.close()
    
    if rank == 0:
        print(f"\n[Rank {rank}] All {total_saved} results saved to {output_path}")
    
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            print(f"\nAll {world_size} ranks completed!")
        dist.destroy_process_group()

if __name__ == "__main__":
    main()