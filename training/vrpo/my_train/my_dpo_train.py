#!/usr/bin/env python


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
import torch
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config, DPOConfig

from my_dpo_trainer import SDARDPOTrainer

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class SDARDPOScriptArguments:
    """Arguments for SDAR DPO training script."""
    
    # Dataset
    dataset_name: str = field(
        default="sdar_dpo",
        metadata={"help": "Name of the dataset (for logging)"}
    )
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset (json/jsonl file or HuggingFace dataset)"}
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training"}
    )
    dataset_eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset split to use for evaluation"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use"}
    )
    
    # SDAR specific
    block_length: int = field(
        default=128,
        metadata={"help": "Block length for block diffusion"}
    )
    num_mc: int = field(
        default=2,
        metadata={"help": "Number of Monte Carlo samples for ELBO estimation"}
    )
    mask_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Mask token ID (if None, auto-detect from tokenizer)"}
    )
    
    # Tool response markers (for trainable mask)
    tool_resp_left: str = field(
        default="<|im_start|>user\n<tool_response>",
        metadata={"help": "Left marker for tool response"}
    )
    tool_resp_right: str = field(
        default="</tool_response><|im_end|>\n<|im_start|>assistant\n",
        metadata={"help": "Right marker for tool response"}
    )


@dataclass
class SDARDPOConfig(DPOConfig):
    """Extended DPO config for SDAR training."""
    
    # Override some defaults
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=0)
    
    # Wandb
    wandb_project: Optional[str] = field(
        default="sdar_dpo",
        metadata={"help": "Wandb project name"}
    )


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dpo_dataset(
    dataset_path: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> datasets.Dataset:
    """
    Load dataset for DPO training.
    
    Expected format:
    {
        "prompt": str,
        "chosen": str,
        "rejected": str,
    }
    """
    if dataset_path is None:
        # Create a dummy dataset for testing
        logger.warning("No dataset path provided, creating dummy dataset")
        data = {
            "prompt": [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
            ],
            "chosen": [
                "<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>",
                "<|im_start|>assistant\nRomeo and Juliet was written by William Shakespeare.<|im_end|>",
            ],
            "rejected": [
                "<|im_start|>assistant\nI don't know.<|im_end|>",
                "<|im_start|>assistant\nI'm not sure about that.<|im_end|>",
            ],
        }
        return datasets.Dataset.from_dict(data)
    
    # Load from file or HuggingFace hub
    if os.path.exists(dataset_path):
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = datasets.load_dataset('json', data_files=dataset_path, split='train')
        elif dataset_path.endswith('.csv'):
            dataset = datasets.load_dataset('csv', data_files=dataset_path, split='train')
        elif os.path.isdir(dataset_path):
            dataset = datasets.load_from_disk(dataset_path)
            if isinstance(dataset, datasets.DatasetDict) and split in dataset:
                dataset = dataset[split]
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
    else:
        dataset = datasets.load_dataset(dataset_path, split=split)
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    return dataset


def validate_dataset(dataset: datasets.Dataset) -> None:
    """Validate dataset format."""
    required_columns = ["prompt", "chosen", "rejected"]
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    logger.info(f"Dataset validated. Columns: {dataset.column_names}")


# =============================================================================
# Main Training Function
# =============================================================================

def main(script_args: SDARDPOScriptArguments, training_args: SDARDPOConfig, model_args: ModelConfig):
    """Main training function."""
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"bf16: {training_args.bf16}"
    )
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")
    
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")
    
    # Load dataset
    logger.info("*** Loading dataset ***")
    train_dataset = load_dpo_dataset(
        script_args.dataset_path,
        script_args.dataset_train_split,
        script_args.max_samples,
    )
    validate_dataset(train_dataset)
    # train_dataset = train_dataset.select(range(128)) 
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    eval_dataset = None
    if script_args.dataset_eval_split is not None:
        eval_dataset = load_dpo_dataset(
            script_args.dataset_path,
            script_args.dataset_eval_split,
        )
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Load tokenizer
    logger.info("*** Loading tokenizer ***")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("*** Loading model ***")
    torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation if hasattr(model_args, 'attn_implementation') else "flash_attention_2",
    )
    
    # Load reference model (if not using PEFT)
    ref_model = None
    if not model_args.use_peft:
        logger.info("*** Loading reference model ***")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation if hasattr(model_args, 'attn_implementation') else "flash_attention_2",
        )
    
    # Get PEFT config if using PEFT
    peft_config = get_peft_config(model_args) if model_args.use_peft else None
    
    # Initialize trainer
    logger.info("*** Initializing SDAR DPO Trainer ***")
    logger.info(f"SDAR Config: block_length={script_args.block_length}, num_mc={script_args.num_mc}")
    
    trainer = SDARDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        # SDAR specific
        block_length=script_args.block_length,
        num_mc=script_args.num_mc,
        mask_token_id=script_args.mask_token_id,
        tool_resp_left=script_args.tool_resp_left,
        tool_resp_right=script_args.tool_resp_right,
    )
    
    # Training
    logger.info("*** Starting training ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Save model
    logger.info("*** Saving model ***")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    
    # Save model card on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            dataset_name=script_args.dataset_name,
            tags=["sdar", "dpo", "diffusion"],
        )
    
    logger.info("*** Training complete ***")


if __name__ == "__main__":
    parser = TrlParser((SDARDPOScriptArguments, SDARDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)