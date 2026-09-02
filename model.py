"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel as flm

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # Goal: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    return flm.from_pretrained(max_seq_length=max_seq_length,load_in_4bit=model_name)

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # Goal: sum p.numel() over every parameter tensor in the module
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
import bitsandbytes as bnb

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # Goal: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    
    for submodule in model.modules():
        if isinstance(submodule,bnb.nn.Linear4bit):
            return True
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # Goal: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # Goal: return the list of attention projection module names LoRA should adapt
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # Goal: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    if target_modules is None:
        target_modules = get_lora_target_modules()
    
    return flm.get_peft_model(
        model,
        r=r,
        target_modules = target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none"
    )

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # Goal: sum p.numel() over model.parameters() where requires_grad is True
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # Goal: return the fraction of parameters that are trainable.
    if total_count == 0:
        return 0.0

    return float(trainable_count / total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # Goal: return a tiny hand-written list of instruction/response example dicts.
    examples = [
        {
            "instruction": "What is the capital of Japan?",
            "response": "The capital of Japan is Tokyo."
        },
        {
            "instruction": "Write a Python function to reverse a string.",
            "response": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "instruction": "Write a haiku about a robot learning to love.",
            "response": "Cold gears start to turn,\nA spark ignites in the wire,\nMetal finds a heart."
        },
        {
            "instruction": "Calculate 15% of 80 and explain the steps.",
            "response": "First, find 10% of 80, which is 8. Then, find 5% of 80, which is half of 8, so 4. Finally, add them together: 8 + 4 = 12. Therefore, 15% of 80 is 12."
        },
        {
            "instruction": "Summarize the water cycle in three simple steps.",
            "response": "1. Evaporation: Water turns into vapor and rises. 2. Condensation: Vapor cools and forms clouds. 3. Precipitation: Water falls back to earth as rain or snow."
        },
        {
            "instruction": "Classify the word 'quickly' as a noun, verb, adjective, or adverb.",
            "response": "The word 'quickly' is an adverb."
        }
    ]
    
    return examples

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # Goal: combine example['instruction'] and example['response'] into one string
    instruction = example["instruction"]
    response = example["response"]
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # Goal: apply format_instruction_example to every example and return the list
    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # Goal: return a datasets.Dataset with one 'text' column holding the given strings
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # Goal: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)["input_ids"]

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # Goal: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
import torch
import transformers

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    # Goal: build TrainingArguments with batch size 1, given max_steps, given lr, bf16 or fp16.
    
    if torch.cuda.is_bf16_supported():
        return transformers.TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            max_steps = max_steps,
            learning_rate = learning_rate,
            output_dir = output_dir,
            bf16 = True,
            logging_steps = 1,
            optim = 'adamw_8bit')

    return transformers.TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            max_steps = max_steps,
            learning_rate = learning_rate,
            output_dir = output_dir,
            fp16 = True,
            logging_steps = 1,
            optim = 'adamw_8bit')

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

