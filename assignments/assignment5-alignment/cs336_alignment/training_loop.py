import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel
import pprint
import numpy as np
from typing import Callable, Literal
from cs336_alignment import train_util, checkpoint, vllm_utils
from dataclasses import dataclass, asdict
import wandb
import json

""" 
Example hyperparameters

n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 9
gradient_accumulation_steps = 32
sampling_temperature = 1.0 
sampling_max_tokens = 512
max_grad_norm = 1.0
optimizer = torch.optim.AdamW(
    policy.parameters(), lr = learning_rate, betas = (0.9, 0.95), weight_decay = 0.0)
steps_to_validation = 10
num_validation_samples = 1024
steps_to_log_training_rollouts = 40

Note that rollout_batch_size and train_batch_size count responses, not prompts. 
taking these = 256 means 32 prompts with 8 rollouts each

"""

@dataclass
class SamplingHyperparameters: 
    rollout_batch_size: int = 256 # rollout_batch_size = num_prompts * group_size 
    group_size: int = 8 # num responses per prompt
    sampling_temperature: float = 1.0 # temperature for generation
    sampling_max_tokens: int = 512 # stop sampling here

@dataclass
class TrainingHyperparameters: 
    n_val_examples: int = 1024 # number of samples to use for validation 

    # for on-policy RL, each rollout step 
    # - samples prompts, 
    # - generates a rollout batch, 
    # - computes rewards and advantages, and 
    # - updates the policy
    num_rollout_steps: int = 200 

    train_batch_size: int = 256 # sample as rollout_batch_size for on-policy RL 
    gradient_accumulation_steps: int = 32 # how many microbatches to split up one training batch into

    ## AdamW parameters
    learning_rate: float = 1e-5 # learning rate for optimizer
    betas: tuple[float, float] = (0.9, 0.95) # betas for AdamW 
    weight_decay: float = 0.0 # weight decay for optimizer
    max_grad_norm: float | None = 1.0 # gradient clipping
    adamw_eps: float = 1e-8

    # perform validation after this many rollout steps 
    steps_to_validation: int = 10 

    # log (some) training rollouts after this many steps
    steps_to_log_training_rollouts: int = 40




def run_training_loop(
        model_id_or_dir, 
        system_prompt: Literal["question_only, r1_zero, r1_zero_three_shot_gsm8k"], 
        training_set_file_path, 
        validation_set_file_path, 
        sampling_hyperparameters, 
        training_hyperparameters, 
        device = None
): 
    """
    model_name: Model to load using checkpoint.get_model_and_tokenizer
    system_prompt: Selects a prompt to use for generation (one of 3 options)
    training_set_file_path: Path to a jsonl file, each line is a dict with question and answer
    validation_set_file_path: Use for validation
    sampling_hyperparameters: 

    Runs a GRPO training loop

    First, initialize 
    - vLLM server
    - wandb logging
    - datasets
    - model
    - optimizer

    Then, run training loop:
    - Sync weights with vLLM server
    - Produce training rollouts
    - Take policy gradients 
    
    While training, 
    - Periodically check performance on validation set 
    - Log generations
    """

    ## If device = None, set automatically
    if device == None: 
        device = train_util.get_auto_device()

    ## Initialize vLLM server
    vllm_server = vllm_utils.VLLMServer(model_id = model_id_or_dir)
    vllm_server.start()

    ## Initialize wandb logging
    wandb_run = wandb.init(
        project="cs336-assignment5", 
        config = {
            "model_id": str(model_id_or_dir), 
            "sampling": asdict(sampling_hyperparameters), 
            "training": asdict(training_hyperparameters) 
        }
    )
    wandb_run.log()

    ## Load training and validation datasets 
    with open(training_set_file_path, encoding="utf-8") as f:
        training_data = [json.loads(line) for line in f if line.strip()]

    with open(validation_set_file_path, encoding="utf-8") as f:
        validation_data = [json.loads(line) for line in f if line.strip()]


    ## Initialize model
    model, tokenizer = checkpoint.get_model_and_tokenizer(model, device)

    ## Initialize optimizer
    opt = torch.optim.AdamW(model.parameters(), 
                            lr = training_hyperparameters.learning_rate, 
                            betas = training_hyperparameters.betas, 
                            eps = 1e-8,
                            weight_decay = training_hyperparameters.weight_decay)

    