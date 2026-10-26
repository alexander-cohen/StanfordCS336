print("Started python")
import torch
import torch.nn.functional as F
import pprint
import numpy as np
from typing import Callable, Literal
from dataclasses import dataclass, asdict
import json
import wandb
import random
print("Imported torch, numpy, pp, wandb, etc")
from transformers import PreTrainedTokenizer, PreTrainedModel
print("imported transformers")
from cs336_alignment import train_util, checkpoint, vllm_utils, drgrpo_grader
print("import train_util, checkpoint, vllm_utils")



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
    num_prompts: int = None
    sampling_temperature: float = 1.0 # temperature for generation
    sampling_max_tokens: int = 512 # stop sampling here

    def __post_init__(self) -> None:
        assert(self.rollout_batch_size % self.group_size == 0)
        self.num_prompts = self.rollout_batch_size // self.group_size

@dataclass
class TrainingHyperparameters: 
    # for on-policy RL, each rollout step 
    # - samples prompts, 
    # - generates a rollout batch, 
    # - computes rewards and advantages, and 
    # - updates the policy
    num_rollout_steps: int = 200 

    train_batch_size: int = 256 # sample as rollout_batch_size for on-policy RL 
    gradient_accumulation_steps: int = 16 # size of microbatch

    ## AdamW parameters
    learning_rate: float = 1e-5 # learning rate for optimizer
    betas: tuple[float, float] = (0.9, 0.95) # betas for AdamW 
    weight_decay: float = 0.0 # weight decay for optimizer
    max_grad_norm: float | None = 1.0 # gradient clipping
    adamw_eps: float = 1e-8

    # perform validation after this many rollout steps 
    steps_to_validation: int = 10 
    n_val_examples: int = 1024 # number of samples to use for validation 

    # log (some) training rollouts after this many steps
    steps_to_log_training_rollouts: int = 40


def run_rl_rollout():
    """
    - Sync weights with vLLM server
    - Produce training rollouts
    - Take policy gradients 
    """
    pass

def extract_answer_gsm8k(answer):
    return answer.split("#### ")[-1]

def perform_validation():
    pass

def run_training_loop(
        model_id_or_dir, 
        system_prompt: Literal["question_only, r1_zero, r1_zero_three_shot_gsm8k"], 
        training_set_file_path, 
        validation_set_file_path, 
        sampling_hyperparameters : SamplingHyperparameters, 
        training_hyperparameters : TrainingHyperparameters, 
        policy_device = "cuda:0", 
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


    ## Retrieve system prompt text
    system_prompt_path = f"cs336_alignment/prompts/{system_prompt}.prompt"
    with open(system_prompt_path, "r") as f: 
        system_prompt_text = f.read()

    ## Initialize model
    model, tokenizer = checkpoint.get_model_and_tokenizer(model_id_or_dir, policy_device)
    print("Loaded model")

    ## Initialize vLLM server
    vllm_server = vllm_utils.VLLMServer(model_id=model_id_or_dir, load_format="dummy")
    vllm_server.start()
    print("Started vLLM server")

    vllm_server.init_weight_sync(policy_device)

    # Initialize wandb logging
    wandb_run = wandb.init(
        project="cs336-assignment5", 
        config = {
            "model_id": str(model_id_or_dir), 
            "sampling": asdict(sampling_hyperparameters), 
            "training": asdict(training_hyperparameters), 
            "system_prompt": system_prompt
        }
    )
    print("Initialized wandb login")

    ## Load training and validation datasets 
    ## Is a list of dicts, with entries "question" and "answer"
    with open(training_set_file_path, encoding="utf-8") as f:
        training_data = [json.loads(line) for line in f if line.strip()]

    with open(validation_set_file_path, encoding="utf-8") as f:
        validation_data = [json.loads(line) for line in f if line.strip()]
    print("Loaded training and validation data")

    ## Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                            lr = training_hyperparameters.learning_rate, 
                            betas = training_hyperparameters.betas, 
                            eps = 1e-8,
                            weight_decay = training_hyperparameters.weight_decay)
    print("Initialized optimizer")

    ## Initialize vLLM sampling params 
    sampling_params = {
        "temperature": sampling_hyperparameters.sampling_temperature,
        "max_tokens": sampling_hyperparameters.sampling_max_tokens,
        "n": sampling_hyperparameters.group_size, 
        "seed": 0,
    }
    print(f"Sampling params: {sampling_params}")

    ## Initialize reward function
    if system_prompt == "r1_zero" or system_prompt == "r1_zero_three_short_gsm8k": 
        reward_fn = drgrpo_grader.r1_zero_reward_fn
    elif system_prompt == "question_only": 
        reward_fn = drgrpo_grader.question_only_reward_fn
    else: 
        raise ValueError(f"system_prompt = {system_prompt} not recognized")

    
    for it in range(training_hyperparameters.num_rollout_steps):
        ### Perform a training iteration
        ## Send model weights to vLLM server
        vllm_server.sync_policy_weights(model)
        # print("Synced weights")

        ## Select prompts for batch
        training_data_batch = random.sample(training_data, sampling_hyperparameters.num_prompts)
        questions = [p["question"] for p in training_data_batch]
        prompts = [system_prompt_text + "\n" + q for q in questions]
        repeated_questions = [q for _ in range(sampling_hyperparameters.group_size) for q in questions]
        repeated_prompts = [p for _ in range(sampling_hyperparameters.group_size) for p in prompts]
        answers = [p["answer"] for p in training_data_batch]
        gt_answers = [extract_answer_gsm8k(a) for a in answers]
        repeated_gt_answers = [a for _ in range(sampling_hyperparameters.group_size) for a in gt_answers]
        # print(f"Prompts: {prompts}")
        # print(f"gt answers: {gt_answers}")


        ## Generate G rollouts per prompt
        ## vllm_response is a num_prompts * group_size list of VLLMCompletion objects (G responses to first prompt, followed by G for second, etc)
        ## Each completion object has entries 
        ## - finish_reason
        ## - text (of response)
        ## - token_ids (of response)
        vllm_response = vllm_server.generate_completions(prompts, sampling_params, batch_size = None)
        rollouts = [r.text for r in vllm_response]

        # print(f"Rollouts: {rollouts}")

        ## Perform training step ! 
        acc_loss, metadata = train_util.grpo_train_step(
            model, 
            tokenizer, 
            optimizer, 
            training_hyperparameters.gradient_accumulation_steps, 
            training_hyperparameters.max_grad_norm, 
            reward_fn, 
            repeated_prompts,
            rollouts,
            repeated_gt_answers, 
            sampling_hyperparameters.group_size
        )

        wandb_run.log({"train/accumulated-loss": acc_loss}, step=it)
        wandb_run.log(metadata, step=it)

        ## Log rollouts 
        if it % training_hyperparameters.steps_to_log_training_rollouts == 0:
            rollout_table = wandb.Table(columns=["prompt", "ground_truth", "response"])
            for q, gt, a in zip(repeated_questions, repeated_gt_answers, rollouts): 
                rollout_table.add_data(q, gt, a)

            wandb_run.log({"eval/rollouts": rollout_table}, step=it)

        ## TODO: Perform validation 

    wandb_run.finish()
        



def vllm_test():
    print("Running vllm test")
    print("Starting vllm server")
    vllm_server = vllm_utils.VLLMServer(gpu = 0, model_id = checkpoint.OLMO)
    print("Created vllm server")
    vllm_server.start()
    print("Started vllm")

    sampling_params = {
        "temperature": 1.0, 
        "max_tokens": 100, 
        "n": 5, 
        "seed": 0,
    }


    
    res = vllm_server.generate_completions(["the capital of nevada is ", "Once upon a time a girl was playing basketball, and"], sampling_params)
    print("Computed generations")

    print("Results:")
    for c in res:
        print(c.text)
        print("\n\n--------------\n\n")

    vllm_server.stop()

if __name__ == "__main__":
    # vllm_test()

    print("Testing run_training_loop script")
    sampling_hyperparams = SamplingHyperparameters()
    training_hyperparams = TrainingHyperparameters()
    run_training_loop(checkpoint.OLMO, 
                      "r1_zero", 
                      "data/gsm8k/test.jsonl", 
                      "data/gsm8k/train.jsonl", 
                      sampling_hyperparams, 
                      training_hyperparams)


## srun command: 
# srun --account=torch_pr_1323_general --nodes=1 --ntasks=1      --gres=gpu:1 --mem=16G --time=00:15:00      uv run --env-file .env --extra gpu      python -u -m cs336_alignment.training_loop
