import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel
import pprint
import numpy as np
from typing import Callable, Literal


def get_auto_device() -> torch.device:
    """ 
    Select first available out of cuda -> mps -> cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def tokenizer_prompt_and_output(
        prompt_strs: list[str], 
        output_strs: list[str], 
        tokenizer: PreTrainedTokenizer 
) -> dict[str, torch.Tensor]:
    """
    prompt_strs: Prompt strings 
    output_strs: Output strings
    tokenizer: Tokenizer to use

    Let prompt_and_output_lens be a list containing lengths of concatenated tokenized prompt and output strings.

    Returns: A dict[str, Tensor] dictionary containing 
    - input_ids: Tensor of shape (batch_size, max(prompt_and_output_lens)-1) with tokenized prompt + output, with no tokens inbetween them, and padded on the right with zeros
    - labels: Tensor of shape (batch_size, max(prompt_and_output_lens)-1): shifted input ids (for next token prediction)
    - response_mask:  Tensor of shape (batch_size, max(prompt_and_output_lens)-1) aligned with labels, with 0 if token is in prompt and 1 if it is in output
    """
    prompt_tok = [tokenizer.encode(prompt_str) for prompt_str in prompt_strs]
    output_tok = [tokenizer.encode(output_str) for output_str in output_strs]
    prompt_lens = map(lambda x: len(x), prompt_tok)
    output_lens = map(lambda x: len(x), output_tok)
    full_len = [pl + ol for pl, ol in zip(prompt_lens, output_lens)]
    max_len = max(full_len)
    
    full_padded_toks = torch.asarray([pt + ot + [0 for _ in range(max_len - full_len)] for pt, ot, full_len in zip(prompt_tok, output_tok, full_len)])
    full_response_mask = torch.asarray([[0 for _ in range(len(pt))] + \
                                    [1 for _ in range(len(ot))] + \
                                    [0 for _ in range(max_len - full_len)] for pt, ot, full_len in zip(prompt_tok, output_tok, full_len)])

   

    return {
        "input_ids": full_padded_toks[:,:-1],
        "labels": full_padded_toks[:,1:], 
        "response_mask": full_response_mask[:,1:]
    }
def get_entropy(logprobs, dim):
    return - torch.sum(logprobs * torch.exp(logprobs), dim=dim)



def get_response_log_probs(
        model: PreTrainedModel, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor, 
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    model: A base model
    input_ids: a (batch_size, sequence_length, vocab_size) tensor 
    labels: a (batch_size, sequence_length, vocab_size) tensor


    Return dict[str, Tensor] with entries
    - log_probs: A tensor with dimensions (batch_size, sequence_length) showing model's log prob of label given inputs 
    - token_entropy (optional): A tensor with dimensions (batch_size, sequence_length) showing entropy of output dist for this token
    """


    outputs = model(input_ids)
    logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    out_dict = {"log_probs": target_log_probs}

    if return_token_entropy: 
        token_entropy = get_entropy(log_probs, dim=-1)
        out_dict["token_entropy"] = token_entropy

    return out_dict




def compute_rollout_rewards(
        reward_fn: Callable[[str, str], dict[str, float]], 
        rollout_responses: list[str], 
        repeated_ground_truths: list[str], 
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Helper funciton that computes rewards for a list of rollout responses, along with metadata for the reward components
    reward_fn: Scores the rollout responses against the ground truths, producing a dict with keys ``reward'', ``format_reward'', and ``answer_reward''
    rollout_responses: list[str] rollouts from the policy. length is rollout_batch_size = n_prompts_per_rollout_batch * group_size
    repeated_ground_truths: the ground gruths for the examples. the length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times

    Returns: 
    raw_rewards, a tuple of size (rollout_batch_size,). unnormalized rewards for each rollout response. 
    metadata: Reward statistics to log. Include the mean total and format rewards over the whole batch.
    """

    all_reward = [reward_fn(rollout, ground_truth) for (rollout, ground_truth) in zip(rollout_responses, repeated_ground_truths)]
    raw_reward = [d["reward"] for d in all_reward]
    format_reward = [d["format_reward"] for d in all_reward]

    mean_raw_reward = np.mean(raw_reward)
    mean_format_reward = np.mean(format_reward)

    return torch.tensor(raw_reward, dtype=torch.float32), {"mean_reward": mean_raw_reward, "mean_format_reward": mean_format_reward}

def compute_group_normalized_rewards_grpo(
        raw_rewards: torch.Tensor, 
        group_size: int, 
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6, 
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
): 
    """
    Compute advantages by applying the requested baseline and normalization within each group
    Args: 
    raw_rewards: (rollout_batch_size,) tensor. Unnormalized rewards for each rollout response, where rollout_batch_size = n_prompts_per_rollout_batch * group_size
    group_size: Number of responses per question (group)
    baseline: Literal["mean", "none"] (For now only support mean, which subtracts per-group reward. )
    advantage_eps: Small constant to avoid division by zero 
    advantage_normalizer: Literal["std", "none", "mean"] Support std, which divides by the per-group std. 

    Returns: 
    advantages: shape (rollout_batch_size,). Group normalized rewards for each rollout batch.
    metadata: dict 
    - reward_mean_per_rollout
    - reward_std_per_rollout
    - max_reward_per_rollout
    - min_reward_per_rollout
    """

    if baseline != "mean":
        raise NotImplementedError(f"Recieved baseline = {baseline}, not implemented")
    if advantage_normalizer != "std":
        raise NotImplementedError(f"Recieved advantage normalizer = {advantage_normalizer}, not implemented")

    batch_size = len(raw_rewards)
    assert(batch_size % group_size == 0)
    num_prompts = batch_size // group_size
    
    # reshape into a matrix with prompts vs. group
    rewards_mat = raw_rewards.reshape(num_prompts, group_size) 
    mean = torch.mean(rewards_mat, dim=1, keepdim = True)
    std = torch.std(rewards_mat, dim=1, keepdim = True)
    normalized = (rewards_mat - mean) / (std+advantage_eps)

    metadata = {
        "reward_mean_per_rollout": mean,
        "reward_std_per_rollout": std,
        "max_reward_per_rollout": torch.max(rewards_mat, dim=1, keepdim = True), 
        "min_reward_per_rollout": torch.min(rewards_mat, dim=1, keepdim = True), 
    }
    return normalized.view(-1), metadata

def compute_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss at each token using the requested importance reweighting method.

    Args:
    raw_rewards_or_advantages: (batch_size,) or (batch_size, 1) tensor.
        Raw reward or already-normalized advantage for each rollout response.
    policy_log_probs: (batch_size, sequence_length) tensor.
        Log probabilities of the tokens under the current policy.
    importance_reweighting_method: "none" for no reweighting, "noclip" for
        reweighting without clipping, "grpo" for token-level reweighting and
        clipping, or "gspo" for sequence-level reweighting and clipping.
    old_log_probs: (batch_size, sequence_length) tensor.
        Log probabilities under the rollout policy. Required for all
        importance reweighting methods except "none".
    cliprange: Clipping parameter epsilon. Required for "grpo" and "gspo".
    response_mask: Optional (batch_size, sequence_length) tensor selecting
        response tokens. Required for GSPO response-only sequence averaging.

    Returns:
    per_token_policy_gradient_loss: (batch_size, sequence_length) tensor.
        Per-token losses, to be aggregated in the training loop.
    metadata: Dictionary of diagnostic tensors, such as clipping statistics.
    """

    ## For now, assume no importance reweighting and ignore old-log-prob and clipping arguments
    assert(importance_reweighting_method == "none")
    assert(old_log_probs == None)
    assert(cliprange == None)
    assert (response_mask == None)

    batch_size = len(raw_rewards_or_advantages)
    per_token_loss = -policy_log_probs * raw_rewards_or_advantages.reshape(batch_size, 1) 
    
    return per_token_loss, {}


def aggregate_loss_across_microbatch_sequence(
        per_token_policy_gradient_loss: torch.Tensor,
        mask: torch.Tensor,
) -> torch.Tensor:
    """
    Aggregate per-token losses by averaging over included tokens within each
    sequence, then averaging across sequences.

    Args:
    per_token_policy_gradient_loss: (batch_size, sequence_length) tensor.
        Per-token policy gradient losses.
    mask: (batch_size, sequence_length) tensor.
        1 for tokens included in the loss and 0 for prompt or padding tokens.

    Returns:
    loss: Scalar tensor containing the average per-sequence loss.
        Sequences with no included tokens contribute zero. Preserves the
        computation graph so backward can be called.
    """
    seq_lens = mask.sum(dim=-1).clamp_min(1)
    per_sequence_loss = (per_token_policy_gradient_loss * mask).sum(dim=-1) / seq_lens
    return per_sequence_loss.mean()


def grpo_train_step(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None,
        reward_fn: Callable[[str, str], dict[str, float]],
        repeated_prompts: list[str],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """
    Perform one optimizer step using gradient accumulation over rollout microbatches.
    Currently supports the standard on-policy GRPO configuration with sequence normalization.

    Args:
    model: HuggingFace model to train.
    tokenizer: Tokenizer for the prompt and response strings.
    optimizer: Optimizer for the model.
    gradient_accumulation_steps: Number of equal-sized microbatches per optimizer step.
    max_grad_norm: Optional gradient norm limit, applied after accumulation.
    reward_fn: Scores each rollout against its ground truth.
    repeated_prompts: (rollout_batch_size,) list, with each prompt repeated group_size times.
    rollout_responses: (rollout_batch_size,) list of sampled responses.
    repeated_ground_truths: (rollout_batch_size,) list of corresponding ground truths.
    group_size: Number of responses per prompt. Groups must be contiguous.
    baseline: Baseline method passed to the reward normalization helper.
    advantage_eps: Small constant for reward normalization.
    advantage_normalizer: Advantage normalization method.
    importance_reweighting_method: Policy loss reweighting method.
    old_log_probs: Optional (rollout_batch_size, sequence_length) tensor of rollout log probabilities.
    cliprange: Optional clipping parameter for the policy loss.
    loss_normalization: Loss aggregation method. Currently supports "sequence".
    normalization_constant: Reserved for constant loss normalization.

    Returns:
    loss: Detached scalar tensor containing the accumulated batch loss.
    metadata: Reward statistics, loss diagnostics, and gradient norm before clipping.

    """
    rollout_batch_size = len(rollout_responses)
    assert(rollout_batch_size % group_size == 0)
    assert(importance_reweighting_method == "none")
    assert(old_log_probs == None)
    assert(cliprange == None)
    assert(loss_normalization == "sequence")
    assert(normalization_constant == None)
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    ## get rewards from rollouts
    raw_rewards, rewards_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)

    ## get group normalized rewards
    advantages, advantages_metadata = compute_group_normalized_rewards_grpo(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    ## get tokenization of prompts and responses
    tokenization_res = tokenizer_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = tokenization_res.values()

    ## compute gradient over several microbatches
    microbatch_size = rollout_batch_size // gradient_accumulation_steps

    ## Zero out grads before the microbatch
    optimizer.zero_grad(set_to_none=True)


    accumulated_loss = 0 
    for i in range(0, rollout_batch_size, gradient_accumulation_steps):
        start = i 
        end = min(i + gradient_accumulation_steps, rollout_batch_size)
        microbatch_size = end - start

        input_ids_microbatch = input_ids[start:end,:]
        labels_microbatch = labels[start:end,:]
        advantages_microbatch = advantages[start:end]
        response_mask_microbatch = response_mask[start:end,:]

        ## Run a forward pass on this microbatch
        policy_log_prob_microbatch = get_response_log_probs(model, input_ids_microbatch, labels_microbatch, return_token_entropy=False)["log_probs"]

        ## Compute the policy gradient loss against this microbatch
        per_token_policy_gradient_loss, policy_gradient_loss_metadata = compute_policy_gradient_loss(advantages_microbatch, policy_log_prob_microbatch, "none", response_mask=None)

        ## Compute full loss 
        loss = aggregate_loss_across_microbatch_sequence(per_token_policy_gradient_loss, response_mask_microbatch)

        ## In the computation of the loss, we divide by the total number of tokens seen
        ## In this computation that will be a factor of 1 / (microbatch_size)
        ## We multiply to get an overall factor of 1 / (rollout_batch_size)
        loss *=  (microbatch_size / rollout_batch_size)
        accumulated_loss += loss.detach()

        ## Perform backward pass    
        loss.backward()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    ## Update weights
    optimizer.step()
    optimizer.zero_grad()

    metadata = {}
    metadata["rewards_metadata"] = rewards_metadata
    metadata["advantages_metadata"] = advantages_metadata
    metadata["policy_gradient_loss_metadata"] = policy_gradient_loss_metadata

    metadata["grad_norm"] = total_norm
    return accumulated_loss, metadata


