
import cs336_basics.transformer as transformer
import cs336_basics.tokenizer as tokenizer
import cs336_basics.nn_utils as nn_utils
import torch

def decode(model: transformer.Transformer, 
		   tok: tokenizer.Tokenizer,
		   input: str, 
		   max_toks = 1000, 
		   temperature = None, 
		   topp = None):

	enc = tok.encode(input)
	space = tok.encode(" ")[0]
	EOT = tok.encode("<|endoftext|>")[0]
	generated = enc.copy()
	for j in range(max_toks):
		context_unpadded = generated[-model.context_length:]
		pad_add = model.context_length - len(context_unpadded)
		context = [space] * pad_add + context_unpadded
		logits = model(torch.tensor(context))[-1:]

		if temperature is not None:
			probs = nn_utils.softmax(logits / temperature, -1)
			next_token = torch.multinomial(probs, num_samples=1)[0].item()
		elif topp is not None:
			probs = nn_utils.softmax(logits, -1)
			sorted_probs, sorted_inds = torch.sort(probs, descending=True)
			cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
			keep = cumulative_probs - sorted_probs < topp
			truncated_probs = sorted_probs * keep
			truncated_probs = truncated_probs / truncated_probs.sum()
			sampled_sorted_idx = torch.multinomial(truncated_probs, num_samples=1)
			next_token = sorted_inds[sampled_sorted_idx][0].item()
		else:
			probs = nn_utils.softmax(logits, -1)
			next_token = torch.multinomial(probs, num_samples=1)[0].item()

		generated.append(next_token)
		if next_token == EOT:
			break

	return generated

