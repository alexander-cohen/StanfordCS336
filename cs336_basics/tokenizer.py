import regex as re

# Regex for pretokenizing
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

"""
Uses byte pair encoding to learn a tokenizer from a corpus
Outputs 
vocab: dict[int, bytes], the tokenizer vocabulary, mapping from int to bytes
merges: list[tuple[bytes, bytes]], a list of BPE merges from training
"""
def train_bpe(
	input_path: str, # Path to a text file with BPE tokenizer training data
	vocab_size: int, # Max vocab size (including initial byte vocab, vocab from merging, and special tokens)
	special_tokens: list[str], # A list of special strings to be added to vocab. These are treated as hard boundaries during training
	) -> tuple[
		dict[int, bytes], # vocab: the tokenizer vocabulary, mapping from int to bytes
		list[tuple[bytes, bytes]]]: # merges: a list of BPE merges from training


	with open(input_path, "rb") as f:
		return