import regex as re
import pprint
from tqdm import tqdm
import pretokenization_example

# Regex for pretokenizing
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def make_pair_freqs(freq_table: dict[tuple[bytes,...], int]) -> dict[tuple[bytes,...], int]:
	
	# create a pair frequency dictionary mapping 
	pair_freqs: dict[tuple(bytes, bytes), int] = {}
	for key, val in freq_table.items():
		for j in range(len(key)-1):
			b1 = key[j]
			b2 = key[j+1]
			count = pair_freqs.get((b1, b2), 0)
			pair_freqs[(b1, b2)] = count+val

	return pair_freqs



def do_merge(
	freq_table: dict[tuple[bytes,...], int], 
	max_key: tuple[bytes, bytes]):
	
	# go through pretokens and perform merges
	new_table = {}
	for k, v in freq_table.items(): # k is a tuple[bytes,...]
		merged_k = []
		j = 0
		while j < len(k):
			if j < len(k) - 1 and (k[j], k[j+1]) == max_key:
				merged_k.append(k[j] + k[j+1])
				j += 2
			else:
				merged_k.append(k[j])
				j += 1

		new_table[tuple(merged_k)] = v

	return new_table


def make_freq_table(doc: bytes):
	decoded_doc = doc.decode('utf-8')
	pretokenized = re.finditer(PAT, decoded_doc)

	# create frequency dictionary 
	freq_table: dict[tuple[bytes,...], int] = {}

	for match in pretokenized:
		tok = match.group(0)
		key = tuple(bytes([c]) for c in tok.encode('utf-8'))
		val = freq_table.get(key, 0)
		freq_table[key] = val + 1

	# print("freq_table:")
	# pprint.pp(freq_table)

"""
Uses byte pair encoding to learn a tokenizer from a corpus
Outputs 
vocab: dict[int, bytes], the tokenizer vocabulary, mapping from int to bytes
merges: list[tuple[bytes, bytes]], a list of BPE merges from training
"""
def train_bpe(
	input_path: str, # Path to a text file with BPE tokenizer training data
	vocab_size: int, # Max vocab size (including initial byte vocab, 
					 # vocab from merging, and special tokens)
	special_tokens: list[str], # A list of special strings to be added to vocab. 
							   # These are treated as hard boundaries during 
							   # training
	) -> tuple[
		dict[int, bytes], # vocab: the tokenizer vocabulary,
						  #  mapping from int to bytes
		list[tuple[bytes, bytes]]]: # merges: a list of BPE merges from training

	with open(input_path, "rb") as f:
	    num_processes = 4
	    boundaries = pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

	    # The following is a serial implementation, but you can parallelize this
	    # by sending each start/end pair to a set of processes.
	    for start, end in zip(boundaries[:-1], boundaries[1:]):
	        f.seek(start)
	        chunk = f.read(end - start).decode("utf-8", errors="ignore")
	        freq_table = make_freq_table(chunk)
	        # Run pre-tokenization on your chunk and store the counts for each pre-token




	

	# create merge list
	merges: list[tuple[bytes, bytes]] = []
	new_tokens = []

	for i in tqdm(range(vocab_size - 256 - len(special_tokens))):
		pair_freqs = make_pair_freqs(freq_table)

		# print("pair_freqs:")
		# pprint.pp(pair_freqs)

		# a list of tuple(int, int, int)
		sort_vals = [(val, key[0], key[1]) for key, val in pair_freqs.items()]

		max_val = max(sort_vals)
		max_key = (max_val[1], max_val[2])
		# print('merge:', max_key)
		merges.append(max_key)
		new_tokens.append(max_key[0] + max_key[1])

		freq_table = do_merge(freq_table, max_key)


	vocab: dict[int, bytes] = {}
	ind = 0 

	# add special tokens to vocab
	for v in special_tokens:
		vocab[ind] = v
		ind += 1

	# add normal chars to vocab
	for j in range(256):
		vocab[ind] = bytes([j])
		ind += 1

	# add merges to vocab
	for v in new_tokens:
		vocab[ind] = v
		ind += 1

	return vocab, merges

tiny_stories_valid = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-valid.txt'
vocab, merges = train_bpe(tiny_stories_valid, 10000, ["|<endoffile>|"])

print("Vocab:")
pprint.pp(vocab)

print("\nMerges:")
pprint.pp(merges)
