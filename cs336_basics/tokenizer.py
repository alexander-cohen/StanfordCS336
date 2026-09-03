import regex as re
import pprint
from tqdm import tqdm
import cs336_basics.pretokenization_example
import concurrent.futures 
import os 
import pathlib
from typing import BinaryIO
from itertools import repeat
from deepdiff import DeepDiff

# Regex for pretokenizing
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def make_pair_freqs(freq_table: dict[tuple[bytes,...], int]) -> dict[tuple[bytes, bytes], int]:
    
    # create a pair frequency dictionary mapping 
    pair_freqs: dict[tuple[bytes, bytes], int] = {}
    for key, val in freq_table.items():
        for j in range(len(key)-1):
            b1 = key[j]
            b2 = key[j+1]
            count = pair_freqs.get((b1, b2), 0)
            pair_freqs[(b1, b2)] = count+val

    return pair_freqs

def soft_add(the_dict, key, inc = 1):
    v = the_dict.get(key, 0)
    the_dict[key] = v+inc

def do_merge(
    freq_table: dict[tuple[bytes,...], int], 
    pair_freqs: dict[tuple[bytes, bytes], int],
    max_key: tuple[bytes, bytes]) -> dict[tuple[bytes,...], int]:


    new_tok = max_key[0] + max_key[1]
    pair_freqs[max_key] = 0
    
    # go through pretokens and perform merges
    # new_table: dict[tuple[bytes,...], int] = {}

    for k, v in list(freq_table.items()): # k is a tuple[bytes,...]
        merged_k = []
        j = 0
        while j < len(k):
            if j < len(k) - 1 and (k[j], k[j+1]) == max_key:
                merged_k.append(k[j] + k[j+1])
                j += 2
            else:
                merged_k.append(k[j])
                j += 1

        merged_k = tuple(merged_k)
        if merged_k != k:  
            # subtract old pair freqs 
            for j in range(len(merged_k)-1):
                b1 = merged_k[j]
                b2 = merged_k[j+1]
                soft_add(pair_freqs, (b1, b2), +v)
             
            # add new pair freqs
            for j in range(len(k)-1):
                b1 = k[j]
                b2 = k[j+1]
                soft_add(pair_freqs, (b1, b2), -v)

            freq_table.pop(k, None)
            freq_table[merged_k] = v


    # return new_table


def add_freq_table(doc: str, freq_table: dict[tuple[bytes,...], int] = None):
    

    # # create frequency dictionary 
    # if freq_table == None:
    #     freq_table = {}

    pretokenized = re.finditer(PAT, doc)
    for match in pretokenized:
        tok = match.group(0)
        key = tuple(bytes([c]) for c in tok.encode('utf-8'))
        val = freq_table.get(key, 0)
        freq_table[key] = val + 1

    # return freq_table

def make_chunk_freq_table(
    input_path: BinaryIO, 
    delim: str,
    start: int, 
    end: int) -> dict[tuple[bytes,...], int]:
    
    freq_table = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        split_chunks = re.split(delim, chunk)

        for sc in split_chunks:
            add_freq_table(sc, freq_table)

    return freq_table

def make_freq_table_multiprocessed(input_path: str, delim: str, num_processes = 4):
    with open(input_path, "rb") as f:
        boundaries = cs336_basics.pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        num_pieces = len(boundaries)-1

        with concurrent.futures.ProcessPoolExecutor(max_workers = num_processes) as executor:
            freq_tables = executor.map(make_chunk_freq_table, 
                                        repeat(input_path, num_pieces),
                                        repeat(delim, num_pieces),
                                        boundaries[:-1], # boundary starts
                                        boundaries[1:]) # boundary ends

            combined_table: dict[tuple[bytes,...], int] = {}
            for table in freq_tables:
                for (k, v) in table.items():
                    count = combined_table.get(k, 0)
                    combined_table[k] = count + v


    return combined_table

def make_freq_table_basic(input_path: str):
    with open(input_path, "rb") as f:
        boundaries = pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        inputs = [(input_path, start, end) for (start, end) in zip(boundaries[:-1], boundaries[1:])]
        input_paths, starts, ends = zip(*inputs)
        with concurrent.futures.ProcessPoolExecutor(max_workers = num_processes) as executor:
            freq_tables = executor.map(make_chunk_freq_table, input_paths, starts, ends)
            combined_table: dict[tuple[bytes,...], int] = {}
            for table in freq_tables:
                for (k, v) in table.items():
                    count = combined_table.get(k, 0)
                    combined_table[k] = count + v


    return combined_table

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
    
    delim = "|".join([re.escape(t) for t in special_tokens])
    freq_table = make_freq_table_multiprocessed(input_path, delim, 4)

    # create merge list
    merges: list[tuple[bytes, bytes]] = []
    new_tokens = []

    pair_freqs = make_pair_freqs(freq_table)

    for i in tqdm(range(vocab_size - 256 - len(special_tokens))):
        # pair_freqs = make_pair_freqs(freq_table)

        # print("pair_freqs:")
        # pprint.pp(pair_freqs)

        # a list of tuple(int, int, int)
        sort_vals = [(val, key[0], key[1]) for key, val in pair_freqs.items()]

        max_val = max(sort_vals)
        max_key = (max_val[1], max_val[2])
        # print('merge:', max_key)
        merges.append(max_key)
        new_tokens.append(max_key[0] + max_key[1])

        do_merge(freq_table, pair_freqs, max_key)


        # hand_made_pair_freqs = make_pair_freqs(freq_table)
        # filtered_dict = {k: v for k, v in pair_freqs.items() if v != 0}
        # print("Freq table:")
        # pprint.pp(freq_table)

        # if filtered_dict != hand_made_pair_freqs:
        #     print("Max key:", max_key)

        #     # Find the difference
        #     diff = DeepDiff(pair_freqs, hand_made_pair_freqs)

        # #     # print("Pair freqs:")
        # #     # pprint.pp(pair_freqs)

        # #     # print("handmade pair freqs:")
        # #     # pprint.pp(hand_made_pair_freqs)

        #     # Pretty print the result
        #     print(diff.pretty())

            # raise AssertionError("pair freqs wrong")


    vocab: dict[int, bytes] = {}
    ind = 0 

    # add special tokens to vocab
    for v in special_tokens:
        vocab[ind] = v.encode('utf-8')
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

def main():
    tiny_stories_valid = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-valid.txt'
    tiny_stories_train = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-train.txt'
    test_corpus = """/Users/alexandercohen/Documents/Research After 2025/Stanford CS336/CS336-Spring-2026/assignments/assignment1-basics/text/test_corpus.txt"""
    special_tokens = ["|<endoffile>|"]

    delim = "|".join([re.escape(t) for t in special_tokens])

    # freq_table = make_freq_table_multiprocessed(tiny_stories_valid, delim, 16)
    # freq_table_words = sorted([(b''.join(k), v) for k, v in freq_table.items()], key = lambda item: -item[1])

    # pprint.pp(freq_table_words)
    # print(len(freq_table_words))

    # # print("Freq table:")
    # # pprint.pp(freq_table)
    # return

    vocab, merges = train_bpe(test_corpus, 500, special_tokens)

    print("Vocab:")
    pprint.pp(vocab)

    print("\nMerges:")
    pprint.pp(merges)

if __name__ == '__main__':
    main()


