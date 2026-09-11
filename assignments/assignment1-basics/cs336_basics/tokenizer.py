import argparse
import cProfile
import pickle
import pstats
import regex as re
import pprint
from tqdm import tqdm
import cs336_basics.pretokenization_example
import concurrent.futures 
import os 
import pathlib
from typing import BinaryIO
from itertools import repeat
from collections.abc import Iterable, Iterator
from pathlib import Path
import time
import uuid
import os
# import diskcache
import multiprocessing
import random
import math
import numpy as np
# from deepdiff import DeepDiff

# Regex for pretokenizing
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
tiny_stories_valid = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-valid.txt'
tiny_stories_valid_tokd = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-valid_tokd.npy'
tiny_stories_train = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-train.txt'
tiny_stories_train_tokd = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-train_tokd.npy'
test_corpus = """/Users/alexandercohen/Documents/Research After 2025/Stanford CS336/CS336-Spring-2026/assignments/assignment1-basics/text/test_corpus.txt"""
owt_train = """/Users/alexandercohen/Local Documents/data/owt_train.txt"""
owt_train_tokd = """/Users/alexandercohen/Local Documents/data/owt_train_tokd.npy"""
owt_train_tokd_prepend = """/Users/alexandercohen/Local Documents/data/owt_train_tokd"""
owt_valid = """/Users/alexandercohen/Local Documents/data/owt_valid.txt"""

owt_valid_tokd = """/Users/alexandercohen/Local Documents/data/owt_valid_tokd.npy"""
cache_dir = """/Users/alexandercohen/Local Documents/cache_dir"""
EOT = "<|endoftext|>"

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

def apply_merge_to_tup(
    byte_tup: tuple[bytes,...], 
    merge: tuple[bytes, bytes]) -> tuple[bytes,...]:
    
    merged = []
    j = 0
    while j < len(byte_tup):
        if j < len(byte_tup) - 1 and (byte_tup[j], byte_tup[j+1]) == merge:
            merged.append(byte_tup[j] + byte_tup[j+1])
            j += 2
        else:
            merged.append(byte_tup[j])
            j += 1

    return tuple(merged)


def do_merge(
    freq_table: dict[tuple[bytes,...], int], 
    tok_to_pretok: dict[bytes, set[tuple[bytes,...]]],
    pair_freqs: dict[tuple[bytes, bytes], int],
    max_key: tuple[bytes, bytes], 
    verbose = False) -> dict[tuple[bytes,...], int]:


    new_tok = max_key[0] + max_key[1]
    pair_freqs[max_key] = 0

    relevant_pretoks = tok_to_pretok[max_key[0]] & tok_to_pretok[max_key[1]]

    if verbose:
        print("freq_table:")
        pprint.pp(freq_table)

        print("tok_to_pretok:")
        pprint.pp(tok_to_pretok)

        print("pair_freqs:")
        pprint.pp(pair_freqs)

        print("max_key:")
        pprint.pp(max_key)

        print("relevant_pretoks:")
        pprint.pp(relevant_pretoks)

    for k in relevant_pretoks:
        v = freq_table[k]
        merged_k = apply_merge_to_tup(k, max_key)
        # j = 0
        # while j < len(k):
        #     if j < len(k) - 1 and (k[j], k[j+1]) == max_key:
        #         merged_k.append(k[j] + k[j+1])
        #         j += 2
        #     else:
        #         merged_k.append(k[j])
        #         j += 1
        # merged_k = tuple(merged_k)

        if merged_k != k:  
            # remove old tok to pretok pointers
            for t in k:
                tok_to_pretok[t].discard(k)
            
            # add new tok to pretok pointers
            for t in merged_k:
                tok_to_pretok.setdefault(t, set()).add(merged_k)    
            



            # delete old pair freqs
            for j in range(len(k)-1):
                b1 = k[j]
                b2 = k[j+1]
                soft_add(pair_freqs, (b1, b2), -v)

            # add new pair freqs 
            for j in range(len(merged_k)-1):
                b1 = merged_k[j]
                b2 = merged_k[j+1]
                soft_add(pair_freqs, (b1, b2), +v)
             
            

            freq_table.pop(k, None)
            freq_table[merged_k] = v
    


def add_freq_table(doc: str, freq_table: dict[tuple[bytes,...], int] = None):
    pretokenized = re.finditer(PAT, doc)
    for match in pretokenized:
        tok = match.group(0)
        key = tuple(bytes([c]) for c in tok.encode('utf-8'))
        soft_add(freq_table, key)

def make_chunk_freq_table(
    input_path: str, 
    delim: str,
    start: int, 
    end: int, 
    position: int,
    use_disk = False) -> dict[tuple[bytes,...], int]:
    
    if use_disk:
        freq_table = diskcache.Index(f'{cache_dir}/cache_{uuid.uuid4()}')
    else:
        freq_table = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        split_chunks = re.split(delim, chunk)

        for sc in tqdm(split_chunks, position=position, leave=False, desc=f"Chunk {position}"):
            add_freq_table(sc, freq_table)

    return freq_table

def make_freq_table_multiprocessed(input_path: str, delim: str, num_processes = 8, num_pieces = 8, use_disk = False):
    with open(input_path, "rb") as f:
        boundaries = cs336_basics.pretokenization_example.find_chunk_boundaries(f, num_pieces, EOT.encode('utf-8'))

        num_pieces = len(boundaries)-1
        print("Num boundaries:", num_pieces)

        # <-- Create a picklable multiprocessing manager lock
        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        mp_lock = manager.RLock()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers = num_processes,
            initializer = tqdm.set_lock,
            initargs = (mp_lock,)  # <-- Pass the picklable manager lock instead of tqdm.get_lock()
        ) as executor:
            
            futures = [
                executor.submit(make_chunk_freq_table, input_path, delim, start, end, idx, use_disk)
                for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
            ]
            freq_tables = [f.result() for f in futures]

            if use_disk:
                combined_table = diskcache.Index(f'{cache_dir}/cache_{uuid.uuid4()}')
            else:
                combined_table = {}

            for table in freq_tables:
                for (k, v) in table.items():
                    count = combined_table.get(k, 0)
                    combined_table[k] = count + v

        # Clean up the manager context once the pool shuts down
        manager.shutdown()

    return combined_table

# def make_chunk_freq_table(
#     input_path: BinaryIO, 
#     delim: str,
#     start: int, 
#     end: int, 
#     use_disk = False) -> dict[tuple[bytes,...], int]:
    
#     if use_disk:
#         freq_table = diskcache.Index(f'{cache_dir}/cache_{uuid.uuid4()}')
#     else:
#         freq_table = {}
#     with open(input_path, "rb") as f:
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         split_chunks = re.split(delim, chunk)

#         for sc in tqdm(split_chunks):
#             add_freq_table(sc, freq_table)

#     return freq_table

# def make_freq_table_multiprocessed(input_path: str, delim: str, num_processes = 8, use_disk = False):
#     with open(input_path, "rb") as f:
#         boundaries = cs336_basics.pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#         num_pieces = len(boundaries)-1

#         print("Num boundaries:", num_pieces)

#         with concurrent.futures.ProcessPoolExecutor(max_workers = num_processes) as executor:
#             freq_tables = executor.map(make_chunk_freq_table, 
#                                         repeat(input_path, num_pieces),
#                                         repeat(delim, num_pieces),
#                                         boundaries[:-1], # boundary starts
#                                         boundaries[1:]) # boundary ends

#             # combined_table: dict[tuple[bytes,...], int] = {}
#             if use_disk:
#                 combined_table = diskcache.Index(f'{cache_dir}/cache_{uuid.uuid4()}')
#             else:
#                 combined_table = {}

#             for table in freq_tables:
#                 for (k, v) in table.items():
#                     count = combined_table.get(k, 0)
#                     combined_table[k] = count + v


#     return combined_table


def get_max_key(pair_freqs):
    sort_vals = [(val, key[0], key[1]) for key, val in pair_freqs.items()]
    max_val = max(sort_vals)
    max_key = (max_val[1], max_val[2])
    return max_key

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
    num_cores: int = 8,
    num_pieces = 8,
    verbose = False,
    use_disk = False
    ) -> tuple[
        dict[int, bytes], # vocab: the tokenizer vocabulary,
                          #  mapping from int to bytes
        list[tuple[bytes, bytes]]]: # merges: a list of BPE merges from training
    

    delim = "|".join([re.escape(t) for t in special_tokens])
    
    if verbose:
        print(f"Training BPE:\n"
              f"Input path = {input_path}\n"
              f"Vocab size = {vocab_size}\n"
              f"Special tokens = {special_tokens}\n"
              f"Num cores = {num_cores}\n"
              f"Use disk = {use_disk}")

    if verbose:
        print(f"Starting pretokenization")

    start_time = time.perf_counter()
    freq_table = make_freq_table_multiprocessed(input_path, delim, num_cores, 
        num_pieces = num_pieces, use_disk = use_disk)
    execution_time = time.perf_counter() - start_time

    if verbose:
        print(f"Finished pretokenization in {execution_time:.6f} seconds\n")


    print(f"Setting up for merges")
    start_time = time.perf_counter()
    # create merge list
    merges: list[tuple[bytes, bytes]] = []
    new_tokens = []

    pair_freqs = make_pair_freqs(freq_table)
    
    tok_to_pretok = {}
    for pretok in freq_table:
        for t in pretok:
            tok_to_pretok.setdefault(t, set()).add(pretok)

    num_merges = vocab_size - 256 - len(special_tokens)

    execution_time = time.perf_counter() - start_time
    if verbose:
        print(f"Finished setting up merges in {execution_time:.6f} seconds.")

    print("Starting merges")
    start_time = time.perf_counter()

    merge_iterable = tqdm(range(num_merges)) if verbose else range(num_merges)
    for i in merge_iterable:
        max_key = get_max_key(pair_freqs)

        merges.append(max_key)
        new_tokens.append(max_key[0] + max_key[1])

        do_merge(freq_table, tok_to_pretok, pair_freqs, max_key, verbose=False)

    execution_time = time.perf_counter() - start_time
    if verbose:
        print(f"Finished merges in {execution_time:.6f} seconds.")
    


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


class Tokenizer:
    # construct a tokenizer from a given vocabulary, 
    # list of merges, and list of special tokens
    def __init__(self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None):
        

        self.vocab = vocab
        self.tok_to_ind = {tok:ind for ind, tok in self.vocab.items()}
        self.merges = merges
        self.merge_ind = {m:j for (j, m) in enumerate(self.merges)}
        self.special_tokens = None
        # sort from longest to shortest. 
        # eg if a and ab are both special tokens, want to check for ab first
    

        self.delim = None
        if special_tokens != None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.delim = "|".join([re.escape(t) for t in self.special_tokens])
            # Add new special tokens to the vocabulary
            ind = len(self.vocab)
            for t in special_tokens:
                t_enc = t.encode('utf-8')
                if t_enc not in self.tok_to_ind:
                    self.vocab[ind] = t_enc
                    self.tok_to_ind[t_enc] = ind
                    ind += 1


    @classmethod 
    def from_files(cls, 
        vocab_filepath: str, 
        merges_filepath: str,
        special_tokens: list[str] | None = None):
        
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)

        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))


    def do_one_merge_fast(self, tok_list):
        best_ind = len(self.merges)
        tup_ind = -1
        for j, (t1, t2) in enumerate(zip(tok_list[:-1], tok_list[1:])):
            mi = self.merge_ind.get((t1, t2), len(self.merges))
            if mi < best_ind:
                best_ind = mi
                tup_ind = j

        if best_ind < len(self.merges):
            return tup_ind
        else:
            return None

    # could be made faster by keeping track of candidate merges and 
    # updating after each
    def merge_tup(self, tok_list):
        # keep on doing merges until none happen
        while True:
            tok_ind = self.do_one_merge_fast(tok_list)
            if tok_ind == None:
                return tok_list
            else:
                tok_list[tok_ind:tok_ind+2] = [tok_list[tok_ind] + tok_list[tok_ind+1]]


    def encode_pretok(self, pretok: str) -> Iterator[int]:
        # encode to bytes
        enc_tup = tuple(bytes([c]) for c in pretok.encode('utf-8'))

        # iteratively apply merges
        # for m in self.merges:
        #     enc_tup = apply_merge_to_tup(enc_tup, m)
        enc_tup = tuple(self.merge_tup(list(enc_tup)))

        for t in enc_tup:
            yield self.tok_to_ind[t]

    def encode_iterable_no_special(self, text: str) -> Iterator[int]:
        pretokenized = re.finditer(PAT, text)
        for pt in pretokenized:
            yield from self.encode_pretok(pt.group())

    # given an iterable of strings, return a generator 
    # that lazily yields token IDs, for better memory efficiency
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if self.special_tokens == None:
            for s in iterable:
                yield from self.encode_iterable_no_special(s)

        else:
            for s in iterable:
                special_toks = re.finditer(self.delim, s)
                last_end = 0
                encoded_text = []
                for match in special_toks:
                    start, end = match.span()
                    if start > last_end:
                        chunk = s[last_end:start]
                        yield from self.encode_iterable_no_special(chunk)

                    # yield special token
                    special_tok = match.group().encode('utf-8')
                    yield self.tok_to_ind[special_tok]

                    last_end = end

                if last_end < len(s):
                    chunk = s[last_end:]
                    yield from self.encode_iterable_no_special(chunk)


    # Decode a sequence of token IDsinto text
    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b''.join([self.vocab[j] for j in ids])
        return decoded_bytes.decode('utf-8', errors='replace')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Mode: train, encode, decode, test_compr, or encode_and_serialize, or none", default="none")
    parser.add_argument("--corpus", type=str, help="filepath OR ts_valid OR ts_train OR owt_valid OR owt_train OR test_corpus", default="none")
    parser.add_argument("--vocab_size", type=int, help="Size of vocab for training", default="257")
    parser.add_argument("--cores", type=int, help="Number of cores for pretokenization", default="8")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--use_disk", action="store_true", help="Store frequency table to disk (so big corpus doesn't crash memory)")
    parser.add_argument("--num_pieces", type=int, help="Number of pieces to chunk the text into", default=8)

    args = parser.parse_args()

    print(f"Mode: {args.mode}, Corpus: {args.corpus}, Vocab Size: {args.vocab_size}")

    if args.mode == "train":
        special_tokens = [EOT]
        delim = "|".join([re.escape(t) for t in special_tokens])

        corpus_filepath = None
        output_prepend = None
        if args.corpus == "ts_valid": 
            corpus_filepath = tiny_stories_valid
            output_prepend = f"ts_valid_{args.vocab_size}"
        elif args.corpus == "ts_train": 
            corpus_filepath = tiny_stories_train
            output_prepend = f"ts_train_{args.vocab_size}"
        elif args.corpus == "owt_valid": 
            corpus_filepath = owt_valid
            output_prepend = f"owt_valid_{args.vocab_size}"
        elif args.corpus == "owt_train": 
            corpus_filepath = owt_train
            output_prepend = f"owt_train_{args.vocab_size}"
        else:
            path = Path(args.corpus)
            assert(path.exists())
            assert(path.is_file())
            corpus_filepath = args.corpus
            output_prepend = f"{path.name}_{args.vocab_size}"

        print(f"corpus_filepath: {corpus_filepath}, output_prepend: {output_prepend}")
        vocab, merges = train_bpe(corpus_filepath, args.vocab_size, special_tokens, verbose = args.verbose, use_disk = args.use_disk, num_pieces = args.num_pieces)
        print("Vocab:")
        pprint.pp(vocab)

        print("\nMerges:")
        pprint.pp(merges)

        with open(f"{output_prepend}_vocab.pkl", "wb") as file:
            pickle.dump(vocab, file)

        with open(f"{output_prepend}_merges.pkl", "wb") as file:
            pickle.dump(merges, file)
    elif args.mode == "decode":
        pass
    elif args.mode == "encode":
        my_tok = Tokenizer.from_files("ts_train_10000_vocab.pkl", "ts_train_10000_merges.pkl", [EOT])
        enc = my_tok.encode("hello to the world! my name is alex.<|endoftext|>on the 4th of july there was a big storm<|endoftext|>my name is now kendra!")
        print(f"Enc: {enc}")
        dec = my_tok.decode(enc)
        print(f"Dec: {dec}")
    elif args.mode == 'test_compr':
        test_compression_ratios()
    elif args.mode == 'encode_and_serialize':
        tiny_stories_tok = Tokenizer.from_files("ts_train_10000_vocab.pkl", "ts_train_10000_merges.pkl", [EOT])
        owt_tok = Tokenizer.from_files("owt_valid_10000_vocab.pkl", "owt_valid_10000_merges.pkl", [EOT])
        # print("Encoding tiny stories validation set w/ tiny stories tokenizer")
        # encode_and_serialize(tiny_stories_valid, tiny_stories_tok, tiny_stories_valid_tokd)
        # print("Encoding tiny stories training set w/ tiny stories tokenizer")
        # encode_and_serialize(tiny_stories_train, tiny_stories_tok, tiny_stories_train_tokd)
        # print("Encoding open web text validation set w/ open web text tokenizer")
        # encode_and_serialize(owt_valid, owt_tok, owt_valid_tokd)
        print("Encoding open web text training set w/ open web text tokenizer")
        encode_and_serialize(owt_train, owt_tok, owt_train_tokd_prepend, num_dumps = 8)
    elif args.mode == "none":
        pass
    else:
        raise ValueError("Mode not recognized: {args.mode}")

def test_compression_ratios(num_docs = 50):
    compare = ['ts_valid', 'ts_train', 'owt_valid', 'owt_train']

    with open(tiny_stories_valid, 'rb') as ts_valid:
        all_stories = ts_valid.read().decode("utf-8", errors="ignore")
        split_stories = re.split(re.escape(EOT), all_stories)
        sampled_ts = random.sample(split_stories, num_docs)

    with open(owt_valid, 'rb') as owt_valid_f:
        all_stories = owt_valid_f.read().decode("utf-8", errors="ignore")
        split_stories = re.split(re.escape(EOT), all_stories)
        sampled_owt = random.sample(split_stories, num_docs)

    # print("Tiny stories:")
    # pprint.pp(sampled_ts)
    # print("\n\nOWT:")
    # pprint.pp(sampled_owt)

    tokenizers = {}
    for c in compare:
        tokenizers[c] = Tokenizer.from_files(f"{c}_10000_vocab.pkl",
                                            f"{c}_10000_merges.pkl",
                                            [EOT])
    
    total_toks_on_ts = {}
    total_toks_on_owt = {}
    total_time_on_ts = {}
    total_time_on_owt = {}
    total_bytes_ts = 0
    total_bytes_owt = 0

    for doc in sampled_ts:
        total_bytes_ts += len(doc.encode("utf-8"))

        for c in compare:
            start = time.perf_counter()
            tokenized = tokenizers[c].encode(doc)
            soft_add(total_time_on_ts, c, time.perf_counter() - start)
            soft_add(total_toks_on_ts, c, len(tokenized))

    for doc in sampled_owt:
        total_bytes_owt += len(doc.encode("utf-8"))

        for c in compare:
            start = time.perf_counter()
            tokenized = tokenizers[c].encode(doc)
            soft_add(total_time_on_owt, c, time.perf_counter() - start)
            soft_add(total_toks_on_owt, c, len(tokenized))

    for c in compare:
        print(f"Bytes/tok for model {c} on tiny stories: {total_bytes_ts/total_toks_on_ts[c]:.4f}")

    for c in compare:
        print(f"Bytes/tok for model {c} on open web text: {total_bytes_owt/total_toks_on_owt[c]:.4f}")

    print(f"Trivial bytes/tok (if we used 256 tokens, just bytes): {math.log(10000)/math.log(256):.4f}")

    for c in compare:
        print(f"Bytes/s for model {c} on tiny stories: {total_bytes_ts/total_time_on_ts[c]:.4f}")

    for c in compare:
        print(f"Bytes/s for model {c} on open web text: {total_bytes_owt/total_time_on_owt[c]:.4f}")

    
def encode_chunk(
    input_path: str | os.PathLike, 
    start: int, end: int, tokenizer: Tokenizer, 
    position = 0) -> list[int]:
    
    with open(input_path, 'rb') as doc:
        doc.seek(start)
        chunk = doc.read(end - start).decode('utf-8')
        subdocs_iterable = re.split('('+ re.escape(EOT) + ')', chunk)
        # print(subdocs_iterable)

    all_toks = []
    for doc in tqdm(subdocs_iterable, position=position, leave=False, desc=f"Chunk {position}"):
        all_toks += tokenizer.encode(doc)

    return all_toks


## This function is a bit of a mess. I used it to tokenize and serialize the training and validation corpi. 
## I had to split owt_train into 8 pieces. I should have used np.memmap to deal with it in one file, without crashing memory.
def encode_and_serialize(input_path, tokenizer, output_path_prepend, num_processes = 8, num_pieces = 128, num_dumps = 1):
    with open(input_path, "rb") as f:
        boundaries = cs336_basics.pretokenization_example.find_chunk_boundaries(f, num_pieces, EOT.encode('utf-8'))

    num_pieces = len(boundaries)-1
    print("Num boundaries:", num_pieces)


    chunks_per_dump = num_pieces // num_dumps
    for dump_ind in range(num_dumps):
        bd_ind_start = dump_ind * chunks_per_dump
        bd_ind_end = min((dump_ind+1) * chunks_per_dump, num_pieces)
        chunks = zip(boundaries[bd_ind_start:bd_ind_end], boundaries[bd_ind_start+1:bd_ind_end+1])
        # pprint.pp(list(chunks))
        print(f"Starting chunk {dump_ind}")

        # <-- Create a picklable multiprocessing manager lock
        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        mp_lock = manager.RLock()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers = num_processes,
            initializer = tqdm.set_lock,
            initargs = (mp_lock,)  # <-- Pass the picklable manager lock instead of tqdm.get_lock()
        ) as executor:
            

                futures = [
                    executor.submit(encode_chunk, input_path, start, end, tokenizer, idx)
                    for idx, (start, end) in enumerate(chunks)
                ]
                # all_toks = [t for f.result() for f in futures]
                all_tok_inds = [ti for f in futures for ti in f.result()]
                all_tok_inds = np.array(all_tok_inds, dtype=np.uint16)

                np.save(f"{output_path_prepend}_{dump_ind}.npy", all_tok_inds)
                print(f"Dumped chunk {dump_ind}")
            
        # Clean up the manager context once the pool shuts down
        manager.shutdown()

def convert_to_npy(filepath, outpath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    pprint.pp(data)
    np.save(outpath, data)

def tokens(text: str, tokenizer = None):
    if tokenizer == None:
        tokenizer = Tokenizer.from_files("ts_train_10000_vocab.pkl", "ts_train_10000_merges.pkl", [EOT])

    enc = tokenizer.encode(text)
    return [tokenizer.vocab[j] for j in enc]
    

if __name__ == '__main__':
    # main()
    tiny_stories_tok = Tokenizer.from_files("ts_train_10000_vocab.pkl", "ts_train_10000_merges.pkl", [EOT])
    owt_tok = Tokenizer.from_files("owt_valid_10000_vocab.pkl", "owt_valid_10000_merges.pkl", [EOT])



