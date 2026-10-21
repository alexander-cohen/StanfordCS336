print("importing fastwarc")
from fastwarc.warc import ArchiveIterator, WarcRecordType
print("importing resiliparse")
import resiliparse.extract.html2text
print("importing typing")
from typing import Literal
print("importing rich")
import rich
print("importing numpy")
import numpy as np
print("importing transformers")
from transformers import AutoTokenizer

WIKI_DATA_PATH = "/scratch/ac6074/cs336-a4-data/wiki/subsampled_positive_urls_10k.warc.warc.gz"
CC_EXAMPLE_PATH = "/scratch/ac6074/cs336-a4-data/CC/example.warc.gz"
CC_EXAMPLE_PATH_WET = "/scratch/ac6074/cs336-a4-data/CC/example.warc.wet.gz"
FASTTEXT_TRAIN_FILE = "/scratch/ac6074/cs336-a4-data/fasttext_train_quality.txt"
CC_WET_OUT = "/scratch/ac6074/cs336-a4-data/cc_wet_docs.txt"
CC_PALOMA_OUT = "/scratch/ac6074/cs336-a4-data/cc_paloma_docs.txt"
PALOMA = "/scratch/ac6074/cs336-a4-data/tokenized_paloma_c4_100_domains_validation.bin"

print(__name__)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
paloma_data = np.memmap(PALOMA, mode="r", dtype=np.uint16)

def make_paloma_file(num_lines, num_chars_per_line):
    approx_toks = num_lines * num_chars_per_line // 4 + 10000
    all_toks = np.array(paloma_data[0:approx_toks])
    as_str = tokenizer.decode(all_toks)
    del all_toks
    all_words = as_str.split(" ")
    cur_word_ind = 0
    all_lines = []
    for l in range(num_lines):
        line = ""
        cur_line_len = 0
        while cur_line_len < num_chars_per_line: 
            word = all_words[cur_word_ind]
            line += word + " " 
            cur_word_ind += 1
            cur_line_len += len(word) + 1
        all_lines.append(line)
    
    with open(CC_PALOMA_OUT, "w") as f:
        for l in all_lines:
            f.write(l + "\n")

if __name__ == "__main__":
    make_paloma_file(3000, 6000)