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
# print("importing transformers")
# from transformers import AutoTokenizer
print("importing data utils")
from cs336_data import data_utils 

WIKI_DATA_PATH = "/scratch/ac6074/cs336-a4-data/wiki/subsampled_positive_urls_10k.warc.warc.gz"
CC_EXAMPLE_PATH = "/scratch/ac6074/cs336-a4-data/CC/example.warc.gz"
CC_EXAMPLE_PATH_WET = "/scratch/ac6074/cs336-a4-data/CC/example.warc.wet.gz"
FASTTEXT_TRAIN_FILE = "/scratch/ac6074/cs336-a4-data/fasttext_train_quality.txt"
CC_WET_OUT = "/scratch/ac6074/cs336-a4-data/cc_wet_docs.txt"
CC_PALOMA_OUT = "/scratch/ac6074/cs336-a4-data/cc_paloma_docs.txt"
PALOMA = "/scratch/ac6074/cs336-a4-data/tokenized_paloma_c4_100_domains_validation.bin"

print(__name__)



def make_wet_iterator(fpath):
    with open(fpath, "rb") as f:
        it = ArchiveIterator(f)
        for i, entry in enumerate(it):
            try:
                wet_str = entry.reader.read().decode()
                classified_text = data_utils.simple_pipeline(wet_str)
                if classified_text not in ["NOT_ENGLISH", "NSFW", "HATESPEECH", "BAD_GOPHER"]:
                    yield classified_text
            except UnicodeDecodeError:
                continue
    

def write_wet_data():
    with open(CC_WET_OUT, "w", encoding="utf-8") as f: 
        for i, text in enumerate(make_wet_iterator(CC_EXAMPLE_PATH_WET)):
            if i % 100 == 0: print(i)
            oneline = text.replace("\n", " ")
            f.write(oneline + "\n")

def get_avg_char_per_line():
    with open(CC_WET_OUT, "r", encoding="utf-8") as f: 
        lines = f.readlines()
        char_counts = [len(l) for l in lines]
        avg_char_count_per_line = np.mean(char_counts)
    return avg_char_count_per_line
        

if __name__ == "__main__":
    print("Avg char per line:", get_avg_char_per_line())
    
