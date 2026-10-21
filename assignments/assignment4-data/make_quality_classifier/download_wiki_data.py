from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse.extract.html2text
from typing import Literal
import rich
from cs336_data import data_utils 
import numpy as np
from transformers import AutoTokenizer

WIKI_DATA_PATH = "/scratch/ac6074/cs336-a4-data/wiki/subsampled_positive_urls_10k.warc.warc.gz"
CC_EXAMPLE_PATH = "/scratch/ac6074/cs336-a4-data/CC/example.warc.gz"
FASTTEXT_TRAIN_FILE = "/scratch/ac6074/cs336-a4-data/fasttext_train_quality.txt"
PALOMA = "/scratch/ac6074/cs336-a4-data/tokenized_paloma_c4_100_domains_validation.bin"

if __name__ == "__main__":
    
    # data = np.fromfile(PALOMA, dtype=np.uint16)
    
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # print(tokenizer.decode(data[0:2000]))
    # upto = 100
    # num_wiki_docs = 0 
    num_cc_docs = 0 
    # upto = 100

    # with open(WIKI_DATA_PATH, "rb") as f:
    #     it = ArchiveIterator(f)
    #     for i, entry in enumerate(it):
    #         if i > upto: break
    #         if entry.headers["WARC-Type"] != "response": continue
    #         print(f"Ind={i} | {entry.headers}")
    #         try: 
    #             warc_str = entry.reader.read().decode()
    #             warc_txt = resiliparse.extract.html2text.extract_plain_text(warc_str).strip()
    #             print(warc_txt)
    #         except UnicodeDecodeError:
    #             continue

    # for i, (entry, text) in enumerate(data_utils.warc_text_iterator(WIKI_DATA_PATH)):
    #     # print(f"Ind={i}\n{text}\n\n---------------\n\n")
    #     classified_text = data_utils.simple_pipeline(text)
    #     print(classified_text)
    #     print(entry.headers)
    #     print("\n\n")
    #     if classified_text not in ["NOT_ENGLISH", "NSFW", "HATESPEECH", "BAD_GOPHER"]:
    #         num_wiki_docs += 1
    #         print(f"Ind={i}, num wiki docs={num_wiki_docs}\n{classified_text}\n\n")

    for i, (entry, text) in enumerate(data_utils.warc_text_iterator(CC_EXAMPLE_PATH)):
        # print(f"Ind={i}\n{text}\n\n---------------\n\n")
        if i > 100: break
        classified_text = data_utils.simple_pipeline(text)
        if classified_text not in ["NOT_ENGLISH", "NSFW", "HATESPEECH", "BAD_GOPHER"]:
            num_cc_docs += 1
            print(f"Ind={i}, num cc docs={num_cc_docs}")
