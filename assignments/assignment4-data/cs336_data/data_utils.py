####
## extract text from html bytes file ## 

import resiliparse.extract.html2text
from fastwarc.warc import ArchiveIterator, WarcRecordType
import nltk
import numpy as np


## load fasttext classifiers

## Depends on the machine
CLASSIFIER_DIR = "/scratch/ac6074/cs336-a4-data/classifiers"

import fasttext
lang_detect_model = fasttext.load_model(f"{CLASSIFIER_DIR}/lid.176.bin")
nsfw_model = fasttext.load_model(f"{CLASSIFIER_DIR}/dolma_fasttext_nsfw_jigsaw_model.bin")
hatespeech_model = fasttext.load_model(f"{CLASSIFIER_DIR}/dolma_fasttext_hatespeech_jigsaw_model.bin")

import re 
EMAIL_REGEX = r"""[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"""
PHONE_REGEX = r"""(?:\+?\d{1,3}[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})""" 
IP_REGEX = r"""(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)""" 

COMPILED_EMAIL_REGEX = re.compile(EMAIL_REGEX)
COMPILED_PHONE_REGEX = re.compile(PHONE_REGEX)
COMPILED_IP_REGEX = re.compile(IP_REGEX)


## 
## 
def warc_text_iterator(file_path): 
    """
    Make an iterator over WARC entries, returning the entry along with extracted plain text using resiliparse
    Use as for entry, parsed_text in warc_text_iterator()
    yields entry, warc_txt
    """
    with open(file_path, "rb") as f:
        it = ArchiveIterator(f)
        for entry in it:
            if entry.headers.get("Content-Type", "NONE") in ["application/http; msgtype=response", "application/http;msgtype=response"]: continue
            try: 
                warc_str = entry.reader.read().decode()
                warc_txt = resiliparse.extract.html2text.extract_plain_text(warc_str).strip()
                if len(warc_txt) == 0: continue
                yield entry, warc_txt
            except UnicodeDecodeError:
                continue

def extract_text_from_warc_bytes(doc_bytes):
    try: 
        warc_str = doc_bytes.decode()
        warc_txt = resiliparse.extract.html2text.extract_plain_text(warc_str).strip()
        return warc_txt
    except UnicodeDecodeError: 
        return "|||FAILED_UTF-8_DECODING|||"

####
## identify language using fasttext classifier ##


def identify_language(text):
    stripped = text.replace("\n", " ").strip()
    lan, conf = lang_detect_model.predict(stripped)
    lan = lan[0]
    conf = conf[0]
    return lan[-2:], conf


def is_english(text, english_cutoff = 0.80):
    lang, conf = identify_language(text)
    return lang == "en" and conf >= english_cutoff

####
## mask email, phone, and IP using regex ## 



def sub(str, regexp, sub):
    start_count = str.count(sub)
    subbed = re.sub(regexp, sub, str)
    end_count = subbed.count(sub)
    return subbed, end_count - start_count

def mask_emails(str):
    return sub(str, COMPILED_EMAIL_REGEX, "|||EMAIL_ADDRESS|||")


def mask_phone(str):
    return sub(str, COMPILED_PHONE_REGEX, "|||PHONE_NUMBER|||")

def mask_ip(str):
    return sub(str, COMPILED_IP_REGEX, "|||IP_ADDRESS|||")


####
## classify NSFW and hatespeech using fasttext classifiers ## 

def get_is_nsfw(text, with_conf = True):
    stripped = text.replace('\n', ' ').strip()
    res = nsfw_model.predict(stripped)
    pred = res[0][0].strip('__label__')
    conf = res[1][0]
    if with_conf:
        return pred , conf
    else:
        return pred == 'nsfw'

def get_is_hatespeech(text, with_conf = True):
    stripped = text.replace('\n', ' ').strip()
    res = hatespeech_model.predict(stripped)
    pred = res[0][0].strip('__label__')
    conf = res[1][0]
    if with_conf:
        return pred , conf
    else:
        return pred == 'hatespeech'



####
## run simple gopher quality filters ## 

# Use nltk.word_tokenizer to count words. 

# - Restrict to num words in [50, 100_000]
# - Mean word length in [3, 10] chars
# - Remove doc if >30% of lines end with ellipsis ...
# - Remove doc is <80% of words contain at least one alphabetic character

## Check if document passes simple gopher quality test

HAS_ALPHANUM_RE = re.compile("[a-zA-Z]")
def has_alphanum(w):
    return re.search(HAS_ALPHANUM_RE, w) != None

def check_gopher_good(text, word_ran = (50, 100_000), mean_word_len_ran = (3, 10), max_frac_ellipsis = 0.3, min_frac_alphanum = 0.8):
    words = nltk.word_tokenize(text)

    num_words_good = len(words) >= word_ran[0] and len(words) <= word_ran[1]
    
    mean_word_len = np.mean([len(w) for w in words])
    mean_word_len_good = mean_word_len >= mean_word_len_ran[0] and mean_word_len <= mean_word_len_ran[1]

    lines = text.split('\n')
    num_lines = len(lines)
    lines_with_ellipsis = len([l for l in lines if l.strip()[-3:] == "..."])
    percent_ellipsis = lines_with_ellipsis / num_lines
    ellipsis_good = percent_ellipsis <= max_frac_ellipsis

    alphanum_words = [w for w in words if has_alphanum(w)]
    percent_alphanum = len(alphanum_words)/len(words)
    alphanum_good = percent_alphanum >= min_frac_alphanum

    return num_words_good and mean_word_len_good and ellipsis_good and alphanum_good

#### 
def simple_pipeline(text, english_cutoff = 0.8, word_ran = (50, 100_000), mean_word_len_ran = (3, 10), max_frac_ellipsis = 0.3, min_frac_alphanum = 0.8):
    """
    Test if text is english, is not NSFW, is not hatespeech, and passes gopher
    If it fails, return None
    If it passes, mask email, phone, IP and return
    """

    ## Check simple filters
    if not is_english(text, english_cutoff): return "NOT_ENGLISH"
    if get_is_nsfw(text, with_conf = False): return "NSFW"
    if get_is_hatespeech(text, with_conf = False): return "HATESPEECH"
    if not check_gopher_good(text, word_ran, mean_word_len_ran, max_frac_ellipsis, min_frac_alphanum): return "BAD_GOPHER"

    # Mask everything
    text, _ = mask_ip(text)
    text, _ = mask_phone(text)
    text, _ = mask_emails(text)

    return text

####
## TODO: Implement quality classifier using wikipedia links & fasttext


### 
## Exact line deduplication ## 

import hashlib
from pathlib import Path
 

def exact_line_deduplication(inputs_paths, output_dir):
    """
    input_paths: a list of input file paths to run deduplification on 
    output_dir: a directory to write deduplified files
    For example, if the input paths are a/1.txt and a/2.txt, and the output directory is b/, this function  writes the files b/1.txt and b/2.txt. 
    """
    times_line_hashed = {}

    ## First, go through the files and count how many times each line appears
    for p in inputs_paths:
        with open(p, "r") as f: 
            dedup_file = ""
            orig_file = ""
            for l in f.readlines():
                hashed = hashlib.sha256(l.encode('utf-8')).hexdigest()
                # print("Line:", l, hashed)
                orig_file += l
                times_hashed = times_line_hashed.get(hashed, 0)
                times_line_hashed[hashed] = times_hashed + 1
                

    # Then, go through and only include unique lines
    for p in inputs_paths:
        with open(p, "r") as f: 
            dedup_file = ''.join([l for l in f.readlines() if 
                            times_line_hashed[hashlib.sha256(l.encode('utf-8')).hexdigest()] == 1])
                    

        new_file = Path(output_dir) / p.name
        with open(new_file, "w+") as f:
            f.write(dedup_file)


### 
## Fuzzy document deduplification ## 

"""
We use MinHash and locally sensative hashing.
We are given a collection of documents d_1, ..., d_N

We want to cluster our documents into a potentially smaller number of similar clusters, and choose just one representative per cluster. 
So, the output should be a subset of the documents

We use Jaccard similarity. Let S(d_1), S(d_2) be the bag of n-grams. 
The Jaccard similarity is |S(d_1) cap S(d_2)| / |S(d_1) cup S(d_2)|

We approximate Jaccard similarity using minhashing. 
We use k random hash functions h_1, ..., h_k. 
For each hash function, we get a single number out of d by taking 

h(d) = min(h(s_1), ..., h(s_m)) , s_j in S(d) is the set of n-grams.

We represent a document the sequence of k numbers 
[h_1(d), ..., h_k(d)]

For a given hash function h, what is the probability h(d_1) = h(d_2)? 
Let 
A = number of n-grams in d_1 / d_2 
B = number of n-grams in d_1 cap d_2 
C = number of n-grams in d_2 / d_1
Then h(d_1) = h(d_2) if the h(d_1) and h(d_2) are both achieved on B, and 
most likely this dominates the probability that h(d_1) = h(d_2).

The probability that this holds is ~ B/(A+B+C) = Jaccard similarity of d_1, d_2

To make the computation more tractable, we only test for Jaccard similarity 
"""

