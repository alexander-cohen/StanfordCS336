####
## extract text from html bytes file ## 
import shutil
import random
import itertools
import string
import sys
import resiliparse.extract.html2text
import unicodedata
import mmh3

def extract_text_from_warc_bytes(doc_bytes):
    try: 
        warc_str = doc_bytes.decode()
        warc_txt = resiliparse.extract.html2text.extract_plain_text(warc_str).strip()
        return warc_txt
    except UnicodeDecodeError: 
        return "|||FAILED_UTF-8_DECODING|||"

####
## identify language using fasttext classifier ##

import fasttext
lang_detect_model = fasttext.load_model("/Users/alexandercohen/Developer/CS336-Spring-2026/assignments/assignment4-data/local-shared-data/classifiers/lid.176.bin")

def identify_language(text):
    stripped = text.replace("\n", " ").strip()
    lan, conf = lang_detect_model.predict(stripped)
    lan = lan[0]
    conf = conf[0]
    return lan[-2:], conf


####
## mask email, phone, and IP using regex ## 

import re 
EMAIL_REGEX = r"""[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"""
PHONE_REGEX = r"""(?:\+?\d{1,3}[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})""" 
IP_REGEX = r"""(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)""" 

COMPILED_EMAIL_REGEX = re.compile(EMAIL_REGEX)
COMPILED_PHONE_REGEX = re.compile(PHONE_REGEX)
COMPILED_IP_REGEX = re.compile(IP_REGEX)

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

import fasttext
nsfw_model = fasttext.load_model("/Users/alexandercohen/Developer/CS336-Spring-2026/assignments/assignment4-data/local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
hatespeech_model = fasttext.load_model("/Users/alexandercohen/Developer/CS336-Spring-2026/assignments/assignment4-data/local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

def get_is_nsfw(text, with_conf = True):
    stripped = text.replace('\n', ' ').strip()
    res = nsfw_model.predict(stripped)
    pred = res[0][0].strip('__label__')
    conf = res[1][0]
    if with_conf:
        return pred , conf
    else:
        return pred

def get_is_hatespeech(text, with_conf = True):
    stripped = text.replace('\n', ' ').strip()
    res = hatespeech_model.predict(stripped)
    pred = res[0][0].strip('__label__')
    conf = res[1][0]
    if with_conf:
        return pred , conf
    else:
        return pred



####
## run simple gopher quality filters ## 

# Use nltk.word_tokenizer to count words. 

# - Restrict to num words in [50, 100_000]
# - Mean word length in [3, 10] chars
# - Remove doc if >30% of lines end with ellipsis ...
# - Remove doc is <80% of words contain at least one alphabetic character

import nltk
import numpy as np

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
B = number of n-grams in d_1 \cap d_2 
C = number of n-grams in d_2 / d_1
Then h(d_1) = h(d_2) if the h(d_1) and h(d_2) are both achieved on B, and 
most likely this dominates the probability that h(d_1) = h(d_2).

The probability that this holds is ~ B/(A+B+C) = Jaccard similarity of d_1, d_2

(This is the Minhash part)

To make the computation more tractable, we only test for Jaccard similarity between documents that completely agree on a few of their minhashes. 
This is the locally sensative hashing part. 
We split up our set of k minhashes into b bands containing r minhashes each, with k = br
We cluster hashes into the same bucket if they have the same minhashes in some band

We test all pairs of documents that appear in the same bucket. 
We compute the true Jaccard similarity (not the approximation using minhash)

Then, we draw a graph by connecting two documents if they have high enough Jaccard similarity. We use connected components in this graph. 
We randomly remove all but one document in each cluster to get a list of representative documents. 
"""
all_accents = "".join(chr(i) for i in range(sys.maxunicode) if unicodedata.combining(chr(i)))
to_remove = string.punctuation + all_accents
remove_punctuation_and_accents = str.maketrans('', '', to_remove)

def normalize(text):
    """ 
    Lowercase, remove punctuation, normalize whitespaces, remove accents, apply NFD unicode normalization
    """
    nfd = unicodedata.normalize('NFD', text) # unicode normalization to NFD
    nfd = nfd.lower() # lowercase
    nfd = nfd.translate(remove_punctuation_and_accents) # remove punctuation
    nfd = " ".join(nfd.split()) # normalize whitespace to one space

    return nfd

def hash_ngram(ng, seed):
    return mmh3.hash("|".join(ng).encode("utf-8"), seed, signed = False)

def make_maxhashes(doc, k, n):
    """
    Create k-many n-gram minhash
    Assume doc is already normalized 
    """

    words = doc.split()
    maxhash = np.array([0 for _ in range(k)])
    for ng in nltk.ngrams(words, n):
        hashes = np.array([hash_ngram(ng, i) for i in range(k)])
        maxhash = np.maximum(maxhash, hashes)

    return maxhash


def make_ngram_set(doc, n):
    words = doc.split()
    return set(nltk.ngrams(words, n))

def jaccard(S1, S2):
    """compute jaccard similarity between two sets"""
    intersection = S1.intersection(S2)
    union = S1.union(S2)

    A = len(intersection)
    B = len(union)
    return A / B if B != 0 else 0


def get_jac_similar(docs, n, jac_cutoff):
    """return pairs of indices of docs with jac similar > jac_cutoff"""
    ngram_sets = [make_ngram_set(doc, n) for doc in docs]
    edges = []
    for i1, i2 in itertools.pairwise(range(len(ngram_sets))):
        jac = jaccard(ngram_sets[i1], ngram_sets[i2])
        if jac >= jac_cutoff: 
            edges.append((i1, i2))

    return edges

def make_clusters(n, edges):
    """
    edges is a list of pairs in 0...(n-1)
    we iteratively merge any pair in edges to lead to some number of clusters
    """

    # unique index describing the connected component of each document
    # we start with each documents cluster being its index
    cluster_ind = list(range(n))

    def get_cluster_ind(i):
        if cluster_ind[i] == i:
            return i
        else:
            return get_cluster_ind(cluster_ind[i])

    # merge indices one-by-one
    # at the end of merging, cluster_ind[i] = i' where i' <= i,
    # and i' and i are in the same cluster.
    for i1, i2 in edges:
        a = get_cluster_ind(i1)
        b = get_cluster_ind(i2)
        c = min(a,b)
        d = max(a,b)
        cluster_ind[d] = c

    for j in range(n):
        cluster_ind[j] = get_cluster_ind(j)

    distinct_vals = set(cluster_ind)
    groups = {v: [] for v in distinct_vals}
    for j in range(n):
        groups[cluster_ind[j]].append(j)

    return list(groups.values())


def fuzzy_document_deduplication(input_paths, # a list of documents to deduplify, 
                                 output_dir,
                                 k, # num hashes total 
                                 b, # num bands 
                                 n, # use n-grams
                                 jac_cutoff = 0.8): # documents with > jac_cutoff Jaccard similarity are combined in one group
    """ 
    Follow the procedure described above to perform fuzzy document deduplicaation 
    Rewrite files to the output dir that are selected representatives of their connected component
    Text will be normalized before computations
    """

    assert( k % b == 0 )
    r = k // b # num hashes per band

    def get_doc(fname):
        with open(fname, "r") as f:
            doc = f.read()
        
        return doc

    num_docs = len(input_paths)
    maxhashes = np.array([make_maxhashes(get_doc(fname), k, n) for fname in input_paths]) # d x k array 

    edges = []

    ## For each block, make buckets based on minhashes in that block. 
    ## For each bucket, compute true Jaccard similarity scores
    ## Add an edge whenever two documents are similar
    for block in range(b):
        inds = range( block * r, (block+1)*r )
        distinct_inds = {}
        unique_rows, inv = np.unique(maxhashes[:,inds], return_inverse = True, axis = 0)
        for j in range(num_docs):
            if inv[j] in distinct_inds:
                distinct_inds[inv[j]].append(j)
            else:
                distinct_inds[inv[j]] = [j]

        for inds in distinct_inds.values():
            ## compute jaccard distance for every pair in inds

            pairs = get_jac_similar([get_doc(input_paths[i]) for i in inds], n, jac_cutoff)
            for i1, i2 in pairs:
                edges.append((inds[i1], inds[i2]))


    
    doc_groups = make_clusters(num_docs, edges)

    rep_docs = [random.choice(group) for group in doc_groups]

    ## Write representative docs to output path
    for d in rep_docs:
        fname = input_paths[d]
        new_file = Path(output_dir) / Path(fname).name
        shutil.copy(fname, new_file)







if __name__ == "__main__":
    input_paths = [f"/Users/alexandercohen/Developer/CS336-Spring-2026/assignments/assignment4-data/test_docs_jaccard/test_docs_in/d{j}.txt" for j in range(1,6)]
    output_dir = "/Users/alexandercohen/Developer/CS336-Spring-2026/assignments/assignment4-data/test_docs_jaccard/test_docs_out"

    k = 4
    b = 2
    n = 2
    jac_cutoff = 0.8

    fuzzy_document_deduplication(input_paths, output_dir, k, b, n, jac_cutoff)