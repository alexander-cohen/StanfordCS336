import cs336_basics.decode as decode
import cs336_basics.transformer as transformer
import cs336_basics.tokenizer as tokenizer
import cs336_basics.nn_utils as nn_utils
import torch


EOT = "<|endoftext|>"
model = transformer.Transformer(10000, 20, 1, 10, 1, 10)
tok = tokenizer.Tokenizer.from_files("ts_train_10000_vocab.pkl", "ts_train_10000_merges.pkl", [EOT])

decode.decode(model, tok, "hello", 10)