import cs336_basics.decode as decode
import cs336_basics.transformer as transformer
import cs336_basics.tokenizer as tokenizer
import cs336_basics.nn_utils as nn_utils
import cs336_basics.training as training
import torch
import numpy as np


EOT = "<|endoftext|>"
model = transformer.Transformer(10000, 20, 1, 10, 1, 10)
tok = tokenizer.Tokenizer.from_files("ts_train_10000_vocab.pkl", "ts_train_10000_merges.pkl", [EOT])

decode.decode(model, tok, "hello", 10)

tiny_stories_valid_tokd = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-valid_tokd.npy'
tiny_stories_valid_dat = np.load(tiny_stories_valid_tokd, mmap_mode='r')

model_params = training.ModelConfig(
    vocab_size=10000,
    context_length=20,
    num_heads=2,
    num_layers=2,
    d_model = 20,
    d_ff = 20,
    rope_theta = 10000
)

training_params = training.TrainingConfig(
    batch_size=4,
    training_steps=100,
    save_dist = 100,
    grad_clip=float('inf'),
    device=torch.device('cpu'),
    lr_mode='basic',
    weight_decay= 0.0,
    betas = (0.9, 0.999),
    lr=0.01
)

