import cs336_basics.decode as decode
import cs336_basics.transformer as transformer
import cs336_basics.tokenizer as tokenizer
import cs336_basics.nn_utils as nn_utils
import cs336_basics.training as training
import torch
import numpy as np
from datetime import datetime
import cs336_basics.decode as decode


EOT = "<|endoftext|>"
model = transformer.Transformer(10000, 20, 1, 10, 1, 10)
tok = tokenizer.Tokenizer.from_files("ts_train_10000_vocab.pkl", "ts_train_10000_merges.pkl", [EOT])

decode.decode(model, tok, "hello", 10)

tiny_stories_valid_tokd = '/Users/alexandercohen/Local Documents/data/tinystories/TinyStoriesV2-GPT4-valid_tokd.npy'
tiny_stories_valid_dat = np.load(tiny_stories_valid_tokd, mmap_mode='r')

checkpoint_dir = f"/Users/alexandercohen/Local Documents/checkpoints/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

model_params = transformer.ModelConfig(
    vocab_size=10000,
    context_length=256,
    num_heads=16,
    num_layers=4,
    d_model = 512,
    d_ff = 1344,
    rope_theta = 10000
)

training_params = training.TrainingConfig(
    batch_size=16,
    training_steps=10000,
    save_dist = 100 ,
    grad_clip=float('inf'),
    device=torch.device('mps'),
    lr_mode='cosine_scheduled',
    weight_decay= 0.0,
    betas = (0.9, 0.999),
    time_warmup=100,
    time_final_anneal=10000,
    lr_max = 3e-4,
    lr_min = 3e-5
)



training.training_loop(tiny_stories_valid_dat, model_params, training_params, checkpoint_out=checkpoint_dir, tok = tok)

