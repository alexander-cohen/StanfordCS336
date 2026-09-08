vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288

weights_embedding = vocab_size * d_model
weights_RMS_per_head = d_model * 2
weights_per_MHA = (d_model**2) * 4 # W_Q, W_K, W_V, W_O
weights_per_FF = d_model * d_ff * 3 # W1, W2, W3
weights_final_RMS = d_model
weights_final_lin = d_model * vocab_size

weights_per_block = weights_RMS_per_head + weights_per_MHA + weights_per_FF
total_weights = weights_embedding + weights_per_block * num_layers + weights_final_RMS + weights_final_lin

muls_per_MHA = 4 * d_model**2 * context_length + 2 * d_model * context_length**2
muls_per_FF = (d_ff * d_model**2 * 2 + d_ff**2 * d_model) * context_length 
muls_final_layer = context_length * d_model * vocab_size
total_muls = (muls_per_FF+muls_per_MHA) * num_layers + muls_final_layer
