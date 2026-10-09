from typing import Literal

from cs336_scaling.schemas.base import FrozenForbidExtraModel


class BasicTransformerConfig(FrozenForbidExtraModel):
    attention_bias: bool
    head_dim: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: int
    tie_word_embeddings: bool
    dtype: Literal["float32", "bfloat16"]
    vocab_size: int

    @property
    def jax_dtype(self):
        import jax.numpy as jnp

        return jnp.dtype(self.dtype)


def get_num_params_for_flops(model_config: BasicTransformerConfig) -> int:
    """Count the total parameters in BasicCausalLM for flop accounting."""
    hidden_size = model_config.hidden_size
    query_size = model_config.num_attention_heads * model_config.head_dim
    key_value_size = model_config.num_key_value_heads * model_config.head_dim

    # Query/output weights each use query_size; key/value use key_value_size.
    attention_params = 2 * hidden_size * (query_size + key_value_size)
    if model_config.attention_bias:
        attention_params += query_size + 2 * key_value_size + hidden_size

    feedforward_params = 3 * hidden_size * model_config.intermediate_size

    # Two decoder RMSNorms and the shared-across-heads query/key RMSNorms.
    layer_norm_params = 2 * hidden_size + 2 * model_config.head_dim
    layer_params = attention_params + feedforward_params + layer_norm_params

    embedding_params = model_config.vocab_size * hidden_size # include logit embeddings

    # The final RMSNorm appears once, outside the decoder layers.
    return (
        embedding_params + model_config.num_hidden_layers * layer_params + hidden_size
    )

def get_trainable_params(model_config: BasicTransformerConfig) -> int:
    total_params = get_num_params_for_flops(model_config)
    if not model_config.tie_word_embeddings:
        embedding_params = model_config.vocab_size * model_config.hidden_size
        return total_params + embedding_params # this many more embedding params
    else:
        return total_params

def get_num_embedding_params(model_config: BasicTransformerConfig) -> int:
    """Count the embedding parameters in BasicCausalLM without creating its weights."""
    embedding_params = model_config.vocab_size * model_config.hidden_size
    if not model_config.tie_word_embeddings:
        embedding_params *= 2
    return embedding_params