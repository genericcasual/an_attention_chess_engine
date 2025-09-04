import jax.numpy as jnp
from flax import nnx
import jax
import os
import utils

WS_DIR = os.getcwd()
DEFAULT_RNG = nnx.Rngs(0)

class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear1(x)
        x = self.norm(x)
        x = nnx.gelu(x)
        x = self.linear2(x)
        return x


class Transformer(nnx.Module):
    """A single Transformer block.

    Each Transformer block processes input sequences via self-attention and feed-forward networks.

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, embed_dim: int, num_heads: int, rngs: nnx.Rngs, rate: float):
        # Multi-Head Attention (MHA) with `flax.nnx.MultiHeadAttention`.
        # Specifies tensor sharding (depending on the mesh configuration)
        # where we shard the weights across devices for parallel computation.
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=embed_dim, decode=False, rngs=rngs
        )
        # The first dropout with `flax.nnx.Dropout`.
        self.dropout1 = nnx.Dropout(rngs=rngs, rate=rate)
        # First layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm1 = nnx.LayerNorm(
            epsilon=1e-6, num_features=embed_dim, rngs=rngs
        )
        # The first linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear1 = nnx.Linear(
            in_features=embed_dim, out_features=embed_dim, rngs=rngs
        )
        # The second linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear2 = nnx.Linear(
            in_features=embed_dim, out_features=embed_dim, rngs=rngs
        )
        # The second dropout with `flax.nnx.Dropout`.
        self.dropout2 = nnx.Dropout(rngs=rngs, rate=rate)
        # Second layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm2 = nnx.LayerNorm(
            epsilon=1e-6, num_features=embed_dim, rngs=rngs
        )

    # Apply the Transformer block to the input sequence.
    def __call__(self, inputs, deterministic: bool = False):
        # Instantiate the causal attention mask.

        # Apply Multi-Head Attention with the causal attention mask.
        attention_output = self.mha(inputs_q=inputs, decode=False)
        # Apply the first dropout.
        attention_output = self.dropout1(attention_output, deterministic=deterministic)
        # Apply the first layer normalization.
        out1 = self.layer_norm1(inputs + attention_output)

        # The feed-forward network.
        # Apply the first linear transformation.
        ffn_output = self.linear1(out1)
        # Apply the ReLU activation with `flax.nnx.relu`.
        # ffn_output = nnx.relu(ffn_output)
        ffn_output = nnx.tanh(ffn_output)
        # Apply the second linear transformation.
        ffn_output = self.linear2(ffn_output)
        # Apply the second dropout.
        ffn_output = self.dropout2(ffn_output, deterministic=deterministic)
        # Apply the second layer normalization and return the output of the Transformer block.
        return self.layer_norm2(out1 + ffn_output)


class Model(nnx.Module):
    """model architecture

    Args:
        board_dim: number of tokens for input, default=64
        embed_dim: dimensions of embedding, default=200
        num_of_transformers: transformer_blocks daisy chained
        num_heads: number of attention heads, must be a divisor of embed_dim
    """
    def __init__(
        self, rngs: nnx.Rngs, board_dim=64, embed_dim=200, num_of_transformers=1, num_heads=200//25
    ):
        self.embed = nnx.Embed(num_embeddings=board_dim, features=embed_dim, rngs=rngs)

        self.transformers = [
            Transformer(
                embed_dim=embed_dim, num_heads=num_heads, rngs=rngs, rate=0.1
            )
            for _ in range(num_of_transformers)
        ]
        # Initialize positional embeddings (using `flax.nnx.Embed`).
        self.pos_emb = nnx.Embed(
            num_embeddings=board_dim, features=embed_dim, rngs=rngs
        )
        self.output_layer = nnx.Linear(
            in_features=board_dim * embed_dim, out_features=1, rngs=rngs
        )

    def __call__(self, x: jax.Array, deterministic: bool = False):
        positions = jnp.array([jnp.arange(0, 64) for i in range(x.shape[0])])
        # print(positions,x.shape)
        position_embedding = self.pos_emb(positions)
        token_embedding = self.embed(x)

        x = token_embedding + position_embedding

        for transformer in self.transformers:
            x = transformer(x, deterministic=deterministic)
        arr = [nnx.tanh(self.output_layer(x[i].flatten())) for i in range(x.shape[0])]
        # x = x.flatten()
        # x = self.output_layer(x)
        return jnp.array(arr)


def create_model():
    return Model(rngs=nnx.Rngs(0), board_dim=64, embed_dim=200, num_of_transformers=5) # the embed dim has to be a multiple of 25


if __name__ == "__main__":
    utils.save_model(create_model(), WS_DIR + "/checkpoints")
