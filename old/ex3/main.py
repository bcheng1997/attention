
import numpy as np
import torch
import torch.nn as nn

"""
x = torch.tensor([[1, 2, 3],
                [4, 5, 6]])       # shape (2, 3)

y = x.reshape(3, 2)               # shape (3, 2)
print(y)

tensor([[1, 2],
        [3, 4],
        [5, 6]])
"""

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # integer division

        assert(self.head_dim * num_heads == embed_size), "embed_size must be divisible by num_heads"

        self.values  = nn.Linear(embed_size, embed_size)
        self.keys    = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out  = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        num_examples = query.shape[0] # number of training examples (aka batch size)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # init W_Q, W_K, W_V
        values  = self.values(values) # (num_examples, value_len, embed_size)
        keys    = self.keys(keys)     # (num_examples, key_len, embed_size)
        queries = self.queries(query) # (num_examples, query_len, embed_size)

        # slice up W_Q W_K W_V for multihead attention
        values  =   values.reshape(num_examples, self.num_heads, value_len, self.head_dim)
        keys    =     keys.reshape(num_examples, self.num_heads, key_len,   self.head_dim)
        queries =  queries.reshape(num_examples, self.num_heads, query_len, self.head_dim)


        # calculate scores for each example for each head
        # scores = torch.einsum("nhqd, nhkd -> nhqk", queries, keys)
        scores = torch.zeros(
            (num_examples, self.num_heads, query_len, key_len),
            device=queries.device,
            dtype=queries.dtype
        )
        for n in range(num_examples):
            for h in range(self.num_heads):
                # (query_len, head_dim) @ (head_dim, key_len) -> (query_len, key_len)
                q = queries[n, h, :, :]              # shape (query_len, head_dim)
                k = keys[n, h, :, :]                 # shape (key_len, head_dim)
                scores[n, h] = q @ k.transpose(0, 1) # shape (query_len, key_len)

        # mask and scale scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        scaled_scores = torch.softmax(scores / (self.embed_size ** (1/2)), dim=3)

        # calculate attention for each example for each head
        # attention = torch.einsum("nhqk, nhkd -> nhqd", scaled_scores, values)
        attention = torch.zeros(
            (num_examples, self.num_heads, query_len, self.head_dim),
            device=queries.device,
            dtype=queries.dtype
        )
        for n in range(num_examples):
            for h in range(self.num_heads):
                s = scaled_scores[n, h, :, :]
                v = values[n, h, :, :]
                attention[n, h] = s @ v

        # concatenate the heads back together
        out = attention.reshape(num_examples, query_len, self.num_heads * self.head_dim)

        # output projection of concatenated heads back into original embedding dimension
        out = self.fc_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        num_heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, num_heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        num_examples, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(num_examples, seq_len).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )
        # In the Encoder, Q, K, V are all the same matrix
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout, device):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, num_heads)
        self.transformer_block = TransformerBlock(
            embed_size, num_heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, num_heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        num_examples, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(num_examples, seq_len).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.positional_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,

        # using some values from the actual paper
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        num_heads=8,
        dropout=0,
        device="cpu",
        max_length=100
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            num_heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            num_heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx)
        src_mask = src_mask[:, None, None, :]
        return src_mask.to(self.device)


    # tril: return lower triangular half, set other all else to zero
    # expand: repeat singleton dimensions
    """
    x = torch.tensor([[1], [2], [3]])  # shape (3, 1)
    y = x.expand(3, 4)                 # shape (3, 4)
    print(y)
    tensor([[1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]])
    """
    def make_trg_mask(self, trg):
        num_examples, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            num_examples, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


def setup_seed(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(42)
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(device)

    x = torch.tensor(
        [
            [1, 5, 6, 4, 3, 9, 5, 2, 0],
            [1, 8, 7, 3, 4, 5, 6, 8, 2]
        ]
    ).to(device)

    trg = torch.tensor(
        [
            [1, 7, 4, 3, 5, 9, 2, 0],
            [1, 5, 6, 2, 4, 7, 6, 2]
        ]
    ).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        device=device
    ).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)
    print(out)



