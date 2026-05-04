import numpy as np
import torch
import torch.nn as nn

"""

Attention(Q, K, V) = softmax( (Q * K^T)/sqrt(d_k) ) * V

head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V )

Multihead(Q, K, V) = Concat(all head_i) * W^O


Q K and V are originally passed in with shape: (num_examples, seq_len, embed_size)
We want to slice them up for multihead attention, so:

               d_model                  d_k                 d_k
            *-----------*            *--------*           *--------*
            |           |            |        |           |        |
    seq_len |           |    d_model |        |   seq_len |        |
            |     Q     |            | W_i^Q  |           |  Q_i'  |
            |           | DOT        |        | =         |        |
            |           |            |        |           |        |
            *-----------*            *--------*           *--------*

               d_model                  d_k                 d_k
            *-----------*            *--------*           *--------*
            |           |            |        |           |        |
    seq_len |           |    d_model |        |   seq_len |        |
            |     K     |            | W_i^K  |           |  K_i'  |
            |           | DOT        |        | =         |        |
            |           |            |        |           |        |
            *-----------*            *--------*           *--------*

               d_model                  d_k                 d_k
            *-----------*            *--------*           *--------*
            |           |            |        |           |        |
    seq_len |           |    d_model |        |   seq_len |        |
            |     V     |            | W_i^V  |           |  V_i'  |
            |           | DOT        |        | =         |        |
            |           |            |        |           |        |
            *-----------*            *--------*           *--------*

               d_k                                           seq_len
            *--------*               seq_len               *-----------*
            |        |            *------------*           |           |
    seq_len |        |        d_k |            |   seq_len |           |
            | Q_i^Q  |            |   K_i'^T   |           | scores_i  |
            |        | DOT        |            | =         |           |
            |        |            *------------*           |           |
            *--------*                                     *-----------*

               seq_len                  d_k                 d_k
            *-----------*            *--------*           *--------*
            |           |            |        |           |        |
    seq_len |           |    seq_len |        |   seq_len |        |
            | scores_i  |            |  V_i'  |           | head_i |
            |           | DOT        |        | =         |        |
            |           |            |        |           |        |
            *-----------*            *--------*           *--------*

"""

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # integer division

        assert(self.head_dim * num_heads == embed_size), "embed_size must be divisible by num_heads"

        self.values     = nn.Linear(embed_size, embed_size)
        self.keys       = nn.Linear(embed_size, embed_size)
        self.queries    = nn.Linear(embed_size, embed_size)
        self.fc_out     = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        """
        values, keys, queries shape: (batch_size, seq_len, embed_size)
        mask shape: (batch_size, 1, 1, seq_len) for some attention masking scenarios
        """

        num_examples = query.shape[0] # number of training examples (aka batch size)

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        values  = values.reshape(num_examples, value_len, self.num_heads, self.head_dim)
        keys    =   keys.reshape(num_examples, key_len,   self.num_heads, self.head_dim)
        queries =  query.reshape(num_examples, query_len, self.num_heads, self.head_dim)

        # queries shape: (num_examples, query_len, num_heads, head_dim)
        # keys shape:    (num_examples, key_len,   num_heads, head_dim)
        # scores shape:  (num_examples, num_heads, num_heads, key_len )

        # scores = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        # or equivalently... (for clarity, not speed)
        # naive matmul
        scores = torch.zeros(
            (num_examples, self.num_heads, query_len, key_len),
            device=queries.device,
            dtype=queries.dtype
        )
        for n in range(num_examples):
            for h in range(self.num_heads):
                # (query_len, head_dim) @ (head_dim, key_len) -> (query_len, key_len)
                q = queries[n, :, h, :] # shape (query_len, head_dim)
                k = keys[n, :, h, :] # shape (head_dim, key_len)
                scores[n, h] = q @ k.transpose(0, 1)

        # queries shape: (num_examples, query_len, num_heads, head_dim)
        # keys shape:    (num_examples, key_len,   num_heads, head_dim)
        # scores:        (num_examples, num_heads, query_len, key_len)

        # Causality: the transformer must only work with present or previously seen words.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        # Normalize the scores
        attention = torch.softmax(scores / (self.embed_size ** (1/2)), dim=3)
        # attention shape: (num_examples, num_heads, query_len, key_len)

        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            num_examples, query_len, self.num_heads * self.head_dim
        )

        # attention shape: (num_examples, num_heads, query_len, key_len)
        # values shape: (num_examples, value_len, num_heads, head_dim)
        # out after matrix multiply: (N, query_len, num_heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # out shape: (num_examples, query_len, embed_size)
        
        return out



class TransformerBlock(nn.Module):
    # forward_expansion multiplies the embedding size in the
    # feedforward network and allows it to operate in a larger hidden space
    # before projecting back down to original embedding size
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
        # example:
        # positions = torch.arange(0, 5)
        #   [0, 1, 2, 3, 4]
        # positions = positions.expand(3, 5) 
        #   tensor([
        #       [0, 1, 2, 3, 4],
        #       [0, 1, 2, 3, 4],
        #       [0, 1, 2, 3, 4]
        #   ])
        #
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
        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (num_examples, 1, 1, src_len)

        # or equivalently...

        # boolean mask where True means "not a padding element"
        src_mask = (src != self.src_pad_idx)

        # reshape from (N, src_len) to (N, 1, 1, src_len)
        src_mask = src_mask[:, None, None, :]
        return src_mask.to(self.device)


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



