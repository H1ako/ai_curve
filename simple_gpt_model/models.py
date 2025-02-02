import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(
        self, vocab_size, n_embed, block_size, num_heads, n_layers=3, dropout=0.5
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_emabedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embed=n_embed,
                    block_size=block_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ],
            nn.LayerNorm(n_embed),
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # {B, T, C}
        pos_emb = self.position_emabedding_table(
            torch.arange(T, device=idx.device)
        )  # {T, C}
        x = token_emb + pos_emb  # {B, T, C}
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # {B, T, vocab_size}

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # {B, C}

            probs = F.softmax(logits, dim=-1)  # {B, C}
            idx_next = torch.multinomial(probs, num_samples=1)  # {B, 1}
            idx = torch.cat((idx, idx_next), dim=1)  # {B, T+1}

        return idx


class Block(nn.Module):
    def __init__(self, n_embed, block_size, num_heads, dropout=0.5):
        super().__init__()

        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(n_embed, head_size, block_size, num_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, head_size, block_size, num_heads, dropout=0.5):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                Head(
                    n_embed=n_embed,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads], dim=-1
        )  # {B, T, C} -> {B, T, C*n_heads}
        out = self.dropout(self.proj(out))  # {B, T, C*n_heads} -> {B, T, C}
        return out


class Head(nn.Module):
    # one head of self-attention

    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()

        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.k(x)  # {B, T, H}
        q = self.q(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # {B, T, T}
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # {B, T, T}  - decoder block(so it doesnt communicate with the future)
        wei = F.softmax(wei, dim=-1)  # {B, T, T}
        wei = self.dropout(wei)

        v = self.v(x)
        out = wei @ v  # {B, T, C}

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
