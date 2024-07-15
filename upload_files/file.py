import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model

class CustomGPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2Model(config).h[0].attn
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2Model(config).h[0].mlp
        self.gelu = nn.GELU()

    def forward(self, x):
        a = self.attn(self.ln_1(x))[0]
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + self.gelu(m)
        return x

class CustomGPT2(nn.Module):
    def __init__(self, vocab_size, n_embd=768, n_layer=12, n_head=12, dropout=0.1):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=1024,
            dropout=dropout,
        )
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([CustomGPT2Block(config) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if input is 1D
        
        b, t = x.size()
        assert t <= 1024, f"Cannot forward sequence of length {t}, max length is 1024"

        token_embeddings = self.tok_emb(x)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx