# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tqdm",
#   "wandb",
#   "accelerate",
#   "einops",
#   "fire",
#   "PoPE-pytorch",
#   "x-transformers"
# ]
# ///

import random
import tqdm
import gzip
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from x_transformers.autoregressive_wrapper import top_k

from accelerate import Accelerator
from PoPE_pytorch import PoPE
from PoPE_pytorch.attention import flash_attn_with_pope

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# modules

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

class CausalAttention(nn.Module):
    def __init__(self, dim, heads = 8, use_pope = True):
        super().__init__()
        self.heads = heads
        self.use_pope = use_pope
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim, bias = False)

    def forward(self, x, pos_emb = None, cache = None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if exists(cache):
            ck, cv = cache
            k, v = (torch.cat(t, dim = -2) for t in ((ck, k), (cv, v)))

        new_cache = (k, v)

        if self.use_pope and exists(pos_emb):
            out = flash_attn_with_pope(
                q, k, v,
                pos_emb = pos_emb,
                causal = True,
                softmax_scale = self.scale,
                fused = True,
                head_dimension_at_first = True
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal = True, scale = self.scale)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), new_cache

# simple transformer

class SimpleTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        heads,
        seq_len,
        use_pope = True,
    ):
        super().__init__()
        self.max_seq_len = seq_len
        self.use_pope = use_pope

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(2048, dim) if not use_pope else None
        self.pope = PoPE(dim // heads, heads = heads) if use_pope else None

        self.layers = nn.ModuleList([nn.ModuleList([
            RMSNorm(dim),
            CausalAttention(dim, heads = heads, use_pope = use_pope),
            RMSNorm(dim),
            FeedForward(dim = dim),
        ]) for _ in range(depth)])

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(self, x, cache = None, return_cache = False):
        seq_len = x.shape[1]
        x = self.token_emb(x)

        seq_len_kv = seq_len if not exists(cache) else (cache[0][0].shape[-2] + seq_len)

        if self.use_pope:
            pos_emb = self.pope(seq_len_kv)
        else:
            pos_emb = None
            x = x + self.pos_emb(torch.arange(seq_len_kv - seq_len, seq_len_kv, device = x.device))

        new_caches = []
        for i, (norm1, attn, norm2, ff) in enumerate(self.layers):
            layer_cache = cache[i] if exists(cache) else None
            attn_out, new_layer_cache = attn(norm1(x), pos_emb, cache = layer_cache)
            new_caches.append(new_layer_cache)
            x = x + attn_out
            x = x + ff(norm2(x))

        logits = self.to_logits(self.norm(x))

        if return_cache:
            return logits, new_caches

        return logits

    @torch.no_grad()
    def generate(self, prompts, seq_len, temperature = 1.0, filter_thres = 0.9):
        b, t = prompts.shape
        out = prompts
        cache = None

        for _ in tqdm.tqdm(range(seq_len), desc='generating'):
            curr_x = out[:, -self.max_seq_len:] if not exists(cache) else out[:, -1:]
            logits, cache = self.forward(curr_x, cache = cache, return_cache = True)
            logits = logits[:, -1]

            # top-k filtering
            logits = top_k(logits, thres = filter_thres)

            probs = F.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
        return out[:, t:]

# autoregressive training logic
def autoregressive_loss(model, x):
    # x shape: (b, n)
    inputs = x[:, :-1]
    targets = x[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss

# training

def train(
    num_batches = int(1e5),
    batch_size = 4,
    gradient_accumulate_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    generate_every = 250,
    generate_len = None,
    seq_len = 128,
    dim = 512,
    depth = 6,
    heads = 8,
    use_pope = True,
    use_wandb = False,
    cpu = False,
):
    generate_len = min(default(generate_len, seq_len), seq_len)
    run_name = 'pope' if use_pope else 'abs_pos'
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    model = SimpleTransformer(
        num_tokens = 256,
        dim = dim,
        depth = depth,
        heads = heads,
        seq_len = seq_len,
        use_pope = use_pope,
    )

    # data

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq.squeeze(0)

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset   = TextSamplerDataset(data_val, seq_len)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True, num_workers = 2)
    val_loader   = DataLoader(val_dataset, batch_size = batch_size, drop_last = True, num_workers = 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    # wandb

    if use_wandb:
        import wandb
        wandb.init(project = 'pope-enwik8', name = run_name)

    # prepare

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader   = cycle(val_loader)

    # train loop

    pbar = tqdm.tqdm(range(num_batches), mininterval = 1., desc = run_name)

    for i in pbar:
        model.train()

        for _ in range(gradient_accumulate_every):
            loss = autoregressive_loss(model, next(train_loader))
            accelerator.backward(loss / gradient_accumulate_every)

        train_loss = loss.item()
        pbar.set_postfix(loss = f'{train_loss:.4f}')

        if use_wandb:
            wandb.log(dict(loss = train_loss), step = i)

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()

        if i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = autoregressive_loss(model, next(val_loader)).item()
                pbar.set_postfix(loss = f'{train_loss:.4f}', val = f'{val_loss:.4f}')
                if use_wandb:
                    wandb.log(dict(valid_loss = val_loss), step = i)

        if i % generate_every == 0 and accelerator.is_main_process:
            model.eval()
            inp = random.choice(val_dataset)[:-1].unsqueeze(0).to(device)
            prime = decode_tokens(inp[0].cpu().numpy())

            sample = accelerator.unwrap_model(model).generate(
                prompts = inp,
                seq_len = generate_len
            )

            output_str = decode_tokens(sample[0].cpu().numpy())
            print(f'\n{"=" * 80}')
            print(f'[prime] {prime[:80]}...')
            print(f'{"=" * 80}')
            print(f'[generated] {output_str[:200]}')
            print(f'{"=" * 80}\n')

if __name__ == '__main__':
    import fire
    fire.Fire(train)
