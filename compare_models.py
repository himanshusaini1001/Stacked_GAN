# compare_models.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pandas as pd
import editdistance
from scipy.stats import entropy
from sdv.single_table.ctgan import CTGAN
from train.training import StackedGAN
from mainex import load_dataset, CharTokenizer

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# GAN Models
# ----------------------------

# ----------------------------
# CGAN Models
# ----------------------------
class CGAN_Gen(nn.Module):
    def __init__(self, z_dim, hidden_dim, seq_len, vocab_size, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(z_dim + num_classes, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, z, labels):
        # One-hot encode labels
        labels_oh = torch.zeros(labels.size(0), 4, device=z.device)
        labels_oh.scatter_(1, labels.unsqueeze(1), 1.0)
        
        x = torch.cat([z, labels_oh], dim=1)
        x = torch.relu(self.fc(x)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return F.softmax(self.out(out), dim=-1)

class CGAN_Disc(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size + num_classes, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x, labels):
        labels_oh = torch.zeros(labels.size(0), 4, device=x.device)
        labels_oh.scatter_(1, labels.unsqueeze(1), 1.0)
        labels_oh = labels_oh.unsqueeze(1).repeat(1, x.size(1), 1)
        
        x = torch.cat([x, labels_oh], dim=-1)
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        return self.fc(out)

# ----------------------------
# CramerGAN Models
# ----------------------------
class CramerGAN_Gen(nn.Module):
    def __init__(self, z_dim, hidden_dim, seq_len, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return F.softmax(self.out(out), dim=-1)

class CramerGAN_Disc(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        return self.fc2(out)

# ----------------------------
# DraGAN Models (with gradient penalty)
# ----------------------------
class DraGAN_Gen(nn.Module):
    def __init__(self, z_dim, hidden_dim, seq_len, vocab_size):
        super().__init__()
        self.fc = nn.Linear(z_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, z):
        B = z.size(0)
        x = torch.relu(self.fc(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return F.softmax(self.out(out), dim=-1)

class DraGAN_Disc(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        return self.fc(out)

# ----------------------------
# WGAN Models
# ----------------------------
class WGAN_Gen(nn.Module):
    def __init__(self, z_dim, hidden_dim, seq_len, vocab_size):
        super().__init__()
        self.fc = nn.Linear(z_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, z):
        B = z.size(0)
        x = torch.relu(self.fc(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return F.softmax(self.out(out), dim=-1)


class WGAN_Disc(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        return self.fc(out)

# ----------------------------
# Metrics
# ----------------------------
def gc_content(seqs, tokenizer):
    g_idx = tokenizer.char2idx.get("G", None)
    c_idx = tokenizer.char2idx.get("C", None)
    if g_idx is None or c_idx is None:
        return 0.0
    arr = np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else np.array(s) for s in seqs])
    gc = ((arr == g_idx) | (arr == c_idx)).mean()
    return float(gc)

def kmer_distribution(seqs, tokenizer, k=3):
    def seq_to_kmers(seq):
        s = tokenizer.decode(seq)
        return [s[i:i+k] for i in range(len(s)-k+1)]
    kmers = []
    for seq in seqs:
        kmers.extend(seq_to_kmers(seq))
    counts = Counter(kmers)
    total = sum(counts.values())
    return {k: v/total for k,v in counts.items()}

def js_divergence(p, q):
    all_keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k,0) for k in all_keys])
    q_vec = np.array([q.get(k,0) for k in all_keys])
    p_vec /= (p_vec.sum() + 1e-9)
    q_vec /= (q_vec.sum() + 1e-9)
    m = 0.5 * (p_vec + q_vec)
    return float(0.5 * (entropy(p_vec, m) + entropy(q_vec, m)))

def uniqueness_ratio(seqs):
    unique = len(set(tuple(s) for s in seqs))
    return float(unique / len(seqs))

def avg_edit_distance(fake, real, tokenizer):
    real_strs = [tokenizer.decode(s) for s in real]
    fake_strs = [tokenizer.decode(s) for s in fake]
    dists = [min(editdistance.eval(f,r) for r in real_strs)/max(len(f),1) for f in fake_strs]
    return float(np.mean(dists))

def motif_score(seqs, tokenizer, motif="ATG"):
    count = 0
    total = 0
    motif_len = len(motif)
    for s in seqs:
        seq_str = tokenizer.decode(s)
        for i in range(len(seq_str)-motif_len+1):
            if seq_str[i:i+motif_len] == motif:
                count += 1
            total += 1
    return float(count / total if total > 0 else 0)

# ----------------------------
# Training WGAN
# ----------------------------
def train_wgan(dataset, tokenizer, seq_len, epochs=10, batch=64, z_dim=128):
    G = WGAN_Gen(z_dim, 128, seq_len, tokenizer.vocab_size).to(device)
    D = WGAN_Disc(tokenizer.vocab_size, 128, seq_len).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.9))

    dataset = dataset.to(device)
    losses_g, losses_d = [], []
    dataset_size = len(dataset)

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch):
            batch_idx = perm[i:i+batch]
            real = dataset[batch_idx]
            cur_batch = real.size(0)
            real_oh = torch.zeros(cur_batch, seq_len, tokenizer.vocab_size, device=device)
            real_oh.scatter_(2, real.unsqueeze(-1), 1.0)

            # Train D
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z).detach()
            d_loss = D(fake).mean() - D(real_oh).mean()
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train G
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z)
            g_loss = -D(fake).mean()
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            losses_g.append(float(g_loss.item()))
            losses_d.append(float(d_loss.item()))

    return G, losses_g, losses_d, device

# ----------------------------
# Training CGAN
# ----------------------------
def train_cgan(dataset, tokenizer, seq_len, epochs=10, batch=64, z_dim=128):
    G = CGAN_Gen(z_dim, 128, seq_len, tokenizer.vocab_size).to(device)
    D = CGAN_Disc(tokenizer.vocab_size, 128, seq_len).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    dataset = dataset.to(device)
    losses_g, losses_d = [], []
    dataset_size = len(dataset)

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch):
            batch_idx = perm[i:i+batch]
            real = dataset[batch_idx]
            cur_batch = real.size(0)
            real_oh = torch.zeros(cur_batch, seq_len, tokenizer.vocab_size, device=device)
            real_oh.scatter_(2, real.unsqueeze(-1), 1.0)
            
            # Generate random labels
            labels = torch.randint(0, 4, (cur_batch,), device=device)

            # Train D
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z, labels).detach()
            d_loss = (D(fake, labels) - D(real_oh, labels)).mean()
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train G
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z, labels)
            g_loss = -D(fake, labels).mean()
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            losses_g.append(float(g_loss.item()))
            losses_d.append(float(d_loss.item()))

    return G, losses_g, losses_d, device

# ----------------------------
# Training CramerGAN
# ----------------------------
def train_cramerga(dataset, tokenizer, seq_len, epochs=10, batch=64, z_dim=128):
    G = CramerGAN_Gen(z_dim, 128, seq_len, tokenizer.vocab_size).to(device)
    D = CramerGAN_Disc(tokenizer.vocab_size, 128, seq_len).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.9))

    dataset = dataset.to(device)
    losses_g, losses_d = [], []
    dataset_size = len(dataset)

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch):
            batch_idx = perm[i:i+batch]
            real = dataset[batch_idx]
            cur_batch = real.size(0)
            real_oh = torch.zeros(cur_batch, seq_len, tokenizer.vocab_size, device=device)
            real_oh.scatter_(2, real.unsqueeze(-1), 1.0)

            # Train D
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z).detach()
            d_loss = D(fake).mean() - D(real_oh).mean()
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train G
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z)
            g_loss = -D(fake).mean()
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            losses_g.append(float(g_loss.item()))
            losses_d.append(float(d_loss.item()))

    return G, losses_g, losses_d, device

# ----------------------------
# Training DraGAN
# ----------------------------
def train_dragan(dataset, tokenizer, seq_len, epochs=10, batch=64, z_dim=128, lambda_gp=10):
    G = DraGAN_Gen(z_dim, 128, seq_len, tokenizer.vocab_size).to(device)
    D = DraGAN_Disc(tokenizer.vocab_size, 128, seq_len).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.9))

    dataset = dataset.to(device)
    losses_g, losses_d = [], []
    dataset_size = len(dataset)

    def gradient_penalty(discriminator, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        with torch.backends.cudnn.flags(enabled=False):
            d_interpolated = discriminator(interpolated)
            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones_like(d_interpolated),
                create_graph=True,
                retain_graph=True
            )[0]
        
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch):
            batch_idx = perm[i:i+batch]
            real = dataset[batch_idx]
            cur_batch = real.size(0)
            real_oh = torch.zeros(cur_batch, seq_len, tokenizer.vocab_size, device=device)
            real_oh.scatter_(2, real.unsqueeze(-1), 1.0)

            # Train D
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z).detach()
            d_loss = D(fake).mean() - D(real_oh).mean()
            gp = gradient_penalty(D, real_oh, fake)
            d_loss += lambda_gp * gp
            
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train G
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z)
            g_loss = -D(fake).mean()
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            losses_g.append(float(g_loss.item()))
            losses_d.append(float(d_loss.item()))

    return G, losses_g, losses_d, device

# ----------------------------
# Evaluate Models (with CSV caching)
# ----------------------------
def evaluate_models(real, tokenizer, seq_len, csv_path="comparison_results_gpu.csv"):
    if os.path.exists(csv_path):
        print(f"{csv_path} exists. Loading cached metrics.")
        df = pd.read_csv(csv_path, index_col=0)
        results_json = df.to_dict(orient="index")
        results_json = {m: {k: float(v) for k,v in metrics.items()} for m, metrics in results_json.items()}
        return results_json

    results = {}
    real_gc = gc_content(real, tokenizer)
    real_kmer = kmer_distribution(real, tokenizer)

    # --- WGAN ---
    print("Training WGAN...")
    G_wgan, _, _, dev = train_wgan(real, tokenizer, seq_len, epochs=5)  # shorter for testing
    z = torch.randn(200, 128, device=dev)
    fake_probs = G_wgan(z).detach()
    fake_idx = fake_probs.argmax(-1).cpu().numpy()
    results["WGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, kmer_distribution(fake_idx, tokenizer)),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer)
    }

    # --- CTGAN ---
    print("Training CTGAN (CPU)...")
    seq_strings = [tokenizer.decode(s) for s in real]
    df_ct = pd.DataFrame(seq_strings, columns=["sequence"])
    ctgan = CTGAN(epochs=20)
    ctgan.fit(df_ct, discrete_columns=["sequence"])
    fake_df = ctgan.sample(200)
    fake_idx = [tokenizer.encode(s, seq_len) for s in fake_df["sequence"]]
    results["CTGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, kmer_distribution(fake_idx, tokenizer)),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer)
    }

    # --- CGAN ---
    print("Training CGAN...")
    G_cgan, _, _, dev = train_cgan(real, tokenizer, seq_len, epochs=5)
    z = torch.randn(200, 128, device=dev)
    labels = torch.randint(0, 4, (200,), device=dev)
    fake_probs = G_cgan(z, labels).detach()
    fake_idx = fake_probs.argmax(-1).cpu().numpy()
    results["CGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, kmer_distribution(fake_idx, tokenizer)),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer)
    }

    # --- CramerGAN ---
    print("Training CramerGAN...")
    G_cramer, _, _, dev = train_cramerga(real, tokenizer, seq_len, epochs=5)
    z = torch.randn(200, 128, device=dev)
    fake_probs = G_cramer(z).detach()
    fake_idx = fake_probs.argmax(-1).cpu().numpy()
    results["CramerGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, kmer_distribution(fake_idx, tokenizer)),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer)
    }

    # --- DraGAN ---
    print("Training DraGAN...")
    G_dragan, _, _, dev = train_dragan(real, tokenizer, seq_len, epochs=5)
    z = torch.randn(200, 128, device=dev)
    fake_probs = G_dragan(z).detach()
    fake_idx = fake_probs.argmax(-1).cpu().numpy()
    results["DraGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, kmer_distribution(fake_idx, tokenizer)),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer)
    }

    # --- StackedGAN ---
    print("Training StackedGAN...")
    gan_model = StackedGAN(seq_len=seq_len, vocab_size=tokenizer.vocab_size, device=dev)
    gan_model.tokenizer = tokenizer
    dataset_size = len(real)
    for epoch in range(10):  # shorter for testing
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, 64):
            batch_idx = perm[i:i+64]
            batch = real[batch_idx].to(dev)
            gan_model.train_step(batch)

    fake_idx = gan_model.generate(200)
    results["StackedGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, kmer_distribution(fake_idx, tokenizer)),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer)
    }

    # --- Convert results to JSON-friendly ---
    results_json = {model: {k: float(v) for k,v in metrics.items()} for model, metrics in results.items()}

    # Save CSV for caching
    df_res = pd.DataFrame(results_json).T
    df_res.to_csv(csv_path)
    print(f"Saved metrics to {csv_path}")

    return results_json

# ----------------------------
# Standalone run
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--csv_path", type=str, default="comparison_results_gpu.csv")
    args = parser.parse_args()

    dataset, tokenizer = load_dataset(args.data, args.seq_len)
    results = evaluate_models(dataset, tokenizer, args.seq_len, csv_path=args.csv_path)
    print(results)
