import argparse
import torch
import numpy as np
import os
import pandas as pd
from train.training import StackedGAN

# -------------------------
# Tokenizer utilities
# -------------------------
class CharTokenizer:
    def __init__(self, sequences):
        chars = sorted(set("".join(sequences)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, seq, seq_len):
        arr = [self.char2idx.get(ch, 0) for ch in seq]
        if len(arr) < seq_len:
            arr += [0] * (seq_len - len(arr))
        return arr[:seq_len]

    def decode(self, arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.tolist()
        if isinstance(arr, int):
            arr = [arr]
        # Clamp indices to valid vocab size
        arr = [min(max(i, 0), self.vocab_size - 1) for i in arr]
        return "".join([self.idx2char.get(i, "?") for i in arr])


# -------------------------
# Dataset Loader
# -------------------------
def load_dataset(file_path, seq_len):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".fasta", ".fa"]:
        raw_data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(">"):
                    raw_data.append(line)
        tokenizer = CharTokenizer(raw_data)
        dataset = [tokenizer.encode(seq, seq_len) for seq in raw_data]
        dataset = torch.tensor(np.array(dataset))
        return dataset, tokenizer

    elif ext == ".csv":
        df = pd.read_csv(file_path, header=None)
        dataset = torch.tensor(df.values).float()
        return dataset, None
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','gen'], default='train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--data', type=str, required=True, help="Path to dataset file (TXT, FASTA, CSV)")
    parser.add_argument('--target_gc', type=float, default=0.42, help="Target GC ratio (0-1)")
    args = parser.parse_args()

    if args.mode == 'train':
        dataset, tokenizer = load_dataset(args.data, args.seq_len)
        print(f"Loaded dataset: {len(dataset)} sequences")

        # Initialize GAN with target GC
        gan = StackedGAN(seq_len=args.seq_len, vocab_size=tokenizer.vocab_size, target_gc=args.target_gc)
        gan.tokenizer = tokenizer  # assign tokenizer for decoding + GC bias

        for epoch in range(args.epochs):
            idx = np.random.choice(len(dataset), args.batch)
            batch = dataset[idx]
            d_loss, g_loss = gan.train_step(batch)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

        os.makedirs('checkpoints', exist_ok=True)
        save_path = f'checkpoints/stacked_epoch{args.epochs}.pt'
        torch.save({"gan": gan, "tokenizer": tokenizer}, save_path)
        print(f"Checkpoint saved: {save_path}")

    elif args.mode == 'gen':
        if args.checkpoint == '':
            raise ValueError("Please provide checkpoint with --checkpoint")

        print(f"Loading checkpoint from {args.checkpoint} ...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        gan = checkpoint["gan"]
        tokenizer = checkpoint["tokenizer"]

        seqs = gan.generate(args.n)

        os.makedirs('outputs', exist_ok=True)
        out_file = "outputs/generated.txt"

        with open(out_file, 'w') as f:
            for s in seqs:
                if isinstance(s, torch.Tensor):
                    s = s.tolist()
                elif isinstance(s, int):
                    s = [s]
                decoded = tokenizer.decode(s)
                f.write(decoded + '\n')

        print(f"Generated sequences saved to {out_file}")


if __name__ == '__main__':
    main()
