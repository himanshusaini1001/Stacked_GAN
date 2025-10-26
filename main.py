import argparse
import torch
import numpy as np
import os
from train.training import StackedGAN
from logger_utils import GANLogger  # <-- import the logger

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
        dataset = torch.tensor(np.array(dataset), dtype=torch.long)
        return dataset, tokenizer

    else:
        raise ValueError(f"Unsupported dataset format: {ext}")

# -------------------------
# Training function
# -------------------------
def train_gan(gan, dataset, batch_size, epochs, device, logger=None):
    dataset = dataset.to(device)
    gan.generator.to(device)
    gan.discriminator.to(device)
    gan.cond_encoder.to(device)

    n_batches = int(np.ceil(len(dataset) / batch_size))

    for epoch in range(1, epochs + 1):
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()

        perm = torch.randperm(len(dataset))
        total_d, total_g = 0.0, 0.0

        for i in range(n_batches):
            idx = perm[i*batch_size : (i+1)*batch_size]
            batch = dataset[idx]
            d_loss, g_loss = gan.train_step(batch)
            total_d += d_loss
            total_g += g_loss

        avg_d = total_d / n_batches
        avg_g = total_g / n_batches

        epoch_end.record()
        torch.cuda.synchronize()
        epoch_time = epoch_start.elapsed_time(epoch_end) / 1000  # convert ms to s

        if logger:
            logger.log_epoch(epoch, epochs, avg_d, avg_g, epoch_time)
        else:
            print(f"Epoch {epoch}/{epochs} | Avg D_loss={avg_d:.4f} | Avg G_loss={avg_g:.4f} | Time={epoch_time:.2f}s")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','gen'], default='train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--data', type=str, required=True, help="Path to dataset file (TXT, FASTA)")
    parser.add_argument('--target_gc', type=float, default=0.42, help="Target GC ratio (0-1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == 'train':
        dataset, tokenizer = load_dataset(args.data, args.seq_len)
        print(f"Loaded dataset: {len(dataset)} sequences")

        # Initialize GAN
        gan = StackedGAN(seq_len=args.seq_len, vocab_size=tokenizer.vocab_size,
                         target_gc=args.target_gc, device=device)
        gan.tokenizer = tokenizer  # needed for GC penalty

        # Initialize Logger
        logger = GANLogger(mode="train")
        logger.start()
        logger.log_params(mode=args.mode, data=args.data, epochs=args.epochs, batch=args.batch,
                          seq_len=args.seq_len, target_gc=args.target_gc, device=device)

        # Train with logger
        train_gan(gan, dataset, args.batch, args.epochs, device, logger)

        logger.finish("Training")

        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        save_path = f'checkpoints/stacked_epoch{args.epochs}.pt'
        torch.save({
            "generator_state": gan.generator.state_dict(),
            "discriminator_state": gan.discriminator.state_dict(),
            "cond_encoder_state": gan.cond_encoder.state_dict(),
            "tokenizer": tokenizer
        }, save_path)
        logger.log_message(f"Checkpoint saved: {save_path}")

    elif args.mode == 'gen':
        if args.checkpoint == '':
            raise ValueError("Please provide checkpoint with --checkpoint")
        print(f"Loading checkpoint from {args.checkpoint} ...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        tokenizer = checkpoint["tokenizer"]

        # Initialize GAN structure and load weights
        gan = StackedGAN(seq_len=args.seq_len, vocab_size=tokenizer.vocab_size,
                         target_gc=args.target_gc, device=device)
        gan.tokenizer = tokenizer
        gan.generator.load_state_dict(checkpoint["generator_state"])
        gan.discriminator.load_state_dict(checkpoint["discriminator_state"])
        gan.cond_encoder.load_state_dict(checkpoint["cond_encoder_state"])

        # Initialize Logger
        logger = GANLogger(mode="gen")
        logger.start()
        logger.log_params(mode=args.mode, checkpoint=args.checkpoint, n=args.n, seq_len=args.seq_len, device=device)

        # Generate sequences
        sequences = gan.generate(n_samples=args.n)
        GANLogger.save_sequences(sequences, tokenizer)

        logger.log_message(f"Generated {args.n} sequences")
        logger.finish("Generation")


if __name__ == '__main__':
    main()
