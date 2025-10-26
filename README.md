# Stacked Sequence GAN Project

This project implements a **Stacked GAN** for generating synthetic  DNA-like sequences composed of the alphabet `A, T, G, C`.  
It demonstrates how to stack two GANs end-to-end: one for coarse sequence generation, and another for refinement.

⚠️ **Note:** This project is for **educational purposes only** and uses synthetic toy data.  
Do **not** use for real biological/genomic applications.

---

## 📂 Project Structure
- `main.py` → entry point (training & generation)
- `models/` → contains generators and discriminators
- `utils/` → helper functions (data encoding, toy sequence generator, sampling)
- `train/` → stacked GAN training logic
- `checkpoints/` → saved model checkpoints
- `outputs/` → generated sequences

---

## 🔧 Installation
Clone or extract the project, then install requirements:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the GAN
Train the stacked GAN on toy sequences:

```bash
python main.py --mode train --epochs 50 --batch 64
```

- Trains for 50 epochs on synthetic data of length 30.  
- Saves checkpoints to `checkpoints/stacked_epoch50.pt`.

### 2. Generate Sequences
Generate synthetic sequences from a trained model:

```bash
python main.py --mode gen --checkpoint checkpoints/stacked_epoch50.pt --n 20
```

- Generates 20 sequences.  
- Saves them into `outputs/generated.txt`.

---

## 🛠️ Arguments

| Argument         | Default | Description |
|------------------|---------|-------------|
| `--mode`         | train   | Choose `train` or `gen` |
| `--epochs`       | 50      | Training epochs |
| `--batch`        | 64      | Training batch size |
| `--seq_len`      | 30      | Length of sequences |
| `--checkpoint`   | ""      | Path to model checkpoint (for generation) |
| `--n`            | 20      | Number of sequences to generate |

---

## 📜 License
Educational use only. Do not apply to real biological data without proper domain expertise and biosafety compliance.

# Train
python main.py --mode train --data data/example_genome.fasta --epochs 150 --batch 64 --seq_len 70 --target_gc 0.4

# Generate sequences
python main.py --mode gen --checkpoint checkpoints/stacked_epoch150.pt --n 150 --data data/example_genome.fasta

#Heatmap
python analyze_sequences.py

Compare result 
python compare_models.py --data data/generated_sequences.fasta --seq_len 70


python main.py --mode train --data "C:\Users\himan\Desktop\4th year Gen ai\stacked_seq_ganHybrid_cursor\data\training.fasta" --epochs 100 --batch 64 --seq_len 70 --target_gc 0.42

python main.py --mode gen --checkpoint checkpoints/stacked_epoch50.pt --n 20 --data data/training.fasta --seq_len 70

