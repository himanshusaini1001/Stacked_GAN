import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import os

# -------------------------
# Import GAN class
# -------------------------
from train.training import StackedGAN  # adjust path if needed

# -------------------------
# CharTokenizer
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
        return "".join([self.idx2char.get(i, "?") for i in arr])

# -------------------------
# GUI functions
# -------------------------
gan = None
tokenizer = None

CHECKPOINT_PATH = "checkpoints/stacked_epoch130.pt"  # default

def load_checkpoint_default():
    global gan, tokenizer
    if not os.path.exists(CHECKPOINT_PATH):
        messagebox.showerror("Error", f"Checkpoint not found:\n{CHECKPOINT_PATH}")
        return
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        gan = checkpoint["gan"]
        tokenizer = checkpoint["tokenizer"]
        messagebox.showinfo("Loaded", f"Checkpoint loaded from {CHECKPOINT_PATH}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load checkpoint:\n{e}")

def generate_and_display():
    if gan is None or tokenizer is None:
        messagebox.showwarning("Warning", "Checkpoint not loaded!")
        return
    try:
        n = int(num_sequences_entry.get())
        sequences_int = gan.generate(n_samples=n)
        sequences_str = [tokenizer.decode(seq) for seq in sequences_int]

        text_box.config(state=tk.NORMAL)
        text_box.delete("1.0", tk.END)
        for i, seq in enumerate(sequences_str, 1):
            text_box.insert(tk.END, f">Seq{i}\n{seq}\n")
        text_box.config(state=tk.DISABLED)

        # GC content
        gc_count = sum(seq.count("G") + seq.count("C") for seq in sequences_str)
        total_bases = sum(len(seq) for seq in sequences_str)
        gc_content = (gc_count / total_bases * 100) if total_bases > 0 else 0
        gc_label.config(text=f"GC Content: {gc_content:.2f}%")
        progress_bar['value'] = gc_content
    except Exception as e:
        messagebox.showerror("Error", str(e))

def save_fasta():
    sequences_text = text_box.get("1.0", tk.END).strip()
    if not sequences_text:
        messagebox.showwarning("Warning", "No sequences to save!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".fasta",
                                             filetypes=[("FASTA files","*.fasta")])
    if file_path:
        with open(file_path, "w") as f:
            f.write(sequences_text)
        messagebox.showinfo("Saved", f"Sequences saved to {file_path}")

# -------------------------
# Tkinter GUI
# -------------------------
root = tk.Tk()
root.title("DNA Sequence Generator")
root.geometry("900x650")
root.configure(bg="#f4f6f8")

# Header
header = tk.Label(root, text="DNA Sequence Generator", font=("Helvetica", 20, "bold"), bg="#f4f6f8")
header.pack(pady=15)

# Controls frame
controls_frame = tk.Frame(root, bg="#f4f6f8")
controls_frame.pack(pady=5, padx=15, fill=tk.X)

tk.Label(controls_frame, text="Number of sequences:", font=("Helvetica", 12), bg="#f4f6f8").pack(side=tk.LEFT, padx=5)
num_sequences_entry = tk.Entry(controls_frame, width=7, font=("Helvetica", 12))
num_sequences_entry.pack(side=tk.LEFT, padx=5)
num_sequences_entry.insert(0, "100")

generate_btn = tk.Button(controls_frame, text="Generate", bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
                         activebackground="#45a049", command=generate_and_display)
generate_btn.pack(side=tk.LEFT, padx=10)

download_btn = tk.Button(controls_frame, text="Download as FASTA", bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"),
                         activebackground="#1976D2", command=save_fasta)
download_btn.pack(side=tk.LEFT, padx=10)

# GC Progress
gc_frame = tk.Frame(root, bg="#f4f6f8")
gc_frame.pack(pady=10, padx=15, fill=tk.X)

gc_label = tk.Label(gc_frame, text="GC Content: 0%", font=("Helvetica", 12, "bold"), bg="#f4f6f8")
gc_label.pack(side=tk.LEFT, padx=5)

progress_bar = ttk.Progressbar(gc_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(side=tk.LEFT, padx=10)
progress_bar['maximum'] = 100

# Text box with scrollbar
text_frame = tk.Frame(root, bg="#f4f6f8", bd=1, relief=tk.SUNKEN)
text_frame.pack(expand=True, fill=tk.BOTH, padx=15, pady=10)

text_scroll = tk.Scrollbar(text_frame)
text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

text_box = tk.Text(text_frame, wrap=tk.NONE, yscrollcommand=text_scroll.set, font=("Consolas", 11))
text_box.pack(expand=True, fill=tk.BOTH)
text_box.config(state=tk.DISABLED)

text_scroll.config(command=text_box.yview)

# Load checkpoint automatically
load_checkpoint_default()

root.mainloop()
