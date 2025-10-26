# analyze_sequences.py

from utils.sequence_heatmap import SequenceHeatmap
import matplotlib.pyplot as plt

# -------------------------
# Files to analyze
# -------------------------
generated_file = "outputs/generated.txt"   # GAN-generated sequences
real_file = "data/real_sequences.txt"      # Optional: real genome sequences

# -------------------------
# Load generated sequences
# -------------------------
sh_generated = SequenceHeatmap(generated_file)
print(f"Loaded {len(sh_generated.sequences)} generated sequences")
sh_generated.plot_heatmap(title="Generated Sequences Heatmap")

# -------------------------
# Optional: Load real sequences
# -------------------------
try:
    sh_real = SequenceHeatmap(real_file)
    print(f"Loaded {len(sh_real.sequences)} real sequences")
    
    # Plot side-by-side heatmaps
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    axs[0].imshow(sh_real.get_frequency_matrix(), cmap='hot', interpolation='nearest', aspect='auto')
    axs[0].set_title("Real Sequences Heatmap")
    axs[0].set_yticks(range(4))
    axs[0].set_yticklabels(['A','C','G','T'])
    
    axs[1].imshow(sh_generated.get_frequency_matrix(), cmap='hot', interpolation='nearest', aspect='auto')
    axs[1].set_title("Generated Sequences Heatmap")
    axs[1].set_yticks(range(4))
    axs[1].set_yticklabels(['A','C','G','T'])
    
    plt.show()
except FileNotFoundError:
    print(f"Real sequences file not found: {real_file}. Only generated sequences heatmap will be shown.")
