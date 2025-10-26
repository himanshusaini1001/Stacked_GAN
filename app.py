#app.py
from flask import Flask, render_template, request, jsonify
import os
import torch
import pandas as pd
import numpy as np
from io import StringIO

from train.training import StackedGAN
from utils.tokenizer import CharTokenizer
import compare_models

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Paths & Checkpoints
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "stacked_epoch130.pt")
CSV_PATH = os.path.join(BASE_DIR, "comparison_results_gpu.csv")
GENERATED_DATA_PATH = os.path.join(BASE_DIR, "data", "generated_sequences.fasta")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "data", "training.fasta")
SEQ_LEN = 70

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

# -------------------------
# Load GAN & Tokenizer
# -------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cuda"))

device = torch.device("cuda")

# Check if checkpoint has 'gan' key (older checkpoint) or only state_dicts
if "gan" in checkpoint:
    gan: StackedGAN = checkpoint["gan"]
    tokenizer: CharTokenizer = checkpoint["tokenizer"]

    # Move submodules to device
    gan.generator.to(device)
    gan.discriminator.to(device)
    gan.cond_encoder.to(device)
else:
    # Initialize GAN and load weights manually
    tokenizer: CharTokenizer = checkpoint["tokenizer"]
    gan = StackedGAN(seq_len=SEQ_LEN, vocab_size=tokenizer.vocab_size, target_gc=0.42, device=device)
    gan.tokenizer = tokenizer
    gan.generator.load_state_dict(checkpoint["generator_state"])
    gan.discriminator.load_state_dict(checkpoint["discriminator_state"])
    gan.cond_encoder.load_state_dict(checkpoint["cond_encoder_state"])

    # Move submodules to device
    gan.generator.to(device)
    gan.discriminator.to(device)
    gan.cond_encoder.to(device)


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    return render_template("metrics.html")


@app.route("/treatment")
def treatment():
    return render_template("treatment.html")


@app.route("/comparison")
def comparison_page():
    """Render model comparison as HTML table."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        dataset, _ = compare_models.load_dataset(GENERATED_DATA_PATH, SEQ_LEN)
        compare_models.evaluate_models(dataset, tokenizer, SEQ_LEN)
        df = pd.read_csv(CSV_PATH)

    return render_template(
        "comparison.html",
        tables=[df.to_html(classes='data', index=False)],
        titles=df.columns.values
    )


@app.route("/metrics_json")
def metrics_json():
    """Return model metrics as JSON."""
    # Check if CSV exists
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, index_col=0)
        results = df.to_dict(orient="index")
        results = {model: {k: float(v) for k, v in metrics.items()} for model, metrics in results.items()}
        return jsonify(results)

    # If CSV missing, compute metrics from sample data
    dataset, _ = compare_models.load_dataset(SAMPLE_DATA_PATH, SEQ_LEN)
    results = compare_models.evaluate_models(dataset, tokenizer, SEQ_LEN)
    return jsonify(results)


@app.route("/sequence_heatmap_data")
def sequence_heatmap_data():
    """Return sequence heatmap data for visualization."""
    # Generate some sample sequences for each model
    nucleotides = ['A', 'C', 'G', 'T']
    seq_len = SEQ_LEN
    num_sequences = 100
    
    heatmap_data = {}
    
    # Generate sequences for each model type
    models = ['WGAN', 'CTGAN', 'CGAN', 'CramerGAN', 'DraGAN', 'StackedGAN']
    
    for model in models:
        sequences = []
        if model == 'StackedGAN':
            # Use actual StackedGAN to generate sequences
            sequences_int = gan.generate(n_samples=num_sequences)
            sequences = [tokenizer.decode(seq) for seq in sequences_int]
        else:
            # Generate mock sequences for other models with different characteristics
            for _ in range(num_sequences):
                if model == 'WGAN':
                    # Lower diversity sequences
                    seq = ''.join(np.random.choice(nucleotides, seq_len, p=[0.4, 0.1, 0.1, 0.4]))
                elif model == 'CTGAN':
                    # Medium diversity sequences
                    seq = ''.join(np.random.choice(nucleotides, seq_len, p=[0.3, 0.2, 0.2, 0.3]))
                elif model == 'CGAN':
                    # Conditional GAN with balanced distribution
                    seq = ''.join(np.random.choice(nucleotides, seq_len, p=[0.25, 0.25, 0.25, 0.25]))
                elif model == 'CramerGAN':
                    # CramerGAN with slight bias towards GC
                    seq = ''.join(np.random.choice(nucleotides, seq_len, p=[0.2, 0.3, 0.3, 0.2]))
                elif model == 'DraGAN':
                    # DraGAN with gradient penalty - more diverse
                    seq = ''.join(np.random.choice(nucleotides, seq_len, p=[0.22, 0.28, 0.28, 0.22]))
                sequences.append(seq)
        
        # Calculate nucleotide frequency matrix
        freq_matrix = np.zeros((4, seq_len))
        for seq in sequences:
            for pos, nucleotide in enumerate(seq):
                if nucleotide in nucleotides:
                    nuc_idx = nucleotides.index(nucleotide)
                    freq_matrix[nuc_idx, pos] += 1
        
        # Normalize frequencies
        freq_matrix = freq_matrix / num_sequences
        
        # Convert to list format for JSON
        heatmap_data[model] = {
            'frequencies': freq_matrix.tolist(),
            'nucleotides': nucleotides,
            'sequence_length': seq_len
        }
    
    return jsonify(heatmap_data)


@app.route("/analyze_treatment", methods=["POST"])
def analyze_treatment():
    """Analyze DNA sequence and provide disease-specific treatment."""
    data = request.get_json()
    disease = data.get('disease')
    dna_sequence = data.get('dna_sequence', '')
    intensity = data.get('intensity', 'moderate')
    
    if not disease:
        return jsonify({"error": "Disease not specified"}), 400
    
    # Disease-specific treatment database
    disease_db = {
        'sickle_cell': {
            'name': 'Sickle Cell Anemia',
            'mutation_positions': [6, 7, 8, 12, 13, 14],
            'target_mutations': ['A', 'T', 'G', 'C'],
            'improvement_range': [60, 85],
            'treatment_description': 'Gene therapy targeting HBB gene to restore normal hemoglobin production'
        },
        'cystic_fibrosis': {
            'name': 'Cystic Fibrosis',
            'mutation_positions': [9, 10, 11, 15, 16, 17],
            'target_mutations': ['A', 'T', 'G', 'C'],
            'improvement_range': [70, 90],
            'treatment_description': 'CFTR gene correction to restore proper ion channel function'
        },
        'huntington': {
            'name': 'Huntington\'s Disease',
            'mutation_positions': [3, 4, 5, 18, 19, 20],
            'target_mutations': ['A', 'T', 'G', 'C'],
            'improvement_range': [50, 75],
            'treatment_description': 'HTT gene silencing to reduce toxic protein accumulation'
        }
    }
    
    if disease not in disease_db:
        return jsonify({"error": "Disease not supported"}), 400
    
    disease_info = disease_db[disease]
    
    # Generate DNA sequence if not provided
    if not dna_sequence:
        import random
        nucleotides = ['A', 'T', 'G', 'C']
        dna_sequence = ''.join(random.choices(nucleotides, k=30))
    
    # Apply treatment modifications
    modified_sequence, changes = apply_dna_treatment(
        dna_sequence, 
        disease_info, 
        intensity
    )
    
    # Calculate effectiveness
    effectiveness = calculate_treatment_effectiveness(disease_info, intensity, len(changes))
    
    return jsonify({
        'original_sequence': dna_sequence,
        'modified_sequence': modified_sequence,
        'changes': changes,
        'effectiveness': effectiveness,
        'disease_info': disease_info,
        'treatment_summary': generate_treatment_summary(disease_info, effectiveness, changes)
    })


def apply_dna_treatment(sequence, disease_info, intensity):
    """Apply disease-specific DNA modifications."""
    import random
    
    sequence_list = list(sequence)
    changes = []
    
    # Determine number of modifications based on intensity
    mutation_count = {'conservative': 2, 'moderate': 4, 'aggressive': 6}[intensity]
    
    positions = disease_info['mutation_positions'][:mutation_count]
    mutations = disease_info['target_mutations']
    
    for pos in positions:
        if pos < len(sequence_list):
            original = sequence_list[pos]
            new_nucleotide = random.choice(mutations)
            sequence_list[pos] = new_nucleotide
            changes.append({
                'position': pos,
                'from': original,
                'to': new_nucleotide
            })
    
    return ''.join(sequence_list), changes


def calculate_treatment_effectiveness(disease_info, intensity, num_changes):
    """Calculate treatment effectiveness percentage."""
    import random
    
    base_range = disease_info['improvement_range']
    effectiveness = base_range[0] + random.random() * (base_range[1] - base_range[0])
    
    # Adjust based on intensity
    intensity_multiplier = {'conservative': 0.8, 'moderate': 1.0, 'aggressive': 1.1}[intensity]
    effectiveness *= intensity_multiplier
    
    # Adjust based on number of changes
    effectiveness += num_changes * 2
    
    return min(round(effectiveness), 95)


def generate_treatment_summary(disease_info, effectiveness, changes):
    """Generate a summary of the treatment."""
    return {
        'disease_name': disease_info['name'],
        'treatment_approach': disease_info['treatment_description'],
        'mutations_applied': len(changes),
        'expected_improvement': f"{effectiveness}%",
        'risk_reduction': f"{round(effectiveness * 0.8)}%",
        'side_effects_risk': f"{round((100 - effectiveness) * 0.3)}%",
        'success_probability': f"{round(effectiveness * 0.9)}%"
    }


@app.route("/generate", methods=["POST"])
def generate():
    """Generate sequences with the pre-trained StackedGAN."""
    num_sequences = int(request.form.get("num_sequences", 10))
    sequences_int = gan.generate(n_samples=num_sequences)
    sequences_str = [tokenizer.decode(seq) for seq in sequences_int]

    # Compute GC content
    gc_count = sum(seq.count("G") + seq.count("C") for seq in sequences_str)
    total_bases = sum(len(seq) for seq in sequences_str)
    gc_content_val = (gc_count / total_bases * 100) if total_bases else 0

    # Prepare FASTA
    fasta_io = StringIO()
    for idx, seq in enumerate(sequences_str, start=1):
        fasta_io.write(f">Seq{idx}\n{seq}\n")

    return jsonify({
        "sequences": sequences_str,
        "gc_content": gc_content_val,
        "fasta": fasta_io.getvalue()
    })


# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
