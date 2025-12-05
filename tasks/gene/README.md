# CTM Gene: Applying Continuous Thought Machines to Gene Expression

This module adapts the Continuous Thought Machine (CTM) architecture for gene expression analysis, specifically for **Gene Regulatory Network (GRN) inference** and **Single-Cell Trajectory Inference**.

## Conceptual Mapping

| CTM Component | Biological Equivalent | Application |
|---------------|----------------------|-------------|
| Neuron | Gene | Each "neuron" represents a specific gene |
| Synapse | Regulatory Interaction | Learns how genes influence each other |
| Internal Tick | Pseudotime / Cell State | Progression through differentiation |
| Synchronization | Co-expression / Regulation | Captures which genes are co-regulated |

## Quick Start

### 1. Setup Environment

```bash
cd continuous-thought-machines
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train on Real Data (Human Embryonic Meninges)

```bash
PYTHONPATH=. python tasks/gene/train.py --use_real_data --training_iterations 1000
```

This will:
- Load the preprocessed meninges scRNA-seq dataset (200 genes, 50 pseudotime steps)
- Train CTM to predict gene expression trajectories
- Save visualizations to `logs/gene/`
- Save a model checkpoint to `logs/gene/model_checkpoint.pt`

### 3. Train on Synthetic Data

```bash
PYTHONPATH=. python tasks/gene/train.py --n_genes 100 --iterations 30 --training_iterations 500
```

## Project Structure

```
tasks/gene/
├── train.py                    # Main training script
├── benchmark_beeline.py        # BEELINE GRN benchmark
├── benchmark_causalbench.py    # CausalBench (Perturb-seq) benchmark
├── benchmark_perturbseq.py     # Knockout prediction benchmark
├── benchmark_trajectory.py     # GSEA trajectory analysis
├── benchmark_perteval.py       # PertEval-scFM integration
├── benchmark_utils.py          # Shared metrics and utilities
├── run_benchmarks.py           # Unified benchmark runner
└── README.md                   # This file

data/
├── gene_data.py                # Synthetic + Meninges datasets
├── hematopoiesis_data.py       # Hematopoiesis (CELLxGENE)
├── perturbseq_data.py          # Norman/Replogle Perturb-seq
├── beeline_data.py             # BEELINE benchmark data
├── processed_meninges.pt       # Preprocessed meninges data
└── human_embryonic_meninges_5-13_wks.h5ad  # Raw scRNA-seq

models/
└── ctm_gene.py                 # CTM adapted for gene expression
```

## Training Options

```bash
PYTHONPATH=. python tasks/gene/train.py [OPTIONS]

Options:
  --use_real_data           Use real scRNA-seq data (meninges)
  --n_genes N               Number of genes (default: 50, overridden by real data)
  --iterations N            Pseudotime steps (default: 20, overridden by real data)
  --training_iterations N   Training steps (default: 2000)
  --batch_size N            Batch size (default: 32)
  --lr FLOAT                Learning rate (default: 1e-3)
  --log_dir PATH            Output directory (default: logs/gene)
  --track_every N           Visualization frequency (default: 200)
```

## Benchmarks

### Run All Benchmarks

```bash
PYTHONPATH=. python tasks/gene/run_benchmarks.py --benchmark all
```

### Individual Benchmarks

#### 1. BEELINE (GRN Accuracy)

Evaluates inferred GRN against ChIP-seq validated TF-target edges.

```bash
# Human ESC
PYTHONPATH=. python tasks/gene/benchmark_beeline.py \
    --dataset hESC \
    --training_iterations 400 \
    --output_dir logs/benchmarks/beeline_hESC

# Mouse ESC
PYTHONPATH=. python tasks/gene/benchmark_beeline.py \
    --dataset mESC \
    --training_iterations 400 \
    --output_dir logs/benchmarks/beeline_mESC
```

**Datasets:** `hESC`, `hHep`, `mDC`, `mESC`, `mHSC-E`, `mHSC-GM`, `mHSC-L`

**Metrics:**
- AUPRC (Area Under Precision-Recall Curve) - primary metric
- AUROC (Area Under ROC Curve)
- Early Precision @100, @500

#### 2. CausalBench (Causal GRN with Perturb-seq)

Evaluates causal GRN inference using real interventional (Perturb-seq) data.

```bash
# First, download data (10GB, takes time)
causalbench_run \
    --dataset_name weissmann_k562 \
    --output_directory logs/benchmarks/causalbench_download \
    --data_directory data/causalbench \
    --training_regime observational \
    --model_name random100 \
    --subset_data 0.1

# Then run CTM benchmark
PYTHONPATH=. python tasks/gene/benchmark_causalbench.py \
    --dataset weissmann_k562 \
    --training_iterations 400 \
    --output_dir logs/benchmarks/causalbench_k562
```

**Datasets:** `weissmann_k562` (K562 cells), `weissmann_rpe1` (RPE1 cells)

#### 3. Trajectory GSEA Analysis

Analyzes what gene sets are enriched at each CTM "thinking" tick.

```bash
PYTHONPATH=. python tasks/gene/benchmark_trajectory.py \
    --dataset meninges \
    --n_ticks 5 \
    --gene_set_library GO_Biological_Process_2023 \
    --output_dir logs/benchmarks/trajectory
```

**Output:**
- `activation_heatmap.png` - Gene activations across ticks
- `variable_genes_trajectory.png` - Top 20 most dynamic genes
- `gsea_tick_*.csv` - GSEA results per tick

#### 4. Perturb-seq Knockout Prediction

Tests CTM's ability to predict expression changes after gene knockouts.

```bash
# Requires trained checkpoint
PYTHONPATH=. python tasks/gene/benchmark_perturbseq.py \
    --checkpoint logs/gene/model_checkpoint.pt \
    --dataset norman \
    --output_dir logs/benchmarks/perturbseq
```

**Metrics:**
- Pearson correlation (predicted vs actual LFC)
- Direction accuracy (sign of change)
- Top-K recall

#### 5. PertEval-scFM

Standardized evaluation framework for perturbation prediction.

```bash
PYTHONPATH=. python tasks/gene/benchmark_perteval.py \
    --checkpoint logs/gene/model_checkpoint.pt \
    --perturb_data data/norman_processed.pt \
    --output_dir logs/benchmarks/perteval
```

## Preprocessing New Data

### From Raw h5ad File

```python
import scanpy as sc
import torch

# Load and preprocess
adata = sc.read_h5ad('your_data.h5ad')
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=200, subset=True)

# Compute pseudotime
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.diffmap(adata)
adata.uns['iroot'] = 0  # Set root cell
sc.tl.dpt(adata)

# Create trajectories (see process_meninges.py for full example)
```

### Using the Preprocessing Script

```bash
# Edit process_meninges.py to point to your data, then:
PYTHONPATH=. python process_meninges.py
```

## Model Architecture

The `ContinuousThoughtMachineGENE` class adapts the base CTM:

```python
from models.ctm_gene import ContinuousThoughtMachineGENE

model = ContinuousThoughtMachineGENE(
    iterations=50,           # Pseudotime steps
    d_model=200,             # Number of genes
    d_input=200,             # Input dimension (same as genes)
    heads=0,                 # No attention (simplified for gene data)
    n_synch_out=200,         # Full synchronization matrix
    n_synch_action=0,        # No action synchronization
    synapse_depth=2,         # Synapse network depth
    memory_length=10,        # NLM memory length
    deep_nlms=True,          # Use deep neuron-level models
    memory_hidden_dims=4,    # NLM hidden dimensions
    neuron_select_type='first-last',  # Full pairwise synchronization
    out_dims=200,            # Output = predicted expression
)
```

### Key Methods

```python
# Forward pass
predictions, certainties, synchronization = model(x0)
# predictions: (B, N_genes, T) - predicted expression trajectory
# synchronization: (B, N_genes*(N_genes+1)/2) - upper triangle of GRN

# Forward with tracking (for analysis)
predictions, certainties, synch_tracking, pre_act, post_act = model(x0, track=True)

# In-silico knockout
predictions = model.forward_with_knockout(x0, knockout_indices=[gene_idx])
```

## Output Visualizations

Training produces three types of visualizations every `--track_every` steps:

1. **`traj_*.png`** - Top 20 variable genes: predicted vs true trajectories
2. **`synch_*.png`** - Synchronization matrix heatmap (log-scale)
3. **`graph_*.png`** - Top 100 regulatory interactions as network graph

## Expected Results

### BEELINE Benchmark (400 iterations, ~100 genes)

| Dataset | AUPRC | Random Baseline | Improvement |
|---------|-------|-----------------|-------------|
| hESC | ~0.08 | 0.08 | ~3% |
| mESC | ~0.09 | 0.08 | ~12% |

Note: With limited genes and training, expect modest improvements over random. Scale up genes/iterations for better results.

### Interpretation

- **AUPRC > random baseline** = CTM is learning real regulatory structure
- **Early Precision @100** = Accuracy of top 100 predicted edges
- Higher values indicate the synchronization matrix captures true TF-target relationships

## Troubleshooting

### Import Errors

```bash
# Always run from project root with PYTHONPATH
cd continuous-thought-machines
PYTHONPATH=. python tasks/gene/train.py
```

### Dependency Conflicts

```bash
# If zarr version issues:
pip install 'zarr<3'

# If numpy/pandas issues:
pip install --force-reinstall pandas numpy
```

### Memory Issues

```bash
# Reduce batch size or gene count
python tasks/gene/train.py --batch_size 16 --n_genes 100
```

### CausalBench Data Download

The K562 dataset is ~10GB. Download separately:
```bash
causalbench_run --dataset_name weissmann_k562 \
    --data_directory data/causalbench \
    --training_regime observational \
    --model_name random100 \
    --subset_data 0.1
```

## References

- **CTM Paper:** [Continuous Thought Machines](https://arxiv.org/abs/...)
- **BEELINE:** [Benchmarking algorithms for GRN inference](https://www.nature.com/articles/s41592-019-0690-6)
- **CausalBench:** [Scalable causal discovery benchmark](https://arxiv.org/abs/2210.17283)
- **Meninges Data:** Human embryonic meninges scRNA-seq, 5-13 weeks post-conception

## Citation

If you use this code, please cite:
```
@article{ctm2024,
  title={Continuous Thought Machines},
  author={...},
  year={2024}
}
```

