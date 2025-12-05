import anndata
import scanpy as sc
import numpy as np
import torch
from scipy.sparse import issparse

# 1. Load Data
print("Loading data...")
adata = anndata.read_h5ad('data/human_embryonic_meninges_5-13_wks.h5ad')

# 2. Filter for Meningeal Lineage
target_classes = ['Progenitor', 'Primary meninx', 'Pia', 'Arachnoid', 'Dura', 'Primitive']
print(f"Filtering for classes: {target_classes}")
adata = adata[adata.obs['ClassAnn'].isin(target_classes)].copy()
print(f"Cells after filtering: {adata.shape[0]}")

# 3. Preprocessing
print("Preprocessing...")
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=200, subset=True) # Keep top 200 genes for CTM
print(f"Shape after HVG selection: {adata.shape}")

# 4. Dimensionality Reduction & Pseudotime
print("Running PCA and Diffusion Map...")
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.diffmap(adata)

# 5. Pseudotime Inference (using dpt)
# We need a root cell. We'll pick a 'Progenitor' or 'Primitive' cell with low Diffusion Component 1 value
# Or just use DC1 as a proxy for pseudotime if it correlates well.
adata.uns['iroot'] = np.flatnonzero(adata.obs['ClassAnn'] == 'Primitive')[0]
sc.tl.dpt(adata)
pseudotime = adata.obs['dpt_pseudotime'].values

# 6. Create Trajectories
# We will bin cells by pseudotime to create smooth trajectories
n_bins = 50 # Number of time steps for the CTM
bins = np.linspace(0, 1, n_bins+1)
bin_indices = np.digitize(pseudotime, bins) - 1

# Calculate mean expression per bin
# X might be sparse
if issparse(adata.X):
    X = adata.X.toarray()
else:
    X = adata.X

print("Creating smooth trajectories...")
# Instead of just one trajectory, we want multiple samples.
# We can sample random walks or just create variation around the mean trajectory.
# For simplicity, let's create a "mean trajectory" and then generate samples by adding noise 
# OR sample actual cells from each bin.

# Strategy: Sample 1 cell from each bin to create a "noisy" trajectory.
# If a bin is empty, use the nearest valid bin.

trajectories = []
initial_states = []
n_samples = 1000

valid_bins = [i for i in range(n_bins) if np.sum(bin_indices == i) > 0]

for _ in range(n_samples):
    traj = []
    # For the initial state, pick a cell from the first valid bin (early time)
    first_bin_cells = np.where(bin_indices == valid_bins[0])[0]
    start_idx = np.random.choice(first_bin_cells)
    initial_states.append(X[start_idx])
    
    for t in range(n_bins):
        # Find bin
        if t in valid_bins:
            bin_cells = np.where(bin_indices == t)[0]
            # Sample a cell
            cell_idx = np.random.choice(bin_cells)
            traj.append(X[cell_idx])
        else:
            # If bin is empty, repeat last state (or interpolate)
            if len(traj) > 0:
                traj.append(traj[-1])
            else:
                traj.append(X[start_idx])
                
    trajectories.append(np.stack(traj))

# Convert to tensors
X0 = torch.tensor(np.stack(initial_states), dtype=torch.float32)
Traj = torch.tensor(np.stack(trajectories), dtype=torch.float32) # (N_samples, T, N_genes)
Traj = Traj.permute(0, 2, 1) # (B, N_genes, T) to match model expectation

print(f"Final Data Shapes: X0={X0.shape}, Traj={Traj.shape}")

# Save
gene_names = adata.var['feature_name'].tolist() if 'feature_name' in adata.var else adata.var_names.tolist()
save_path = 'data/processed_meninges.pt'
torch.save({'x0': X0, 'trajectory': Traj, 'gene_names': gene_names}, save_path)
print(f"Saved to {save_path}")

