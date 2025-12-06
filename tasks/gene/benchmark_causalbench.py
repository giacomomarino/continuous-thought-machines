"""
CausalBench Benchmark for CTM Gene Model.

Evaluates CTM's GRN inference using real Perturb-seq interventional data.
Uses K562 and RPE1 datasets with biologically meaningful metrics.

APPROACH: Perturbation Prediction
- Train CTM to predict: control_expression + knockout → perturbed_expression
- The synchronization matrix captures causal regulatory relationships
- Internal ticks represent perturbation signal propagating through the GRN

Paper: https://arxiv.org/abs/2210.17283
GitHub: https://github.com/causalbench/causalbench
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Tuple
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.ctm_gene import ContinuousThoughtMachineGENE
from models.ctm_gene_attention import ContinuousThoughtMachineGENEAttention
from tasks.gene.benchmark_utils import reconstruct_synch_matrix

# CausalBench imports
try:
    from causalscbench.models.abstract_model import AbstractInferenceModel
    from causalscbench.models.training_regimes import TrainingRegime
    from causalscbench.evaluation.statistical_evaluation import Evaluator
    from causalscbench.data_access.create_dataset import CreateDataset
    from causalscbench.data_access.utils.splitting import DatasetSplitter
    CAUSALBENCH_AVAILABLE = True
except ImportError as e:
    print(f"CausalBench import error: {e}")
    CAUSALBENCH_AVAILABLE = False


CAUSALBENCH_DATASETS = ['weissmann_k562', 'weissmann_rpe1']


def parse_args():
    parser = argparse.ArgumentParser(description='CausalBench Benchmark for CTM')
    parser.add_argument('--dataset', type=str, default='weissmann_k562',
                        choices=CAUSALBENCH_DATASETS,
                        help='CausalBench dataset to use')
    parser.add_argument('--output_dir', type=str, default='logs/benchmark_causalbench',
                        help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='data/causalbench',
                        help='Directory for CausalBench data cache')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--training_iterations', type=int, default=2000,
                        help='Number of training iterations for CTM')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_ticks', type=int, default=30,
                        help='Number of CTM ticks (perturbation propagation steps)')
    parser.add_argument('--top_k_edges', type=int, default=1000,
                        help='Number of top edges to return from CTM')
    parser.add_argument('--max_path_length', type=int, default=3,
                        help='Max path length for evaluation (-1 for unlimited)')
    parser.add_argument('--subset_data', type=float, default=1.0,
                        help='Fraction of data to use')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use attention-based synapse (directional GRN)')
    parser.add_argument('--use_combined', action='store_true',
                        help='Use COMBINED: attention direction × sync magnitude')
    parser.add_argument('--n_attention_heads', type=int, default=4,
                        help='Number of attention heads (when using attention)')
    parser.add_argument('--d_attention_head', type=int, default=16,
                        help='Dimension per attention head')
    return parser.parse_args()


class PerturbationDataset(Dataset):
    """
    Dataset for perturbation prediction.
    
    Each sample is:
    - Input: control expression with knocked-out gene zeroed
    - Target: actual expression after knockout
    """
    
    def __init__(self, expression_matrix, interventions, gene_names, min_cells=10):
        """
        Args:
            expression_matrix: (n_cells, n_genes) expression data
            interventions: list of which gene was knocked out per cell
            gene_names: list of gene names
            min_cells: minimum cells per perturbation to include
        """
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.n_genes = len(gene_names)
        
        # Get control (non-targeting) expression
        control_mask = np.array([i == "non-targeting" for i in interventions])
        if control_mask.sum() == 0:
            raise ValueError("No control (non-targeting) cells found")
        
        self.control_expr = expression_matrix[control_mask].mean(axis=0)
        self.control_expr = torch.tensor(self.control_expr, dtype=torch.float32)
        
        # Build perturbation data
        self.perturbations = []
        unique_interventions = set(interventions)
        
        for gene in unique_interventions:
            if gene == "non-targeting":
                continue
            
            # Check if this gene is in our gene list
            if gene not in self.gene_to_idx:
                continue
            
            gene_idx = self.gene_to_idx[gene]
            mask = np.array([i == gene for i in interventions])
            
            if mask.sum() < min_cells:
                continue
            
            # Mean expression after this knockout
            perturbed_expr = expression_matrix[mask].mean(axis=0)
            
            self.perturbations.append({
                'knockout_idx': gene_idx,
                'knockout_name': gene,
                'perturbed_expr': torch.tensor(perturbed_expr, dtype=torch.float32),
                'n_cells': mask.sum(),
            })
        
        print(f"Created dataset with {len(self.perturbations)} perturbations")
        print(f"Control expression from {control_mask.sum()} cells")
    
    def __len__(self):
        return len(self.perturbations)
    
    def __getitem__(self, idx):
        item = self.perturbations[idx]
        
        # Create input: control expression with KO gene zeroed
        x_input = self.control_expr.clone()
        x_input[item['knockout_idx']] = 0.0
        
        return {
            'input': x_input,
            'target': item['perturbed_expr'],
            'knockout_idx': item['knockout_idx'],
            'knockout_name': item['knockout_name'],
        }


class CTMPerturbationModel(nn.Module):
    """
    CTM wrapper for perturbation prediction.
    
    Takes control expression + knockout mask → predicts post-knockout expression
    
    Three variants:
    1. Synchronization-based (default): Uses U-Net synapse, extracts GRN from sync matrix
    2. Attention-based: Uses attention synapse, extracts DIRECTIONAL GRN from attention weights
    3. Combined: Uses attention synapse, combines attention direction × sync magnitude
    """
    
    def __init__(self, n_genes, n_ticks=30, device='cpu', 
                 use_attention=False, use_combined=False,
                 n_attention_heads=4, d_attention_head=16):
        super().__init__()
        self.n_genes = n_genes
        self.n_ticks = n_ticks
        self.device = device
        self.use_attention = use_attention or use_combined  # Combined requires attention
        self.use_combined = use_combined
        
        if self.use_attention:
            mode = "COMBINED (attention direction × sync magnitude)" if use_combined else "ATTENTION-based (directional GRN)"
            print(f"Using {mode}")
            print(f"  Heads: {n_attention_heads}, Head dim: {d_attention_head}")
            # Build Attention-based CTM
            self.ctm = ContinuousThoughtMachineGENEAttention(
                iterations=n_ticks,
                d_model=n_genes,
                d_input=n_genes,
                n_synch_out=n_genes,
                synapse_depth=2,
                memory_length=10,
                deep_nlms=True,
                memory_hidden_dims=4,
                do_layernorm_nlm=False,
                out_dims=n_genes,
                neuron_select_type='first-last',
                dropout=0.0,
                n_attention_heads=n_attention_heads,
                d_attention_head=d_attention_head,
            ).to(device)
        else:
            print("Using standard synapse (symmetric GRN from sync matrix)")
            # Build standard CTM
            self.ctm = ContinuousThoughtMachineGENE(
                iterations=n_ticks,
                d_model=n_genes,
                d_input=n_genes,
                heads=0,
                n_synch_out=n_genes,
                n_synch_action=0,
                synapse_depth=2,
                memory_length=10,
                deep_nlms=True,
                memory_hidden_dims=4,
                do_layernorm_nlm=False,
                backbone_type='none',
                positional_embedding_type='none',
                out_dims=n_genes,
                neuron_select_type='first-last',
                dropout=0.0
            ).to(device)
        
        # Initialize lazy modules
        dummy = torch.randn(1, n_genes, device=device)
        self.ctm(dummy)
    
    def forward(self, x, track=False):
        """
        Args:
            x: (B, N_genes) input expression with knockout gene zeroed
            track: whether to return internal states
            
        Returns:
            final_pred: (B, N_genes) predicted expression after perturbation
            tracking_data: dict with 'synch' and optionally 'attention'
        """
        if track:
            if self.use_attention:
                # Attention model returns both sync and attention tracking
                predictions, certainties, synch_tracking, pre_act, post_act, attn_tracking = self.ctm(x, track=True)
                final_pred = predictions[:, :, -1]
                return final_pred, {
                    'synch': synch_tracking,
                    'attention': attn_tracking
                }
            else:
                predictions, certainties, synch_tracking, pre_act, post_act = self.ctm(x, track=True)
                final_pred = predictions[:, :, -1]
                return final_pred, {'synch': synch_tracking[-1]}
        else:
            predictions, certainties, synch = self.ctm(x, track=False)
            final_pred = predictions[:, :, -1]
            return final_pred, synch
    
    def get_attention_grn(self, aggregate='mean'):
        """Get GRN from attention weights (only for attention model)."""
        if not self.use_attention:
            raise ValueError("Attention GRN only available with use_attention=True")
        return self.ctm.get_attention_grn(aggregate=aggregate)


def train_perturbation_model(model, dataset, device='cpu', training_iterations=2000, 
                              lr=1e-3, batch_size=32):
    """
    Train CTM to predict perturbation effects.
    """
    from utils.schedulers import WarmupCosineAnnealingLR
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupCosineAnnealingLR(
        optimizer, 100, training_iterations,
        warmup_start_lr=1e-20, eta_min=1e-7
    )
    criterion = nn.MSELoss()
    
    iterator = iter(dataloader)
    losses = []
    
    pbar = tqdm(range(training_iterations), desc='Training CTM (Perturbation)')
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        
        x_input = batch['input'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred, _ = model(x_input)
        
        # Loss: predict expression after knockout
        loss = criterion(pred, target)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        pbar.set_description(f'Loss: {loss.item():.4f}')
        wandb.log({"loss": loss.item(), "step": step})
    
    return losses


def extract_edges_from_model(model, dataset, device='cpu', top_k=1000):
    """
    Extract top-k edges from trained CTM.
    
    For sync-based model: extracts from synchronization matrix (symmetric)
    For attention-based model: extracts from attention weights (DIRECTIONAL)
    For combined model: attention_direction × sync_magnitude (DIRECTIONAL + strong)
    """
    model.eval()
    gene_names = dataset.gene_names
    n_genes = len(gene_names)
    
    with torch.no_grad():
        x = dataset.control_expr.unsqueeze(0).to(device)
        _, result = model(x, track=True)
        
        if model.use_combined:
            # COMBINED: attention direction × sync magnitude
            print("Combining attention (direction) × sync (magnitude)...")
            
            attn_history = result['attention']  # (T, B, n_heads, n_genes, n_genes)
            synch_history = result['synch']     # (T, B, synch_dim)
            
            # Get attention matrix (direction): average over time, batch, heads
            attn_matrix = attn_history.mean(axis=(0, 1, 2))  # (n_genes, n_genes)
            
            # Get sync matrix (magnitude): use final timestep
            synch_vec = synch_history[-1, 0]  # Final time, first batch
            synch_matrix = reconstruct_synch_matrix(synch_vec, n_genes)
            
            # COMBINE: attention provides direction, sync provides magnitude
            # Attention is already normalized (softmax), sync has raw magnitude
            # Strategy: Use attention to "mask" sync matrix directionally
            
            # Make sync matrix absolute (we'll use attention for sign/direction)
            sync_magnitude = np.abs(synch_matrix)
            
            # Combined GRN: attention_weight * sync_magnitude
            # This is DIRECTIONAL (from attention) with MAGNITUDE (from sync)
            combined_grn = attn_matrix * sync_magnitude
            
            # Zero diagonal
            np.fill_diagonal(combined_grn, 0)
            
            # Get indices of all non-diagonal elements (directional)
            rows, cols = np.where(np.ones((n_genes, n_genes)) - np.eye(n_genes))
            weights = combined_grn[rows, cols]
            
            sorted_idx = np.argsort(weights)[::-1]
            top_k = min(top_k, len(sorted_idx))
            
            edges = []
            seen_edges = set()
            for i in range(len(sorted_idx)):
                if len(edges) >= top_k:
                    break
                idx = sorted_idx[i]
                gene_a_idx = int(rows[idx])
                gene_b_idx = int(cols[idx])
                gene_a = gene_names[gene_a_idx]
                gene_b = gene_names[gene_b_idx]
                edge_key = (gene_a, gene_b)
                if edge_key not in seen_edges and gene_a != gene_b:
                    edges.append((gene_a, gene_b))
                    seen_edges.add(edge_key)
            
            print(f"Extracted {len(edges)} COMBINED edges (attention×sync)")
            print(f"  Attention provides DIRECTION, Sync provides MAGNITUDE")
            return edges, combined_grn
        
        elif model.use_attention:
            # Attention-only: result is dict with 'attention' key
            attn_history = result['attention']  # (T, B, n_heads, n_genes, n_genes)
            
            # Average over time, batch, and heads to get (n_genes, n_genes) matrix
            grn_matrix = attn_history.mean(axis=(0, 1, 2))
            
            # Attention is DIRECTIONAL: A[i,j] = "gene i regulates gene j"
            np.fill_diagonal(grn_matrix, 0)
            
            rows, cols = np.where(np.ones((n_genes, n_genes)) - np.eye(n_genes))
            weights = grn_matrix[rows, cols]
            
            sorted_idx = np.argsort(weights)[::-1]
            top_k = min(top_k, len(sorted_idx))
            
            edges = []
            seen_edges = set()
            for i in range(len(sorted_idx)):
                if len(edges) >= top_k:
                    break
                idx = sorted_idx[i]
                gene_a_idx = int(rows[idx])
                gene_b_idx = int(cols[idx])
                gene_a = gene_names[gene_a_idx]
                gene_b = gene_names[gene_b_idx]
                edge_key = (gene_a, gene_b)
                if edge_key not in seen_edges and gene_a != gene_b:
                    edges.append((gene_a, gene_b))
                    seen_edges.add(edge_key)
            
            print(f"Extracted {len(edges)} DIRECTIONAL edges from attention")
            return edges, grn_matrix
        
        else:
            # Sync-based: result is dict with 'synch' key
            synch_vec = result['synch']
            if isinstance(synch_vec, np.ndarray):
                if synch_vec.ndim > 1:
                    synch_vec = synch_vec[0]  # First batch element
            else:
                synch_vec = synch_vec.cpu().numpy()
            
            synch_matrix = reconstruct_synch_matrix(synch_vec, n_genes)
            
            # Sync matrix is symmetric - only need upper triangle
            np.fill_diagonal(synch_matrix, 0)
            triu_idx = np.triu_indices(n_genes, k=1)
            weights = synch_matrix[triu_idx]
            abs_weights = np.abs(weights)
            
            sorted_idx = np.argsort(abs_weights)[::-1]
            top_k = min(top_k, len(sorted_idx))
            
            edges = []
            for i in range(top_k):
                idx = sorted_idx[i]
                gene_a = gene_names[triu_idx[0][idx]]
                gene_b = gene_names[triu_idx[1][idx]]
                edges.append((gene_a, gene_b))
            
            return edges, synch_matrix


def run_causalbench_evaluation(args):
    """Run full CausalBench evaluation with perturbation prediction."""
    import matplotlib.pyplot as plt
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    if not CAUSALBENCH_AVAILABLE:
        print("CausalBench not available. Please install: pip install causalbench")
        return None
    
    # Load data
    print(f"Loading CausalBench dataset: {args.dataset}")
    print("This may download ~10GB of data on first run...")
    
    try:
        dataset_creator = CreateDataset(args.data_dir, filter=True)
        path_k562, path_rpe1 = dataset_creator.load()
        
        if args.dataset == 'weissmann_k562':
            data_path = path_k562
        else:
            data_path = path_rpe1
        
        # Load data directly (we'll do our own split)
        loaded = np.load(data_path, allow_pickle=True)
        expression_matrix = loaded['expression_matrix']
        gene_names = list(loaded['var_names'])
        interventions = list(loaded['interventions'])
        
        # Subset if requested
        if args.subset_data < 1.0:
            n = int(len(interventions) * args.subset_data)
            idx = np.random.choice(len(interventions), n, replace=False)
            expression_matrix = expression_matrix[idx]
            interventions = [interventions[i] for i in idx]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"Loaded: {expression_matrix.shape[0]} cells, {expression_matrix.shape[1]} genes")
    
    # Split into train and test (80/20)
    from sklearn.model_selection import train_test_split
    
    train_idx, test_idx = train_test_split(
        range(len(interventions)), 
        test_size=0.2, 
        random_state=42,
        stratify=interventions
    )
    
    train_expr = expression_matrix[train_idx]
    train_interventions = [interventions[i] for i in train_idx]
    test_expr = expression_matrix[test_idx]
    test_interventions = [interventions[i] for i in test_idx]
    
    print(f"Train: {len(train_idx)} cells, Test: {len(test_idx)} cells")
    
    # Create perturbation dataset
    print("\nCreating perturbation dataset...")
    train_dataset = PerturbationDataset(
        train_expr, train_interventions, gene_names, min_cells=5
    )
    
    if len(train_dataset) == 0:
        print("Error: No valid perturbations found in training data")
        return None
    
    # Initialize wandb
    if args.use_combined:
        synapse_type = "combined"
    elif args.use_attention:
        synapse_type = "attention"
    else:
        synapse_type = "synch"
    wandb.init(
        project="ctm-gene-causalbench",
        name=f"{args.dataset}_{synapse_type}",
        config=vars(args)
    )
    
    # Create and train model
    print(f"\nBuilding CTM: {len(gene_names)} genes, {args.n_ticks} ticks")
    model = CTMPerturbationModel(
        n_genes=len(gene_names),
        n_ticks=args.n_ticks,
        device=args.device,
        use_attention=args.use_attention,
        use_combined=args.use_combined,
        n_attention_heads=args.n_attention_heads,
        d_attention_head=args.d_attention_head,
    ).to(args.device)
    
    print(f"\nTraining for {args.training_iterations} iterations...")
    losses = train_perturbation_model(
        model, train_dataset, 
        device=args.device,
        training_iterations=args.training_iterations,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss (Perturbation Prediction)')
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'), dpi=150)
    plt.close()
    
    # Extract edges
    if args.use_combined:
        print("\nExtracting edges using COMBINED (attention×sync)...")
    elif args.use_attention:
        print("\nExtracting edges from attention weights (DIRECTIONAL)...")
    else:
        print("\nExtracting edges from synchronization matrix...")
    edges, grn_matrix = extract_edges_from_model(
        model, train_dataset, args.device, args.top_k_edges
    )
    print(f"Extracted {len(edges)} edges")
    
    # Plot GRN matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(grn_matrix), cmap='viridis', aspect='auto')
    if args.use_combined:
        plt.colorbar(label='|Attention × Sync|')
        plt.title('Inferred GRN (COMBINED: Direction × Magnitude)')
    elif args.use_attention:
        plt.colorbar(label='|Attention Weight|')
        plt.title('Inferred GRN (Attention Weights - DIRECTIONAL)')
    else:
        plt.colorbar(label='|Synchronization|')
        plt.title('Inferred GRN (Synchronization Matrix)')
    plt.savefig(os.path.join(args.output_dir, 'grn_matrix.png'), dpi=150)
    plt.close()
    
    # Evaluate with CausalBench
    print("\nEvaluating with CausalBench...")
    evaluator = Evaluator(
        expression_matrix=test_expr,
        interventions=test_interventions,
        gene_names=gene_names,
    )
    
    results = evaluator.evaluate_network(
        network=edges,
        max_path_length=args.max_path_length,
        check_false_omission_rate=True,
        omission_estimation_size=500
    )
    
    # Print results
    print("\n" + "="*60)
    print("CAUSALBENCH RESULTS (Perturbation Prediction)")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Training iterations: {args.training_iterations}")
    print(f"Number of edges: {len(edges)}")
    print("-"*60)
    print("Output Graph:")
    print(f"  True Positives: {results['output_graph']['true_positives']}")
    print(f"  False Positives: {results['output_graph']['false_positives']}")
    tp = results['output_graph']['true_positives']
    fp = results['output_graph']['false_positives']
    print(f"  Precision: {tp/(tp+fp)*100:.1f}%")
    print(f"  Wasserstein Distance (mean): {results['output_graph']['wasserstein_distance']['mean']:.4f}")
    print("-"*60)
    print(f"False Omission Rate: {results['false_omission_rate']:.4f}")
    print(f"Negative Mean Wasserstein: {results['negative_mean_wasserstein']:.4f}")
    print("="*60)
    
    # Compute additional metrics
    # If our predicted edges have HIGHER wasserstein than random negatives, we're doing well
    our_wasserstein = results['output_graph']['wasserstein_distance']['mean']
    neg_wasserstein = results['negative_mean_wasserstein']
    wasserstein_ratio = our_wasserstein / neg_wasserstein if neg_wasserstein > 0 else 0
    
    print(f"\nWasserstein Ratio (ours/random): {wasserstein_ratio:.2f}")
    print("(>1.0 means our edges have larger effects than random = GOOD)")
    
    # Save results
    results_to_save = {
        'dataset': args.dataset,
        'training_iterations': args.training_iterations,
        'n_ticks': args.n_ticks,
        'n_edges': len(edges),
        'true_positives': results['output_graph']['true_positives'],
        'false_positives': results['output_graph']['false_positives'],
        'precision': tp/(tp+fp),
        'wasserstein_mean': results['output_graph']['wasserstein_distance']['mean'],
        'false_omission_rate': results['false_omission_rate'],
        'negative_mean_wasserstein': results['negative_mean_wasserstein'],
        'wasserstein_ratio': wasserstein_ratio,
        'final_loss': losses[-1] if losses else None,
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save edges
    with open(os.path.join(args.output_dir, 'edges.txt'), 'w') as f:
        for a, b in edges[:100]:
            f.write(f"{a}\t{b}\n")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'gene_names': gene_names,
        'args': vars(args),
    }, os.path.join(args.output_dir, 'model.pt'))
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Log final metrics to wandb
    wandb.log({
        "true_positives": results['output_graph']['true_positives'],
        "false_positives": results['output_graph']['false_positives'],
        "precision": tp/(tp+fp),
        "wasserstein_mean": results['output_graph']['wasserstein_distance']['mean'],
        "wasserstein_ratio": wasserstein_ratio,
        "false_omission_rate": results['false_omission_rate'],
        "negative_mean_wasserstein": results['negative_mean_wasserstein'],
        "final_loss": losses[-1] if losses else None,
    })
    wandb.finish()
    
    return results


def main():
    args = parse_args()
    run_causalbench_evaluation(args)


if __name__ == '__main__':
    main()
