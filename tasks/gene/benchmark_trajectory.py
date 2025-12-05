"""
Trajectory Inference Benchmark using GSEA.

Tests whether the CTM's internal ticks mirror the biological differentiation process.
As the model "thinks," the active gene sets should reflect the known lineage progression.

Example: For hematopoiesis, we expect:
- Early ticks: HSC (Hematopoietic Stem Cell) signatures
- Middle ticks: Progenitor signatures
- Late ticks: Differentiated cell signatures
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from data.hematopoiesis_data import HematopoiesisDataset
from data.gene_data import MeningesDataset
from models.ctm_gene import ContinuousThoughtMachineGENE


def parse_args():
    parser = argparse.ArgumentParser(description='Trajectory GSEA Benchmark for CTM')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to trained CTM checkpoint (if None, trains new model)')
    parser.add_argument('--dataset', type=str, default='meninges', 
                        choices=['meninges', 'hematopoiesis'],
                        help='Dataset to use')
    parser.add_argument('--output_dir', type=str, default='logs/benchmark_trajectory',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--gene_set_library', type=str, default='GO_Biological_Process_2023',
                        help='Gene set library for GSEA')
    parser.add_argument('--n_ticks', type=int, default=5, 
                        help='Number of ticks to analyze')
    return parser.parse_args()


def load_or_train_model(checkpoint_path, dataset, device='cpu'):
    """
    Load a trained model or train a new one.
    """
    from utils.schedulers import WarmupCosineAnnealingLR
    
    n_genes = dataset.n_genes
    n_steps = dataset.n_steps
    
    model = ContinuousThoughtMachineGENE(
        iterations=n_steps,
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
    dummy_input = torch.randn(1, n_genes, device=device)
    model(dummy_input)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Training new model...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineAnnealingLR(optimizer, 100, 1000, 
                                            warmup_start_lr=1e-20, eta_min=1e-7)
        criterion = torch.nn.MSELoss()
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        iterator = iter(dataloader)
        
        for step in tqdm(range(1000), desc='Training'):
            try:
                x0, trajectory = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                x0, trajectory = next(iterator)
            
            x0 = x0.to(device)
            trajectory = trajectory.to(device)
            
            optimizer.zero_grad()
            predictions, _, _ = model(x0)
            loss = criterion(predictions, trajectory)
            loss.backward()
            optimizer.step()
            scheduler.step()
    
    model.eval()
    return model


def extract_activations_at_ticks(model, x0, tick_indices, device='cpu'):
    """
    Extract gene activations at specific ticks during the CTM's "thinking".
    
    Returns:
        activations: dict mapping tick_idx -> (B, N_genes) activation tensor
    """
    model.eval()
    x0 = x0.to(device)
    
    with torch.no_grad():
        # Run model with tracking
        _, _, synch_tracking, _, post_activations = model(x0, track=True)
    
    activations = {}
    for tick_idx in tick_indices:
        if tick_idx < len(post_activations):
            # post_activations are returned as numpy arrays from model.forward(track=True)
            act = post_activations[tick_idx]
            if isinstance(act, np.ndarray):
                activations[tick_idx] = torch.from_numpy(act)
            else:
                activations[tick_idx] = act.cpu()
    
    return activations


def compute_mean_activations(activations_dict, gene_names):
    """
    Compute mean activation per gene at each tick.
    Returns a dict of {tick: pd.Series of gene->activation}.
    """
    import pandas as pd
    
    results = {}
    for tick, act in activations_dict.items():
        mean_act = act.mean(dim=0).numpy()
        results[tick] = pd.Series(mean_act, index=gene_names)
    
    return results


def run_gsea_at_tick(activation_series, gene_set_library='GO_Biological_Process_2023', 
                     organism='human'):
    """
    Run GSEA prerank on the activations at a specific tick.
    
    Args:
        activation_series: pd.Series with gene names as index, activation as values
        gene_set_library: MSigDB gene set library name
        organism: 'human' or 'mouse'
    
    Returns:
        GSEA results dataframe
    """
    try:
        import gseapy as gp
    except ImportError:
        print("gseapy not installed. Install with: pip install gseapy")
        return None
    
    # Remove genes with zero activation
    activation_series = activation_series[activation_series != 0]
    
    if len(activation_series) < 10:
        print("Warning: Too few genes with non-zero activation")
        return None
    
    try:
        # Run prerank GSEA
        pre_res = gp.prerank(
            rnk=activation_series.sort_values(ascending=False),
            gene_sets=gene_set_library,
            threads=4,
            min_size=5,
            max_size=500,
            permutation_num=100,  # Lower for speed
            outdir=None,
            seed=42,
            verbose=False,
        )
        
        results = pre_res.res2d
        return results
    
    except Exception as e:
        print(f"GSEA failed: {e}")
        return None


def analyze_trajectory(model, dataset, tick_indices, gene_set_library, output_dir, device='cpu'):
    """
    Run full trajectory analysis with GSEA at each tick.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample input
    x0 = dataset.x0[:10].to(device)  # Use first 10 samples
    gene_names = dataset.gene_names
    
    # Extract activations
    print("Extracting activations at ticks...")
    activations = extract_activations_at_ticks(model, x0, tick_indices, device)
    mean_activations = compute_mean_activations(activations, gene_names)
    
    # Run GSEA at each tick
    gsea_results = {}
    for tick in tqdm(tick_indices, desc='Running GSEA'):
        print(f"\nTick {tick}:")
        result = run_gsea_at_tick(
            mean_activations[tick], 
            gene_set_library=gene_set_library
        )
        if result is not None:
            gsea_results[tick] = result
            # Print top enriched terms
            significant = result[result['FDR q-val'] < 0.25].head(5)
            if len(significant) > 0:
                print(f"  Top enriched terms (FDR < 0.25):")
                for _, row in significant.iterrows():
                    print(f"    {row['Term']}: NES={row['NES']:.2f}, FDR={row['FDR q-val']:.3f}")
            else:
                print("  No significant enrichments")
    
    return mean_activations, gsea_results


def visualize_trajectory_gsea(mean_activations, gsea_results, output_dir, tick_indices):
    """
    Visualize the GSEA results across ticks.
    """
    import pandas as pd
    
    # Plot 1: Activation heatmap across ticks
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create activation matrix (genes x ticks)
    genes = list(mean_activations[tick_indices[0]].index)
    act_matrix = np.zeros((len(genes), len(tick_indices)))
    
    for i, tick in enumerate(tick_indices):
        act_matrix[:, i] = mean_activations[tick].values
    
    # Normalize per gene for visualization
    act_matrix_norm = (act_matrix - act_matrix.min(axis=1, keepdims=True)) / \
                      (act_matrix.max(axis=1, keepdims=True) - act_matrix.min(axis=1, keepdims=True) + 1e-10)
    
    import seaborn as sns
    sns.heatmap(act_matrix_norm, xticklabels=[f'Tick {t}' for t in tick_indices],
                yticklabels=genes if len(genes) < 50 else [],
                cmap='viridis', ax=ax)
    ax.set_title('Gene Activations Across CTM Ticks')
    ax.set_xlabel('Internal Tick')
    ax.set_ylabel('Genes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_heatmap.png'), dpi=150)
    plt.close()
    
    # Plot 2: Top GSEA terms per tick
    if gsea_results:
        fig, axes = plt.subplots(1, len(tick_indices), figsize=(5*len(tick_indices), 8))
        if len(tick_indices) == 1:
            axes = [axes]
        
        for i, tick in enumerate(tick_indices):
            if tick in gsea_results and gsea_results[tick] is not None:
                df = gsea_results[tick].head(10)
                
                # Truncate long term names
                terms = [t[:40] + '...' if len(t) > 40 else t for t in df['Term']]
                nes = df['NES'].values
                
                colors = ['green' if n > 0 else 'red' for n in nes]
                axes[i].barh(terms, nes, color=colors)
                axes[i].set_xlabel('NES')
                axes[i].set_title(f'Tick {tick}')
                axes[i].axvline(0, color='black', linestyle='-', linewidth=0.5)
            else:
                axes[i].text(0.5, 0.5, 'No results', ha='center', va='center')
                axes[i].set_title(f'Tick {tick}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gsea_per_tick.png'), dpi=150)
        plt.close()
    
    # Plot 3: Gene activation trajectories (top variable genes)
    gene_variance = np.var(act_matrix, axis=1)
    top_genes_idx = np.argsort(gene_variance)[-20:]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in top_genes_idx:
        ax.plot(tick_indices, act_matrix[idx], alpha=0.7, label=genes[idx])
    
    ax.set_xlabel('Internal Tick')
    ax.set_ylabel('Activation')
    ax.set_title('Top 20 Most Variable Genes Across CTM Ticks')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_genes_trajectory.png'), dpi=150)
    plt.close()


def compute_trajectory_metrics(mean_activations, gsea_results, tick_indices, expected_progression=None):
    """
    Compute metrics for trajectory quality.
    
    Args:
        expected_progression: Optional dict mapping tick to expected enriched terms
    """
    metrics = {}
    
    # Metric 1: Gene activation dynamics (variance over ticks)
    genes = list(mean_activations[tick_indices[0]].index)
    act_matrix = np.zeros((len(genes), len(tick_indices)))
    for i, tick in enumerate(tick_indices):
        act_matrix[:, i] = mean_activations[tick].values
    
    # Average variance across genes (higher = more dynamic)
    metrics['mean_gene_variance'] = float(np.mean(np.var(act_matrix, axis=1)))
    
    # Number of genes with significant change (> 0.5 std)
    gene_std = np.std(act_matrix, axis=1)
    metrics['n_dynamic_genes'] = int(np.sum(gene_std > 0.5))
    
    # Metric 2: GSEA term diversity
    if gsea_results:
        all_terms = set()
        for tick, df in gsea_results.items():
            if df is not None:
                significant = df[df['FDR q-val'] < 0.25]
                all_terms.update(significant['Term'].tolist())
        metrics['n_significant_terms_total'] = len(all_terms)
        
        # Terms per tick
        metrics['terms_per_tick'] = {}
        for tick, df in gsea_results.items():
            if df is not None:
                n_sig = len(df[df['FDR q-val'] < 0.25])
                metrics['terms_per_tick'][f'tick_{tick}'] = n_sig
    
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'hematopoiesis':
        try:
            dataset = HematopoiesisDataset()
        except FileNotFoundError:
            print("Hematopoiesis data not found. Please download first:")
            print("  python data/hematopoiesis_data.py")
            return
    else:
        dataset = MeningesDataset()
    
    print(f"Dataset: {dataset.n_genes} genes, {dataset.n_samples} samples, {dataset.n_steps} steps")
    
    # Load or train model
    model = load_or_train_model(args.checkpoint, dataset, args.device)
    
    # Define ticks to analyze
    total_ticks = dataset.n_steps
    tick_indices = np.linspace(0, total_ticks - 1, args.n_ticks, dtype=int).tolist()
    print(f"Analyzing ticks: {tick_indices}")
    
    # Run analysis
    print("\nRunning trajectory analysis...")
    mean_activations, gsea_results = analyze_trajectory(
        model, dataset, tick_indices, args.gene_set_library, args.output_dir, args.device
    )
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_trajectory_gsea(mean_activations, gsea_results, args.output_dir, tick_indices)
    
    # Compute metrics
    metrics = compute_trajectory_metrics(mean_activations, gsea_results, tick_indices)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAJECTORY BENCHMARK RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Ticks analyzed: {tick_indices}")
    print(f"Mean gene variance: {metrics['mean_gene_variance']:.4f}")
    print(f"Dynamic genes: {metrics['n_dynamic_genes']}")
    if 'n_significant_terms_total' in metrics:
        print(f"Significant GSEA terms: {metrics['n_significant_terms_total']}")
    print("="*50)
    
    # Save results
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save activation data
    import pandas as pd
    for tick, series in mean_activations.items():
        series.to_csv(os.path.join(args.output_dir, f'activations_tick_{tick}.csv'))
    
    # Save GSEA results
    if gsea_results:
        for tick, df in gsea_results.items():
            if df is not None:
                df.to_csv(os.path.join(args.output_dir, f'gsea_tick_{tick}.csv'))
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()

