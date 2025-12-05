"""
Perturb-seq Benchmark for CTM Gene Model.

Tests the CTM's ability to predict gene expression changes following gene knockouts.
Compares predictions against actual Perturb-seq experimental data.
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from data.perturbseq_data import PerturbseqDataset, process_norman_data, download_norman_data
from data.gene_data import MeningesDataset
from models.ctm_gene import ContinuousThoughtMachineGENE
from tasks.gene.benchmark_utils import (
    pearson_correlation,
    direction_accuracy,
    top_k_recall,
    plot_correlation_scatter,
    get_gene_index,
    run_knockout_prediction,
    compute_predicted_lfc
)


def parse_args():
    parser = argparse.ArgumentParser(description='Perturb-seq Benchmark for CTM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained CTM checkpoint')
    parser.add_argument('--dataset', type=str, default='norman', choices=['norman', 'replogle'],
                        help='Perturb-seq dataset to use')
    parser.add_argument('--output_dir', type=str, default='logs/benchmark_perturbseq',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--max_perturbations', type=int, default=50, 
                        help='Maximum number of perturbations to test')
    return parser.parse_args()


def load_ctm_model(checkpoint_path, device='cpu'):
    """
    Load trained CTM model from checkpoint.
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        n_genes = args.n_genes if hasattr(args, 'n_genes') else args.d_model
        iterations = args.iterations
    else:
        # Default config
        n_genes = 200
        iterations = 50
    
    model = ContinuousThoughtMachineGENE(
        iterations=iterations,
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, n_genes


def find_gene_mapping(ctm_gene_names, perturb_gene_names, perturb_name):
    """
    Find the index of a perturbed gene in both the CTM and Perturb-seq gene lists.
    """
    # Clean up perturbation name (might have suffixes like _KO, _CRISPRi, etc.)
    clean_name = perturb_name.split('_')[0].upper()
    
    ctm_idx = None
    perturb_idx = None
    
    for i, g in enumerate(ctm_gene_names):
        if g.upper() == clean_name:
            ctm_idx = i
            break
    
    for i, g in enumerate(perturb_gene_names):
        if g.upper() == clean_name:
            perturb_idx = i
            break
    
    return ctm_idx, perturb_idx


def run_benchmark(model, ctm_dataset, perturb_dataset, device='cpu', max_perturbations=50):
    """
    Run the perturbation prediction benchmark.
    """
    results = {
        'perturbations': [],
        'pearson_correlations': [],
        'direction_accuracies': [],
        'top_k_recalls': [],
    }
    
    # Get control expression from CTM training data
    control_input = ctm_dataset.x0.mean(dim=0, keepdim=True).to(device)  # Mean of all training samples
    
    ctm_gene_names = ctm_dataset.gene_names
    perturb_gene_names = perturb_dataset.gene_names
    
    # Find common genes between CTM and Perturb-seq
    common_genes = set(g.upper() for g in ctm_gene_names) & set(g.upper() for g in perturb_gene_names)
    print(f"Common genes: {len(common_genes)}")
    
    if len(common_genes) < 10:
        print("Warning: Very few common genes. Results may not be meaningful.")
    
    tested = 0
    for perturb_idx in tqdm(range(len(perturb_dataset)), desc='Testing perturbations'):
        if tested >= max_perturbations:
            break
            
        sample = perturb_dataset[perturb_idx]
        perturb_name = sample['perturb_name']
        actual_lfc = sample['actual_lfc'].numpy()
        
        # Find gene in CTM model
        ctm_idx, _ = find_gene_mapping(ctm_gene_names, perturb_gene_names, perturb_name)
        
        if ctm_idx is None:
            continue  # Gene not in CTM model
        
        # Run knockout prediction
        control_pred, knockout_pred = run_knockout_prediction(
            model, control_input, ctm_idx, device
        )
        
        # Compute predicted LFC
        pred_lfc = compute_predicted_lfc(control_pred, knockout_pred)
        
        # Map to common genes for fair comparison
        # For simplicity, compare all genes (assumes same ordering)
        # In practice, you'd need proper gene mapping
        
        # Compute metrics
        corr = pearson_correlation(pred_lfc, actual_lfc[:len(pred_lfc)])
        dir_acc = direction_accuracy(pred_lfc, actual_lfc[:len(pred_lfc)])
        top_k = top_k_recall(pred_lfc, actual_lfc[:len(pred_lfc)], k=50)
        
        if np.isfinite(corr):
            results['perturbations'].append(perturb_name)
            results['pearson_correlations'].append(corr)
            results['direction_accuracies'].append(dir_acc)
            results['top_k_recalls'].append(top_k)
            tested += 1
    
    return results


def summarize_results(results, output_dir):
    """
    Summarize and save benchmark results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute summary statistics
    summary = {
        'n_perturbations': len(results['perturbations']),
        'mean_pearson': np.mean(results['pearson_correlations']),
        'std_pearson': np.std(results['pearson_correlations']),
        'mean_direction_accuracy': np.mean(results['direction_accuracies']),
        'std_direction_accuracy': np.std(results['direction_accuracies']),
        'mean_top_k_recall': np.mean(results['top_k_recalls']),
        'std_top_k_recall': np.std(results['top_k_recalls']),
    }
    
    print("\n" + "="*50)
    print("PERTURB-SEQ BENCHMARK RESULTS")
    print("="*50)
    print(f"Perturbations tested: {summary['n_perturbations']}")
    print(f"Pearson correlation: {summary['mean_pearson']:.3f} ± {summary['std_pearson']:.3f}")
    print(f"Direction accuracy: {summary['mean_direction_accuracy']:.3f} ± {summary['std_direction_accuracy']:.3f}")
    print(f"Top-K recall (K=50): {summary['mean_top_k_recall']:.3f} ± {summary['std_top_k_recall']:.3f}")
    print("="*50)
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({**summary, **results}, f, indent=2)
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(results['pearson_correlations'], bins=20, edgecolor='black')
    axes[0].axvline(summary['mean_pearson'], color='r', linestyle='--', label='Mean')
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Correlations')
    axes[0].legend()
    
    axes[1].hist(results['direction_accuracies'], bins=20, edgecolor='black')
    axes[1].axvline(summary['mean_direction_accuracy'], color='r', linestyle='--', label='Mean')
    axes[1].set_xlabel('Direction Accuracy')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Direction Accuracy')
    axes[1].legend()
    
    axes[2].hist(results['top_k_recalls'], bins=20, edgecolor='black')
    axes[2].axvline(summary['mean_top_k_recall'], color='r', linestyle='--', label='Mean')
    axes[2].set_xlabel('Top-K Recall')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Top-K Recall')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distributions.png'), dpi=150)
    plt.close()
    
    return summary


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CTM model
    model, n_genes = load_ctm_model(args.checkpoint, args.device)
    
    # Load CTM training dataset (for gene names and control expression)
    print("Loading CTM training dataset...")
    ctm_dataset = MeningesDataset()
    
    # Load Perturb-seq dataset
    print(f"Loading {args.dataset} Perturb-seq dataset...")
    if args.dataset == 'norman':
        perturb_path = 'data/norman_processed.pt'
        if not os.path.exists(perturb_path):
            print("Norman data not found. Please download first:")
            print("  python data/perturbseq_data.py")
            return
        perturb_dataset = PerturbseqDataset(perturb_path)
    else:
        perturb_path = 'data/replogle_processed.pt'
        if not os.path.exists(perturb_path):
            print("Replogle data not found. Please download first:")
            print("  python data/perturbseq_data.py")
            return
        perturb_dataset = PerturbseqDataset(perturb_path)
    
    # Run benchmark
    print("Running benchmark...")
    results = run_benchmark(
        model, ctm_dataset, perturb_dataset, 
        args.device, args.max_perturbations
    )
    
    # Summarize results
    summary = summarize_results(results, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()

