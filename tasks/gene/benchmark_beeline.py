"""
BEELINE Benchmark for CTM Gene Model.

Evaluates the CTM's inferred Gene Regulatory Network against ChIP-seq validated edges.
Primary metric: AUPRC (Area Under Precision-Recall Curve)
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

from data.beeline_data import BeelineDataset, BEELINE_DATASETS
from models.ctm_gene import ContinuousThoughtMachineGENE
from tasks.gene.benchmark_utils import (
    compute_auprc,
    compute_auroc,
    compute_early_precision,
    reconstruct_synch_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description='BEELINE GRN Benchmark for CTM')
    parser.add_argument('--dataset', type=str, default='hESC', 
                        choices=list(BEELINE_DATASETS.keys()),
                        help='BEELINE dataset to use')
    parser.add_argument('--output_dir', type=str, default='logs/benchmark_beeline',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--training_iterations', type=int, default=2000,
                        help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    return parser.parse_args()


def train_ctm_on_beeline(dataset, device='cpu', training_iterations=2000, lr=1e-3, batch_size=32):
    """
    Train a CTM model on BEELINE data.
    """
    from utils.schedulers import WarmupCosineAnnealingLR
    
    n_genes = dataset.n_genes
    n_steps = dataset.n_steps
    
    print(f"Training CTM: {n_genes} genes, {n_steps} time steps")
    
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupCosineAnnealingLR(optimizer, 100, training_iterations, 
                                        warmup_start_lr=1e-20, eta_min=1e-7)
    criterion = torch.nn.MSELoss()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    iterator = iter(dataloader)
    
    losses = []
    pbar = tqdm(range(training_iterations), desc='Training')
    
    for step in pbar:
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
        
        losses.append(loss.item())
        pbar.set_description(f'Loss: {loss.item():.4f}')
    
    return model, losses


def extract_grn_from_ctm(model, dataset, device='cpu'):
    """
    Extract the inferred GRN (synchronization matrix) from the trained CTM.
    """
    model.eval()
    
    with torch.no_grad():
        # Use mean of all samples as input
        x0 = dataset.x0.mean(dim=0, keepdim=True).to(device)
        
        # Run model with tracking
        _, _, synch_tracking, _, _ = model(x0, track=True)
        
        # Get final synchronization vector
        final_synch_vec = synch_tracking[-1][0]  # (Synch_Dim,)
        
        # Reconstruct full matrix
        n_genes = model.d_model
        synch_matrix = reconstruct_synch_matrix(final_synch_vec, n_genes)
    
    return synch_matrix


def evaluate_grn(pred_matrix, ground_truth_matrix, gene_names):
    """
    Evaluate predicted GRN against ground truth.
    """
    # Flatten matrices (upper triangle only to avoid counting edges twice)
    n = pred_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    pred_flat = np.abs(pred_matrix[triu_idx])  # Use absolute value for ranking
    gt_flat = ground_truth_matrix[triu_idx]
    
    # Binarize ground truth
    gt_binary = (gt_flat > 0).astype(int)
    
    # Compute metrics
    auprc = compute_auprc(gt_binary, pred_flat)
    auroc = compute_auroc(gt_binary, pred_flat)
    ep100 = compute_early_precision(gt_binary, pred_flat, k=100)
    ep500 = compute_early_precision(gt_binary, pred_flat, k=500)
    
    # Random baseline
    n_positives = gt_binary.sum()
    n_total = len(gt_binary)
    random_auprc = n_positives / n_total
    
    results = {
        'auprc': auprc,
        'auroc': auroc,
        'early_precision_100': ep100,
        'early_precision_500': ep500,
        'random_baseline': random_auprc,
        'n_ground_truth_edges': int(n_positives),
        'n_possible_edges': int(n_total),
    }
    
    return results, pred_flat, gt_binary


def run_benchmark(args):
    """
    Run the full BEELINE benchmark.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BEELINE dataset
    print(f"Loading BEELINE {args.dataset} dataset...")
    dataset = BeelineDataset(args.dataset)
    
    print(f"Dataset: {dataset.n_genes} genes, {dataset.n_samples} samples")
    print(f"Ground truth edges: {len(dataset.ground_truth_edges)}")
    
    # Train CTM
    print("\nTraining CTM on BEELINE data...")
    model, losses = train_ctm_on_beeline(
        dataset, args.device, args.training_iterations, args.lr, args.batch_size
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'), dpi=150)
    plt.close()
    
    # Extract GRN from trained CTM
    print("\nExtracting inferred GRN...")
    pred_matrix = extract_grn_from_ctm(model, dataset, args.device)
    
    # Get ground truth matrix
    gt_matrix = dataset.get_ground_truth_matrix()
    
    # Evaluate
    print("\nEvaluating GRN...")
    results, pred_flat, gt_binary = evaluate_grn(pred_matrix, gt_matrix, dataset.gene_names)
    
    # Print results
    print("\n" + "="*50)
    print("BEELINE BENCHMARK RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Ground truth edges: {results['n_ground_truth_edges']}")
    print(f"Possible edges: {results['n_possible_edges']}")
    print("-"*50)
    print(f"AUPRC: {results['auprc']:.4f} (random: {results['random_baseline']:.4f})")
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"Early Precision @100: {results['early_precision_100']:.4f}")
    print(f"Early Precision @500: {results['early_precision_500']:.4f}")
    print("="*50)
    
    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot PR curve
    plot_pr_curve(
        gt_binary, pred_flat,
        title=f'Precision-Recall Curve ({args.dataset})',
        save_path=os.path.join(args.output_dir, 'pr_curve.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        gt_binary, pred_flat,
        title=f'ROC Curve ({args.dataset})',
        save_path=os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # Plot predicted GRN heatmap
    plt.figure(figsize=(12, 10))
    import seaborn as sns
    sns.heatmap(np.abs(pred_matrix), cmap='viridis')
    plt.title(f'Inferred GRN ({args.dataset})')
    plt.savefig(os.path.join(args.output_dir, 'inferred_grn.png'), dpi=150)
    plt.close()
    
    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'dataset': args.dataset,
        'results': results,
    }, os.path.join(args.output_dir, 'model_checkpoint.pt'))
    
    print(f"\nResults saved to {args.output_dir}")
    
    return results


def main():
    args = parse_args()
    results = run_benchmark(args)


if __name__ == '__main__':
    main()

