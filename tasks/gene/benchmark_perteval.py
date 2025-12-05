"""
PertEval-scFM Integration for CTM Gene Model.

PertEval-scFM is a standardized framework for evaluating perturbation predictions
from single-cell foundation models.

This module wraps the CTM in the PertEval interface for standardized evaluation.
See: https://github.com/aaronwtr/PertEval-scFM
"""
import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
import json
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.ctm_gene import ContinuousThoughtMachineGENE


def parse_args():
    parser = argparse.ArgumentParser(description='PertEval-scFM Benchmark for CTM')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained CTM checkpoint')
    parser.add_argument('--perturb_data', type=str, required=True,
                        help='Path to perturbation data (h5ad or pt format)')
    parser.add_argument('--output_dir', type=str, default='logs/benchmark_perteval',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    return parser.parse_args()


class CTMPerturbationPredictor:
    """
    Wrapper class that provides CTM perturbation predictions in PertEval format.
    """
    
    def __init__(self, model, gene_names, device='cpu'):
        self.model = model
        self.gene_names = gene_names
        self.gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
        self.device = device
        self.model.eval()
    
    def get_gene_idx(self, gene_name):
        return self.gene_to_idx.get(gene_name.upper(), None)
    
    def predict_perturbation(self, control_expr, perturbed_gene, perturbation_type='knockout'):
        gene_idx = self.get_gene_idx(perturbed_gene)
        if gene_idx is None:
            return None
        
        if len(control_expr.shape) == 1:
            control_expr = control_expr.unsqueeze(0)
        control_expr = control_expr.to(self.device)
        
        with torch.no_grad():
            control_pred, _, _ = self.model(control_expr)
            
            perturbed_input = control_expr.clone()
            if perturbation_type == 'knockout':
                perturbed_input[:, gene_idx] = 0.0
            elif perturbation_type == 'knockdown':
                perturbed_input[:, gene_idx] *= 0.5
            elif perturbation_type == 'overexpress':
                perturbed_input[:, gene_idx] *= 2.0
            
            perturbed_pred, _, _ = self.model(perturbed_input)
        
        return perturbed_pred.mean(dim=2).squeeze(0).cpu()
    
    def predict_batch_perturbations(self, control_expr, perturbed_genes, perturbation_type='knockout'):
        predictions = {}
        for gene in tqdm(perturbed_genes, desc='Predicting perturbations'):
            pred = self.predict_perturbation(control_expr, gene, perturbation_type)
            if pred is not None:
                predictions[gene] = pred
        return predictions


def compute_perteval_metrics(pred_expr, actual_expr, control_expr, gene_names):
    pred = np.array(pred_expr)
    actual = np.array(actual_expr)
    control = np.array(control_expr)
    
    mse = np.mean((pred - actual) ** 2)
    
    mask = (actual != 0) & (pred != 0)
    if mask.sum() > 10:
        pearson_r, _ = pearsonr(pred[mask], actual[mask])
        spearman_r, _ = spearmanr(pred[mask], actual[mask])
    else:
        pearson_r = 0.0
        spearman_r = 0.0
    
    pseudocount = 0.1
    pred_lfc = np.log2((pred + pseudocount) / (control + pseudocount))
    actual_lfc = np.log2((actual + pseudocount) / (control + pseudocount))
    
    lfc_mse = np.mean((pred_lfc - actual_lfc) ** 2)
    
    mask = np.isfinite(pred_lfc) & np.isfinite(actual_lfc)
    if mask.sum() > 10:
        lfc_pearson, _ = pearsonr(pred_lfc[mask], actual_lfc[mask])
        lfc_spearman, _ = spearmanr(pred_lfc[mask], actual_lfc[mask])
    else:
        lfc_pearson = 0.0
        lfc_spearman = 0.0
    
    k = min(100, len(gene_names))
    pred_top_de = set(np.argsort(np.abs(pred_lfc))[-k:])
    actual_top_de = set(np.argsort(np.abs(actual_lfc))[-k:])
    de_overlap = len(pred_top_de & actual_top_de) / k
    
    sign_mask = np.abs(actual_lfc) > 0.5
    if sign_mask.sum() > 0:
        direction_acc = np.mean(np.sign(pred_lfc[sign_mask]) == np.sign(actual_lfc[sign_mask]))
    else:
        direction_acc = 0.0
    
    metrics = {
        'mse': float(mse),
        'pearson': float(pearson_r) if np.isfinite(pearson_r) else 0.0,
        'spearman': float(spearman_r) if np.isfinite(spearman_r) else 0.0,
        'lfc_mse': float(lfc_mse),
        'lfc_pearson': float(lfc_pearson) if np.isfinite(lfc_pearson) else 0.0,
        'lfc_spearman': float(lfc_spearman) if np.isfinite(lfc_spearman) else 0.0,
        'de_overlap_100': float(de_overlap),
        'direction_accuracy': float(direction_acc),
    }
    
    return metrics


def load_perturbation_data(data_path):
    if data_path.endswith('.h5ad'):
        import anndata
        adata = anndata.read_h5ad(data_path)
        
        if 'condition' in adata.obs:
            control_mask = adata.obs['condition'] == 'control'
        elif 'perturbation' in adata.obs:
            control_mask = adata.obs['perturbation'].str.contains('ctrl|control', case=False)
        else:
            raise ValueError("Cannot find control condition in data")
        
        X = adata[control_mask].X
        control_expr = X.mean(axis=0).A1 if hasattr(X, 'A1') else np.array(X.mean(axis=0)).flatten()
        
        perturbations = {}
        perturb_col = 'perturbation' if 'perturbation' in adata.obs else 'condition'
        for perturb in adata.obs[perturb_col].unique():
            if 'ctrl' not in perturb.lower() and 'control' not in perturb.lower():
                mask = adata.obs[perturb_col] == perturb
                X_pert = adata[mask].X
                expr = X_pert.mean(axis=0)
                if hasattr(expr, 'A1'):
                    expr = expr.A1
                else:
                    expr = np.array(expr).flatten()
                perturbations[perturb] = expr
        
        gene_names = adata.var_names.tolist()
        
    elif data_path.endswith('.pt'):
        data = torch.load(data_path)
        control_expr = data['control_expr'].numpy()
        perturbations = {k: v.numpy() for k, v in data['perturbation_mean_expr'].items()}
        gene_names = data['gene_names']
        
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return control_expr, perturbations, gene_names


def run_perteval_benchmark(predictor, control_expr, perturbations, gene_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = []
    per_gene_metrics = {}
    
    control_tensor = torch.tensor(control_expr, dtype=torch.float32)
    
    for perturb_gene, actual_expr in tqdm(perturbations.items(), desc='Evaluating'):
        pred_expr = predictor.predict_perturbation(control_tensor, perturb_gene)
        
        if pred_expr is None:
            continue
        
        pred_expr = pred_expr.numpy()
        
        metrics = compute_perteval_metrics(pred_expr, actual_expr, control_expr, gene_names)
        metrics['gene'] = perturb_gene
        
        all_metrics.append(metrics)
        per_gene_metrics[perturb_gene] = metrics
    
    if len(all_metrics) == 0:
        print("No valid perturbations found. Check gene name overlap.")
        return None
    
    df = pd.DataFrame(all_metrics)
    
    summary = {
        'n_perturbations': len(all_metrics),
        'mean_mse': float(df['mse'].mean()),
        'mean_pearson': float(df['pearson'].mean()),
        'mean_spearman': float(df['spearman'].mean()),
        'mean_lfc_mse': float(df['lfc_mse'].mean()),
        'mean_lfc_pearson': float(df['lfc_pearson'].mean()),
        'mean_lfc_spearman': float(df['lfc_spearman'].mean()),
        'mean_de_overlap': float(df['de_overlap_100'].mean()),
        'mean_direction_accuracy': float(df['direction_accuracy'].mean()),
        'std_pearson': float(df['pearson'].std()),
        'std_lfc_pearson': float(df['lfc_pearson'].std()),
    }
    
    return summary, df, per_gene_metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading CTM from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        n_genes = model_args.n_genes if hasattr(model_args, 'n_genes') else model_args.d_model
        iterations = model_args.iterations
    else:
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
    ).to(args.device)
    
    dummy = torch.randn(1, n_genes, device=args.device)
    model(dummy)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loading perturbation data from {args.perturb_data}...")
    control_expr, perturbations, gene_names = load_perturbation_data(args.perturb_data)
    print(f"Loaded {len(perturbations)} perturbations, {len(gene_names)} genes")
    
    if 'gene_names' in checkpoint:
        model_gene_names = checkpoint['gene_names']
    else:
        from data.gene_data import MeningesDataset
        dataset = MeningesDataset()
        model_gene_names = dataset.gene_names
    
    predictor = CTMPerturbationPredictor(model, model_gene_names, args.device)
    
    model_genes = set(g.upper() for g in model_gene_names)
    data_genes = set(g.upper() for g in gene_names)
    perturb_genes = set(g.split('_')[0].upper() for g in perturbations.keys())
    
    overlap = model_genes & data_genes
    perturb_overlap = model_genes & perturb_genes
    print(f"Gene overlap: {len(overlap)}/{len(data_genes)} genes")
    print(f"Perturbation overlap: {len(perturb_overlap)}/{len(perturb_genes)} perturbations")
    
    print("\nRunning PertEval benchmark...")
    result = run_perteval_benchmark(predictor, control_expr, perturbations, gene_names, args.output_dir)
    
    if result is None:
        return
    
    summary, df, per_gene = result
    
    print("\n" + "="*50)
    print("PERTEVAL-SCFM BENCHMARK RESULTS")
    print("="*50)
    print(f"Perturbations evaluated: {summary['n_perturbations']}")
    print("-"*50)
    print("Expression Metrics:")
    print(f"  MSE: {summary['mean_mse']:.4f}")
    print(f"  Pearson: {summary['mean_pearson']:.4f} +/- {summary['std_pearson']:.4f}")
    print(f"  Spearman: {summary['mean_spearman']:.4f}")
    print("-"*50)
    print("Log Fold Change Metrics:")
    print(f"  LFC MSE: {summary['mean_lfc_mse']:.4f}")
    print(f"  LFC Pearson: {summary['mean_lfc_pearson']:.4f} +/- {summary['std_lfc_pearson']:.4f}")
    print(f"  LFC Spearman: {summary['mean_lfc_spearman']:.4f}")
    print("-"*50)
    print("DE Gene Metrics:")
    print(f"  DE overlap (top 100): {summary['mean_de_overlap']:.4f}")
    print(f"  Direction accuracy: {summary['mean_direction_accuracy']:.4f}")
    print("="*50)
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    df.to_csv(os.path.join(args.output_dir, 'per_gene_metrics.csv'), index=False)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(df['pearson'], bins=20, edgecolor='black')
    axes[0, 0].axvline(summary['mean_pearson'], color='r', linestyle='--')
    axes[0, 0].set_xlabel('Pearson Correlation')
    axes[0, 0].set_title('Expression Prediction Accuracy')
    
    axes[0, 1].hist(df['lfc_pearson'], bins=20, edgecolor='black')
    axes[0, 1].axvline(summary['mean_lfc_pearson'], color='r', linestyle='--')
    axes[0, 1].set_xlabel('LFC Pearson Correlation')
    axes[0, 1].set_title('Perturbation Effect Prediction')
    
    axes[1, 0].hist(df['de_overlap_100'], bins=20, edgecolor='black')
    axes[1, 0].axvline(summary['mean_de_overlap'], color='r', linestyle='--')
    axes[1, 0].set_xlabel('DE Gene Overlap')
    axes[1, 0].set_title('Top DE Gene Recovery')
    
    axes[1, 1].hist(df['direction_accuracy'], bins=20, edgecolor='black')
    axes[1, 1].axvline(summary['mean_direction_accuracy'], color='r', linestyle='--')
    axes[1, 1].set_xlabel('Direction Accuracy')
    axes[1, 1].set_title('Perturbation Direction Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_distributions.png'), dpi=150)
    plt.close()
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()

