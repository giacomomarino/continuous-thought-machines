"""
Shared utilities for CTM gene benchmarks.
"""
import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_auprc(y_true, y_score):
    """
    Compute Area Under Precision-Recall Curve.
    Primary metric for GRN evaluation.
    """
    return average_precision_score(y_true, y_score)


def compute_auroc(y_true, y_score):
    """
    Compute Area Under ROC Curve.
    """
    return roc_auc_score(y_true, y_score)


def compute_early_precision(y_true, y_score, k=100):
    """
    Compute precision at top K predictions.
    """
    # Get top K indices
    top_k_idx = np.argsort(y_score)[-k:]
    
    # Compute precision
    precision = np.sum(y_true[top_k_idx]) / k
    return precision


def pearson_correlation(pred, actual):
    """
    Compute Pearson correlation coefficient.
    """
    pred = np.array(pred).flatten()
    actual = np.array(actual).flatten()
    
    # Remove NaN and Inf
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[mask]
    actual = actual[mask]
    
    if len(pred) < 2:
        return 0.0
    
    return np.corrcoef(pred, actual)[0, 1]


def direction_accuracy(pred, actual):
    """
    Compute fraction of genes where predicted LFC sign matches actual.
    """
    pred = np.array(pred).flatten()
    actual = np.array(actual).flatten()
    
    # Only consider genes with non-zero actual change
    mask = np.abs(actual) > 0.1
    if mask.sum() == 0:
        return 0.0
    
    pred_sign = np.sign(pred[mask])
    actual_sign = np.sign(actual[mask])
    
    return np.mean(pred_sign == actual_sign)


def top_k_recall(pred, actual, k=50):
    """
    Check if top K predicted affected genes are also top K in ground truth.
    """
    pred = np.array(pred).flatten()
    actual = np.array(actual).flatten()
    
    # Get top K by absolute value
    pred_top_k = set(np.argsort(np.abs(pred))[-k:])
    actual_top_k = set(np.argsort(np.abs(actual))[-k:])
    
    overlap = len(pred_top_k & actual_top_k)
    return overlap / k


def synch_matrix_to_edge_list(synch_matrix, gene_names, threshold=None, top_k=None):
    """
    Convert synchronization matrix to edge list.
    
    Args:
        synch_matrix: NxN matrix of synchronization values
        gene_names: list of gene names
        threshold: minimum absolute value to include edge
        top_k: if set, only return top K edges by absolute value
    
    Returns:
        List of (gene1, gene2, weight) tuples
    """
    n = len(gene_names)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only
            weight = synch_matrix[i, j]
            if threshold is not None and np.abs(weight) < threshold:
                continue
            edges.append((gene_names[i], gene_names[j], weight))
    
    # Sort by absolute weight
    edges.sort(key=lambda x: np.abs(x[2]), reverse=True)
    
    if top_k is not None:
        edges = edges[:top_k]
    
    return edges


def reconstruct_synch_matrix(synch_vec, n_genes):
    """
    Reconstruct full synchronization matrix from upper triangle vector.
    """
    synch_matrix = np.zeros((n_genes, n_genes))
    
    triu_indices = np.triu_indices(n_genes)
    synch_matrix[triu_indices] = synch_vec
    synch_matrix = synch_matrix + synch_matrix.T
    np.fill_diagonal(synch_matrix, synch_matrix.diagonal() / 2)
    
    return synch_matrix


def plot_pr_curve(y_true, y_score, title='Precision-Recall Curve', save_path=None):
    """
    Plot precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'AUPRC = {auprc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add random baseline
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Random = {baseline:.3f}')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_score, title='ROC Curve', save_path=None):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = compute_auroc(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_scatter(pred, actual, title='Predicted vs Actual LFC', save_path=None):
    """
    Scatter plot of predicted vs actual values.
    """
    pred = np.array(pred).flatten()
    actual = np.array(actual).flatten()
    
    corr = pearson_correlation(pred, actual)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, pred, alpha=0.5, s=10)
    
    # Add diagonal
    lims = [min(pred.min(), actual.min()), max(pred.max(), actual.max())]
    plt.plot(lims, lims, 'r--', alpha=0.5)
    
    plt.xlabel('Actual LFC')
    plt.ylabel('Predicted LFC')
    plt.title(f'{title}\nPearson r = {corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_gene_index(gene_name, gene_names):
    """
    Get index of a gene by name (case-insensitive).
    """
    for i, g in enumerate(gene_names):
        if g.upper() == gene_name.upper():
            return i
    return None


def run_knockout_prediction(model, control_input, knockout_gene_idx, device='cpu'):
    """
    Run CTM prediction with a gene knocked out.
    
    Args:
        model: trained CTM model
        control_input: control gene expression (B, N_genes)
        knockout_gene_idx: index of gene to knock out
        device: torch device
    
    Returns:
        control_pred: predictions without knockout
        knockout_pred: predictions with knockout
    """
    model.eval()
    with torch.no_grad():
        # Control prediction
        control_input = control_input.to(device)
        control_pred, _, _ = model(control_input)
        
        # Knockout prediction
        knockout_input = control_input.clone()
        knockout_input[:, knockout_gene_idx] = 0.0
        knockout_pred, _, _ = model(knockout_input)
        
    return control_pred, knockout_pred


def compute_predicted_lfc(control_pred, knockout_pred, pseudocount=1e-6):
    """
    Compute predicted log fold change from control and knockout predictions.
    
    Args:
        control_pred: (B, N_genes, T) control predictions
        knockout_pred: (B, N_genes, T) knockout predictions
        pseudocount: small value to avoid log(0)
    
    Returns:
        lfc: (N_genes,) mean log fold change across time and batch
    """
    # Take mean across time and batch
    control_mean = control_pred.mean(dim=(0, 2)).cpu().numpy()  # (N_genes,)
    knockout_mean = knockout_pred.mean(dim=(0, 2)).cpu().numpy()  # (N_genes,)
    
    # Log fold change
    lfc = np.log2((knockout_mean + pseudocount) / (control_mean + pseudocount))
    
    return lfc

