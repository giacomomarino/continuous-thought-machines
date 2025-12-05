"""
CausalBench Benchmark for CTM Gene Model.

Evaluates CTM's GRN inference using real Perturb-seq interventional data.
Uses K562 and RPE1 datasets with biologically meaningful metrics.

Paper: https://arxiv.org/abs/2210.17283
GitHub: https://github.com/causalbench/causalbench

Datasets:
- weissmann_k562: K562 cells, day 6 Perturb-seq targeting DepMap essential genes
- weissmann_rpe1: RPE1 cells, day 7 Perturb-seq targeting DepMap essential genes
"""
import argparse
import os
import sys
import numpy as np
import torch
import json
from typing import List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.ctm_gene import ContinuousThoughtMachineGENE
from tasks.gene.benchmark_utils import reconstruct_synch_matrix

# CausalBench imports
try:
    from causalscbench.models.abstract_model import AbstractInferenceModel
    from causalscbench.models.training_regimes import TrainingRegime
    from causalscbench.evaluation.statistical_evaluation import Evaluator
    from causalscbench.data_access.create_dataset import CreateDataset
    from causalscbench.data_access.create_evaluation_datasets import CreateEvaluationDatasets
    from causalscbench.data_access.utils.splitting import DatasetSplitter
    CAUSALBENCH_AVAILABLE = True
except ImportError as e:
    print(f"CausalBench import error: {e}")
    CAUSALBENCH_AVAILABLE = False


CAUSALBENCH_DATASETS = ['weissmann_k562', 'weissmann_rpe1']
TRAINING_REGIMES = ['observational', 'partial_interventional', 'full_interventional']


def parse_args():
    parser = argparse.ArgumentParser(description='CausalBench Benchmark for CTM')
    parser.add_argument('--dataset', type=str, default='weissmann_k562',
                        choices=CAUSALBENCH_DATASETS,
                        help='CausalBench dataset to use')
    parser.add_argument('--training_regime', type=str, default='observational',
                        choices=TRAINING_REGIMES,
                        help='Training regime for CausalBench')
    parser.add_argument('--output_dir', type=str, default='logs/benchmark_causalbench',
                        help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='data/causalbench',
                        help='Directory for CausalBench data cache')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--training_iterations', type=int, default=400,
                        help='Number of training iterations for CTM')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_genes', type=int, default=200,
                        help='Number of top variable genes to use')
    parser.add_argument('--subset_data', type=float, default=1.0,
                        help='Fraction of data to use (for faster testing)')
    parser.add_argument('--top_k_edges', type=int, default=1000,
                        help='Number of top edges to return from CTM')
    parser.add_argument('--max_path_length', type=int, default=3,
                        help='Max path length for evaluation (-1 for unlimited)')
    return parser.parse_args()


class CTMInferenceModel(AbstractInferenceModel):
    """
    CTM wrapper that implements CausalBench's AbstractInferenceModel interface.
    
    The __call__ method learns a GRN from expression data and returns edges.
    """
    
    def __init__(self, n_steps=50, device='cpu', training_iterations=400, 
                 lr=1e-3, batch_size=32, top_k_edges=1000):
        self.n_steps = n_steps
        self.device = device
        self.training_iterations = training_iterations
        self.lr = lr
        self.batch_size = batch_size
        self.top_k_edges = top_k_edges
        self.model = None
        
    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0
    ) -> List[Tuple]:
        """
        Learn a GRN from expression data using CTM.
        
        Args:
            expression_matrix: (n_samples, n_genes) expression data
            interventions: list indicating which gene was perturbed per sample
            gene_names: list of gene names
            training_regime: observational, partial, or full interventional
            seed: random seed
            
        Returns:
            List of (gene_A, gene_B) tuples representing directed edges
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        n_genes = len(gene_names)
        
        # For observational regime, use only non-targeting (control) samples
        if training_regime == TrainingRegime.Observational:
            mask = np.array([i == "non-targeting" for i in interventions])
            expression_matrix = expression_matrix[mask]
            print(f"Using {expression_matrix.shape[0]} observational samples")
        
        # Build and train CTM
        print(f"Building CTM: {n_genes} genes, {self.n_steps} steps")
        self.model = self._build_model(n_genes)
        
        # Create pseudo-trajectories and train
        x0, trajectories = self._create_trajectories(expression_matrix)
        self._train(x0, trajectories)
        
        # Extract GRN edges from synchronization matrix
        edges = self._extract_edges(gene_names)
        
        return edges
    
    def _build_model(self, n_genes):
        """Build CTM model."""
        model = ContinuousThoughtMachineGENE(
            iterations=self.n_steps,
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
        ).to(self.device)
        
        # Initialize lazy modules
        dummy = torch.randn(1, n_genes, device=self.device)
        model(dummy)
        
        return model
    
    def _create_trajectories(self, expression_matrix, n_bins=50):
        """Create pseudo-trajectories from expression data using PCA ordering."""
        from sklearn.decomposition import PCA
        
        n_bins = min(n_bins, self.n_steps)
        
        # Order cells by PC1 (pseudo-time)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(expression_matrix).flatten()
        order = np.argsort(pc1)
        
        # Bin cells into trajectory steps
        bins = np.array_split(order, n_bins)
        
        # Sample trajectories
        n_samples = min(500, len(order) // max(1, n_bins // 5))
        trajectories = []
        initial_states = []
        
        for _ in range(n_samples):
            traj = []
            start_idx = np.random.choice(bins[0])
            initial_states.append(expression_matrix[start_idx])
            
            for bin_indices in bins:
                cell_idx = np.random.choice(bin_indices)
                traj.append(expression_matrix[cell_idx])
            
            trajectories.append(np.stack(traj))
        
        x0 = torch.tensor(np.stack(initial_states), dtype=torch.float32)
        traj = torch.tensor(np.stack(trajectories), dtype=torch.float32)
        traj = traj.permute(0, 2, 1)  # (B, N_genes, T)
        
        return x0, traj
    
    def _train(self, x0, trajectories):
        """Train CTM on trajectory data."""
        from utils.schedulers import WarmupCosineAnnealingLR
        
        dataset = torch.utils.data.TensorDataset(x0, trajectories)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = WarmupCosineAnnealingLR(
            optimizer, 50, self.training_iterations,
            warmup_start_lr=1e-20, eta_min=1e-7
        )
        criterion = torch.nn.MSELoss()
        
        iterator = iter(dataloader)
        
        pbar = tqdm(range(self.training_iterations), desc='Training CTM')
        for step in pbar:
            try:
                batch_x0, batch_traj = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch_x0, batch_traj = next(iterator)
            
            batch_x0 = batch_x0.to(self.device)
            batch_traj = batch_traj.to(self.device)
            
            optimizer.zero_grad()
            predictions, _, _ = self.model(batch_x0)
            loss = criterion(predictions, batch_traj)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_description(f'Loss: {loss.item():.4f}')
        
        self.model.eval()
    
    def _extract_edges(self, gene_names):
        """Extract top-k edges from CTM synchronization matrix."""
        n_genes = len(gene_names)
        
        with torch.no_grad():
            x = torch.randn(1, n_genes, device=self.device)
            _, _, synch_tracking, _, _ = self.model(x, track=True)
            final_synch = synch_tracking[-1][0]
            synch_matrix = reconstruct_synch_matrix(final_synch, n_genes)
        
        # Get top-k edges by absolute weight
        # Upper triangle only (no self-loops)
        triu_idx = np.triu_indices(n_genes, k=1)
        weights = synch_matrix[triu_idx]
        abs_weights = np.abs(weights)
        
        # Sort by absolute weight
        sorted_idx = np.argsort(abs_weights)[::-1]
        top_k = min(self.top_k_edges, len(sorted_idx))
        
        edges = []
        for i in range(top_k):
            idx = sorted_idx[i]
            gene_a = gene_names[triu_idx[0][idx]]
            gene_b = gene_names[triu_idx[1][idx]]
            edges.append((gene_a, gene_b))
        
        print(f"Extracted {len(edges)} edges from CTM")
        return edges


def run_causalbench_evaluation(args):
    """Run full CausalBench evaluation."""
    import matplotlib.pyplot as plt
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    if not CAUSALBENCH_AVAILABLE:
        print("CausalBench not available. Please install: pip install causalbench")
        return None
    
    # Load data using CausalBench's data loader
    print(f"Loading CausalBench dataset: {args.dataset}")
    print("This may download ~10GB of data on first run...")
    
    try:
        dataset_creator = CreateDataset(args.data_dir)
        
        if args.dataset == 'weissmann_k562':
            data = dataset_creator.load_weissmann(
                'k562',
                subset_data=args.subset_data
            )
        else:
            data = dataset_creator.load_weissmann(
                'rpe1', 
                subset_data=args.subset_data
            )
        
        expression_matrix = data['expression_matrix']
        interventions = data['interventions']
        gene_names = data['gene_names']
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure data is downloaded. Run:")
        print(f"  causalbench_run --dataset_name {args.dataset} --data_directory {args.data_dir} ...")
        return None
    
    print(f"Loaded: {expression_matrix.shape[0]} samples, {expression_matrix.shape[1]} genes")
    
    # Split data for training and evaluation
    splitter = DatasetSplitter(expression_matrix, interventions, gene_names)
    train_data, eval_data = splitter.split_data(
        train_fraction=0.8,
        seed=42
    )
    
    # Map training regime
    regime_map = {
        'observational': TrainingRegime.Observational,
        'partial_interventional': TrainingRegime.PartialIntervational,
        'full_interventional': TrainingRegime.Interventional,
    }
    training_regime = regime_map[args.training_regime]
    
    # Create CTM model
    ctm_model = CTMInferenceModel(
        n_steps=50,
        device=args.device,
        training_iterations=args.training_iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        top_k_edges=args.top_k_edges
    )
    
    # Run inference
    print("\nRunning CTM inference...")
    edges = ctm_model(
        expression_matrix=train_data['expression_matrix'],
        interventions=train_data['interventions'],
        gene_names=gene_names,
        training_regime=training_regime,
        seed=42
    )
    
    # Evaluate using CausalBench's evaluator
    print("\nEvaluating network...")
    evaluator = Evaluator(
        expression_matrix=eval_data['expression_matrix'],
        interventions=eval_data['interventions'],
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
    print("CAUSALBENCH RESULTS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Training regime: {args.training_regime}")
    print(f"Number of edges: {len(edges)}")
    print("-"*60)
    print("Output Graph:")
    print(f"  True Positives: {results['output_graph']['true_positives']}")
    print(f"  False Positives: {results['output_graph']['false_positives']}")
    print(f"  Wasserstein Distance (mean): {results['output_graph']['wasserstein_distance']['mean']:.4f}")
    print("-"*60)
    print(f"False Omission Rate: {results['false_omission_rate']:.4f}")
    print(f"Negative Mean Wasserstein: {results['negative_mean_wasserstein']:.4f}")
    print("="*60)
    
    # Save results
    results_to_save = {
        'dataset': args.dataset,
        'training_regime': args.training_regime,
        'n_edges': len(edges),
        'true_positives': results['output_graph']['true_positives'],
        'false_positives': results['output_graph']['false_positives'],
        'wasserstein_mean': results['output_graph']['wasserstein_distance']['mean'],
        'false_omission_rate': results['false_omission_rate'],
        'negative_mean_wasserstein': results['negative_mean_wasserstein'],
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save edges
    with open(os.path.join(args.output_dir, 'edges.txt'), 'w') as f:
        for a, b in edges[:100]:  # Save top 100
            f.write(f"{a}\t{b}\n")
    
    print(f"\nResults saved to {args.output_dir}")
    
    return results


def run_benchmark_without_data(args):
    """Run benchmark with synthetic data when CausalBench data unavailable."""
    import matplotlib.pyplot as plt
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Running with synthetic data (CausalBench data not available)")
    
    # Create synthetic data
    n_cells, n_genes = 1000, args.n_genes
    expression_matrix = np.random.lognormal(0, 1, (n_cells, n_genes)).astype(np.float32)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    interventions = ["non-targeting"] * n_cells
    
    if CAUSALBENCH_AVAILABLE:
        training_regime = TrainingRegime.Observational
    else:
        training_regime = None
    
    # Create and run CTM model
    ctm_model = CTMInferenceModel(
        n_steps=50,
        device=args.device,
        training_iterations=args.training_iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        top_k_edges=args.top_k_edges
    )
    
    if CAUSALBENCH_AVAILABLE:
        edges = ctm_model(
            expression_matrix=expression_matrix,
            interventions=interventions,
            gene_names=gene_names,
            training_regime=training_regime,
            seed=42
        )
    else:
        # Manual extraction without CausalBench
        ctm_model.model = ctm_model._build_model(n_genes)
        x0, traj = ctm_model._create_trajectories(expression_matrix)
        ctm_model._train(x0, traj)
        edges = ctm_model._extract_edges(gene_names)
    
    # Save results
    stats = {
        'dataset': 'synthetic',
        'n_genes': n_genes,
        'n_cells': n_cells,
        'n_edges': len(edges),
        'training_iterations': args.training_iterations,
    }
    
    with open(os.path.join(args.output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY (Synthetic Data)")
    print("="*50)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("="*50)
    
    print(f"\nResults saved to {args.output_dir}")
    
    return stats


def main():
    args = parse_args()
    
    # Try to run with real CausalBench data
    try:
        result = run_causalbench_evaluation(args)
        if result is None:
            # Fall back to synthetic
            run_benchmark_without_data(args)
    except Exception as e:
        print(f"CausalBench evaluation failed: {e}")
        print("Falling back to synthetic data...")
        run_benchmark_without_data(args)


if __name__ == '__main__':
    main()
