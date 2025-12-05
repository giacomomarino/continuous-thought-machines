import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
from tqdm.auto import tqdm
import networkx as nx

from data.gene_data import GeneTrajectoryDataset, MeningesDataset
from models.ctm_gene import ContinuousThoughtMachineGENE
from utils.housekeeping import set_seed
from utils.schedulers import warmup, WarmupCosineAnnealingLR
from matplotlib.colors import SymLogNorm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_real_data', action='store_true', help='Use real scRNA-seq data instead of synthetic.')
    
    # These defaults are for synthetic, but will be overridden by real data if --use_real_data is set
    parser.add_argument('--n_genes', type=int, default=50, help='Number of genes (overridden by real data).')
    parser.add_argument('--iterations', type=int, default=20, help='Number of time steps (overridden by real data).')
    
    parser.add_argument('--d_model', type=int, default=50, help='Dimension of the model (should match n_genes).')
    parser.add_argument('--synapse_depth', type=int, default=2, help='Depth of U-NET synapse.')
    parser.add_argument('--memory_length', type=int, default=10, help='Memory length.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--training_iterations', type=int, default=2000, help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_dir', type=str, default='logs/gene')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--track_every', type=int, default=200)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    if args.use_real_data:
        print("Using Real scRNA-seq Meninges Dataset...")
        dataset = MeningesDataset()
        args.n_genes = dataset.n_genes
        args.iterations = dataset.n_steps
        args.d_model = args.n_genes # Ensure model matches data
    else:
        print("Using Synthetic Gene Dataset...")
        dataset = GeneTrajectoryDataset(n_genes=args.n_genes, n_steps=args.iterations, n_samples=1000, seed=args.seed)
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Configuration: n_genes={args.n_genes}, iterations={args.iterations}")

    # Model
    # We map genes 1:1 to neurons, so d_model = n_genes
    # We want full synchronization matrix, so we use 'first-last' with n_synch_out = n_genes
    model = ContinuousThoughtMachineGENE(
        iterations=args.iterations,
        d_model=args.n_genes,
        d_input=args.n_genes, # Input is initial state
        heads=0,
        n_synch_out=args.n_genes, 
        n_synch_action=0,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=True,
        memory_hidden_dims=4,
        do_layernorm_nlm=False,
        backbone_type='none',
        positional_embedding_type='none',
        out_dims=args.n_genes,
        neuron_select_type='first-last', 
        dropout=0.0
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = WarmupCosineAnnealingLR(optimizer, 100, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
    criterion = torch.nn.MSELoss()
    
    iterator = iter(dataloader)
    losses = []
    
    print(f"Training on {device}...")
    
    with tqdm(total=args.training_iterations) as pbar:
        for step in range(args.training_iterations):
            try:
                x0, trajectory = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                x0, trajectory = next(iterator)
                
            x0 = x0.to(device) # (B, N)
            trajectory = trajectory.to(device) # (B, N, T)
            
            optimizer.zero_grad()
            
            # CTM Output: (B, N, T)
            predictions, certainties, synchronisation = model(x0)
            
            # Loss: Compare predicted trajectory to ground truth
            loss = criterion(predictions, trajectory)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
            
            if step % args.track_every == 0:
                visualize(model, dataset, device, step, args.log_dir, args.use_real_data)
    
    # Save checkpoint at the end of training
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': args,
        'gene_names': dataset.gene_names if hasattr(dataset, 'gene_names') else None,
    }
    checkpoint_path = os.path.join(args.log_dir, 'model_checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def visualize(model, dataset, device, step, log_dir, use_real_data):
    model.eval()
    with torch.no_grad():
        # Get one sample
        x0, true_traj = dataset[0]
        x0 = x0.unsqueeze(0).to(device) # (1, N)
        true_traj = true_traj.unsqueeze(0).to(device) # (1, N, T)
        
        # Run model with tracking
        predictions, certainties, synch_out_tracking, pre_act, post_act = model(x0, track=True)
        
        pred_traj = predictions.cpu().numpy()[0] # (N, T)
        true_traj = true_traj.cpu().numpy()[0] # (N, T)
        
        # Gene names (if real data)
        gene_names = dataset.gene_names if use_real_data and hasattr(dataset, 'gene_names') else [f"Gene {i}" for i in range(model.d_model)]
        
        # Select genes to plot: top 20 most variable in the TRUE trajectory for this sample
        # Calculate variance across time for each gene
        gene_variances = np.var(true_traj, axis=1)
        top_genes_idx = np.argsort(gene_variances)[-20:][::-1] # Descending order

        # Plot Trajectories of top 20 genes
        n_cols = 5
        n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True)
        axes = axes.flatten()
        
        for i, idx in enumerate(top_genes_idx):
            ax = axes[i]
            ax.plot(true_traj[idx], linestyle='--', alpha=0.6, label='True', color='black')
            ax.plot(pred_traj[idx], linestyle='-', alpha=0.8, label='Pred', color='tab:blue')
            ax.set_title(f'{gene_names[idx]}')
            if i == 0:
                ax.legend()
        
        plt.suptitle(f'Top 20 Variable Gene Trajectories (Step {step})')
        plt.tight_layout()
        plt.savefig(f'{log_dir}/traj_{step}.png')
        plt.close()
        
        # Plot Synchronization Matrix (Final Step)
        final_synch_vec = synch_out_tracking[-1][0] # (Synch_Dim,)
        
        # Reconstruct Matrix from Upper Triangle
        n_genes = model.d_model
        synch_matrix = np.zeros((n_genes, n_genes))
        
        triu_indices = torch.triu_indices(n_genes, n_genes)
        rows, cols = triu_indices
        rows, cols = rows.numpy(), cols.numpy()
        
        synch_matrix[rows, cols] = final_synch_vec
        synch_matrix[cols, rows] = final_synch_vec # Symmetric
        
        plt.figure(figsize=(12, 10))
        # Use SymLogNorm for log-scale visualization that handles negative values
        norm = SymLogNorm(linthresh=0.01, linscale=1.0, base=10)
        
        sns.heatmap(synch_matrix, cmap='viridis', norm=norm, cbar_kws={'label': 'Synchronization (Log Scale)'})
            
        plt.title(f'Inferred Gene Regulation Synchronization (Step {step})')
        plt.savefig(f'{log_dir}/synch_{step}.png')
        plt.close()

        # NEW: Plot Network Graph of Strongest Interactions
        try:
            G = nx.Graph()
            
            # Add nodes (all genes)
            for i in range(n_genes):
                G.add_node(i, label=gene_names[i])
            
            # Mask diagonal to ignore self-loops in top-k selection
            np.fill_diagonal(synch_matrix, 0)
            
            # Get indices of top 100 strongest edges
            triu_indices_np = np.triu_indices(n_genes, k=1)
            weights = synch_matrix[triu_indices_np]
            abs_weights = np.abs(weights)
            
            n_edges = 100
            top_indices = np.argsort(abs_weights)[-n_edges:]
            
            # Add edges to graph
            active_nodes = set()
            for idx in top_indices:
                row = triu_indices_np[0][idx]
                col = triu_indices_np[1][idx]
                w = weights[idx]
                G.add_edge(row, col, weight=w)
                active_nodes.add(row)
                active_nodes.add(col)
            
            # Filter graph to only show active nodes (connected in top 100)
            subgraph = G.subgraph(list(active_nodes))
            
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(subgraph, k=0.5, seed=42)
            
            edges = subgraph.edges(data=True)
            pos_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
            neg_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
            
            # Draw
            nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='lightgrey', alpha=0.8)
            
            # Draw labels only for nodes in the subgraph
            labels = {i: gene_names[i] for i in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
            
            nx.draw_networkx_edges(subgraph, pos, edgelist=pos_edges, edge_color='green', width=1.0, alpha=0.6)
            nx.draw_networkx_edges(subgraph, pos, edgelist=neg_edges, edge_color='red', width=1.0, alpha=0.6)
            
            plt.title(f'Top {n_edges} Strongest Regulatory Interactions (Step {step})')
            plt.axis('off')
            plt.savefig(f'{log_dir}/graph_{step}.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting graph: {e}")
        
    model.train()

if __name__ == '__main__':
    main()
