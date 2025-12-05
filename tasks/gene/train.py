import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
from tqdm.auto import tqdm

from data.gene_data import GeneTrajectoryDataset
from models.ctm_gene import ContinuousThoughtMachineGENE
from utils.housekeeping import set_seed
from utils.schedulers import warmup, WarmupCosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_genes', type=int, default=50, help='Number of genes.')
    parser.add_argument('--iterations', type=int, default=20, help='Number of time steps.')
    
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
    dataset = GeneTrajectoryDataset(n_genes=args.n_genes, n_steps=args.iterations, n_samples=1000, seed=args.seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
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
                visualize(model, dataset, device, step, args.log_dir)

def visualize(model, dataset, device, step, log_dir):
    model.eval()
    with torch.no_grad():
        # Get one sample
        x0, true_traj = dataset[0]
        x0 = x0.unsqueeze(0).to(device) # (1, N)
        true_traj = true_traj.unsqueeze(0).to(device) # (1, N, T)
        
        # Run model with tracking
        # Returns: predictions, certainties, (synch_out_tracking, ...), ...
        # Note: CTM.forward return signature with track=True is:
        # predictions, certainties, (synch_out, synch_action), pre_act, post_act, attn
        # But CTMGENE.forward override returns:
        # predictions, certainties, synch_out_tracking, pre_act, post_act
        
        predictions, certainties, synch_out_tracking, pre_act, post_act = model(x0, track=True)
        
        pred_traj = predictions.cpu().numpy()[0] # (N, T)
        true_traj = true_traj.cpu().numpy()[0] # (N, T)
        
        # Plot Trajectories of first 5 genes
        plt.figure(figsize=(12, 6))
        for i in range(5):
            plt.plot(true_traj[i], linestyle='--', alpha=0.5, label=f'Gene {i} True')
            plt.plot(pred_traj[i], linestyle='-', alpha=0.8, label=f'Gene {i} Pred')
        plt.title(f'Gene Expression Trajectories (Step {step})')
        plt.legend()
        plt.savefig(f'{log_dir}/traj_{step}.png')
        plt.close()
        
        # Plot Synchronization Matrix (Final Step)
        # synch_out_tracking is shape (T, B, Synch_Dim) or similar? 
        # In CTMGENE: synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
        # synchronisation_out is vector of upper triangle if first-last
        # synch_out_tracking is list of arrays
        
        final_synch_vec = synch_out_tracking[-1][0] # (Synch_Dim,)
        
        # Reconstruct Matrix from Upper Triangle (assuming first-last)
        n_genes = model.d_model
        synch_matrix = np.zeros((n_genes, n_genes))
        
        # Logic matches CTM.compute_synchronisation for 'first-last'
        # i, j = torch.triu_indices(n_synch, n_synch)
        # pairwise_product = outer[:, i, j]
        
        triu_indices = torch.triu_indices(n_genes, n_genes)
        rows, cols = triu_indices
        rows, cols = rows.numpy(), cols.numpy()
        
        synch_matrix[rows, cols] = final_synch_vec
        synch_matrix[cols, rows] = final_synch_vec # Symmetric
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(synch_matrix, cmap='viridis')
        plt.title(f'Inferred Gene Regulation Synchronization (Step {step})')
        plt.savefig(f'{log_dir}/synch_{step}.png')
        plt.close()
        
    model.train()

if __name__ == '__main__':
    main()

