"""
Attention-Based CTM for Gene Regulatory Network Inference.

This variant replaces the U-Net synapse model with self-attention,
enabling DIRECTIONAL (causal) GRN inference.

Key insight: The attention weights A[i,j] directly represent
"To compute gene j's next state, attend to gene i's current value"
This is ASYMMETRIC (A[i,j] != A[j,i]), capturing causal regulation.

Paper: Continuous Thought Machines (https://arxiv.org/abs/2505.05522)
"""
import torch
import torch.nn as nn
import numpy as np
from models.ctm import ContinuousThoughtMachine
from models.modules import AttentionSynapse


class ContinuousThoughtMachineGENEAttention(ContinuousThoughtMachine):
    """
    CTM for Gene Regulatory Networks using Attention-based Synapses.
    
    Key differences from standard CTM-Gene:
    1. Uses AttentionSynapse instead of U-Net synapse
    2. Attention weights are the GRN (no need to extract from sync matrix)
    3. GRN is inherently DIRECTIONAL (asymmetric)
    
    Mapping:
    - Neuron -> Gene
    - Internal Tick -> Pseudotime / Perturbation Propagation
    - Attention Weight A[i,j] -> Regulatory edge from gene i to gene j
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 n_synch_out,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='first-last',
                 n_random_pairing_self=0,
                 n_attention_heads=4,
                 d_attention_head=64,
                 ):
        # Initialize parent with minimal config
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=0,  # No standard attention
            n_synch_out=n_synch_out,
            n_synch_action=0,
            synapse_depth=1,  # Will be overridden
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            dropout_nlm=dropout_nlm,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
        )

        # Disable action synchronization
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None
        self.attention = None
        self.q_proj = None
        self.kv_proj = None

        # Replace U-Net synapse with Attention-based synapse
        self.synapses = AttentionSynapse(
            d_model=d_model,
            n_heads=n_attention_heads,
            dropout=dropout,
            d_head=d_attention_head
        )
        
        # Store attention weights across ticks for temporal GRN
        self.attention_history = []
        self.n_attention_heads = n_attention_heads
        self.d_attention_head = d_attention_head

    def forward(self, x, track=False):
        """
        Forward pass through the Attention-CTM.
        
        Args:
            x: Input gene expression (B, n_genes)
            track: Whether to track internal states and attention weights
            
        Returns:
            predictions: Gene expression at each tick (B, out_dims, iterations)
            certainties: Model certainty at each tick (B, 2, iterations)
            synchronisation_out: Final synchronization representation
            (if track=True) Also returns tracking arrays
        """
        B = x.size(0)
        device = x.device

        # Tracking initialization
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        attention_tracking = []  # NEW: track attention weights
        
        # Initialize recurrent state
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        # Prepare storage
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # Initialize synchronization decay
        r_out = torch.exp(-torch.clamp(self.decay_params_out, 0, 15)).unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )

        # Recurrent loop (each tick = perturbation propagation step)
        for stepi in range(self.iterations):
            # Concatenate input with current state
            pre_synapse_input = torch.cat((x, activated_state), dim=-1)

            # Apply Attention-based Synapse (the key difference!)
            state, attn_weights = self.synapses(pre_synapse_input, return_attention=True)
            
            # Update state trace
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # Apply Neuron-Level Models
            activated_state = self.trace_processor(state_trace)

            # Calculate synchronization for output
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )

            # Get predictions
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # Tracking
            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())

        if track:
            return (
                predictions, 
                certainties, 
                np.array(synch_out_tracking),
                np.array(pre_activations_tracking), 
                np.array(post_activations_tracking),
                np.array(attention_tracking)  # Shape: (T, B, n_heads, d_head, d_head)
            )
        return predictions, certainties, synchronisation_out

    def get_attention_grn(self, x=None, aggregate='mean'):
        """
        Extract the Gene Regulatory Network from attention weights.
        
        The attention weights capture: "To update gene j, attend to gene i"
        This is DIRECTIONAL: GRN[i,j] = regulatory influence of gene i on gene j
        
        Args:
            x: Optional input to run forward pass (if attention not cached)
            aggregate: How to aggregate across heads ('mean', 'max', 'first')
            
        Returns:
            grn: Asymmetric GRN matrix of shape (d_attention_head, d_attention_head)
        """
        if x is not None:
            # Run forward to get attention weights
            with torch.no_grad():
                self.forward(x, track=True)
        
        # Get attention from synapse module
        attn = self.synapses.last_attention_weights
        if attn is None:
            raise ValueError("No attention weights cached. Run forward() first or provide input x.")
        
        # attn shape: (B, n_heads, d_head, d_head)
        # Average over batch
        attn = attn.mean(dim=0)  # (n_heads, d_head, d_head)
        
        if aggregate == 'mean':
            grn = attn.mean(dim=0)  # Average over heads
        elif aggregate == 'max':
            grn = attn.max(dim=0)[0]  # Max over heads
        elif aggregate == 'first':
            grn = attn[0]  # First head only
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        return grn.cpu().numpy()

    def get_temporal_grn(self, attention_history):
        """
        Extract GRN that evolves over internal ticks.
        
        Args:
            attention_history: Array of shape (T, B, n_heads, d_head, d_head)
            
        Returns:
            temporal_grn: Array of shape (T, d_head, d_head)
        """
        # attention_history: (T, B, n_heads, d_head, d_head)
        # Average over batch and heads
        if isinstance(attention_history, list):
            attention_history = np.array(attention_history)
        return attention_history.mean(axis=(1, 2))  # (T, d_head, d_head)

    def forward_with_knockout(self, x, knockout_indices, track=False):
        """
        Run forward pass with specific genes knocked out.
        
        This is key for perturbation prediction: zero out a gene's input
        and observe how the attention-based regulation propagates effects.
        """
        x_perturbed = x.clone()
        if isinstance(knockout_indices, int):
            knockout_indices = [knockout_indices]
        for idx in knockout_indices:
            x_perturbed[:, idx] = 0.0
        return self.forward(x_perturbed, track=track)

