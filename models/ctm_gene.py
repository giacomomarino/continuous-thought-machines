import torch
import numpy as np
from models.ctm import ContinuousThoughtMachine

class ContinuousThoughtMachineGENE(ContinuousThoughtMachine):
    """
    Adaptation of CTM for Gene Regulatory Networks (GRNs) and Trajectory Inference.
    
    Mapping:
    - Neuron -> Gene
    - Internal Tick -> Pseudotime / Cell State
    - Synapse -> Regulatory Interaction
    - Synchronization -> Co-expression / Regulation
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 ):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=0, # No attention heads for this version
            n_synch_out=n_synch_out,
            n_synch_action=0, # No action synchronization needed
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type='none', # No backbone needed
            positional_embedding_type='none',
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            dropout_nlm=dropout_nlm,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
        )

        # --- Use a minimal CTM w/out input (action) synch ---
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None

        self.attention = None 
        self.q_proj = None 
        self.kv_proj = None 

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        
        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)

        # --- Prepare Storage for Outputs per Iteration ---
        # Predictions will be gene expression at each pseudotime step
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Initialise Recurrent Synch Values  ---
        r_out = torch.exp(-torch.clamp(self.decay_params_out, 0, 15)).unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        # --- Recurrent Loop (Pseudotime evolution) ---
        for stepi in range(self.iterations):

            # Input x is the initial state or context. 
            # We concatenate it with the current internal state to drive the dynamics.
            pre_synapse_input = torch.concatenate((x, activated_state), dim=-1)

            # --- Apply Synapses (Gene Regulatory Interactions) ---
            state = self.synapses(pre_synapse_input)
            # The 'state_trace' is the history of incoming pre-activations
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models (Gene-Specific Processing) ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronization (Co-expression Network) ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions (Gene Expression at this time step) ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, np.array(synch_out_tracking), np.array(pre_activations_tracking), np.array(post_activations_tracking)
        return predictions, certainties, synchronisation_out

    def get_gene_regulation_graph(self, step_index=-1):
        """
        Returns the synchronization matrix for a specific step, which serves as 
        the inferred Gene Regulatory Network.
        """
        # Note: This requires running with track=True to get the history.
        # Or we can implement a method to compute full pairwise sync if needed.
        pass

