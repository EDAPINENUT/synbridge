import torch
from omegaconf import OmegaConf
from synflow.data.deltagraph import DeltaGraphBatch
from .vaedifm_uniform import VAEDIFMUniform
def get_vae_diffusion(config) -> "Diffusion":
    """
    Get diffusion model based on config.
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        Diffusion: Initialized diffusion model instance
    """
    if config.name == "vaedifm_uniform":
        return VAEDIFMUniform(config)
    else:
        raise NotImplementedError(f"unknown diffusion: {config.name}")

class Diffusion:
    def __init__(self, config: OmegaConf):
        self.config = config

    def get_loss(self, batch: DeltaGraphBatch, t: torch.Tensor, mode: str = 'retro') -> torch.Tensor:
        pass

    def sample(self, batch: DeltaGraphBatch, num_samples: int, sample_steps: int) -> torch.Tensor:
        pass

    def prepare_ground_truth(self, batch: DeltaGraphBatch, sample_num: int) -> torch.Tensor:
        pass