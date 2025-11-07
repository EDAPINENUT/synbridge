from functools import partial
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
import math
from synflow.data.deltagraph import DeltaGraphBatch
from synflow.utils.train import get_optimizer, get_scheduler, sum_weighted_losses, is_loss_nan_check
from synflow.models.misc import expand_batch, shape_back
from synflow.models.diffusion import get_vae_diffusion, Diffusion
from synflow.models.diffusion.schedulers import get_schedule_sampler
from typing import Any, Dict, List, Optional, Union
from synflow.models.diffusion.utils import prepare_ground_truth
from synflow.models.roundtrip_sample import prepare_roundtrip_sample, roundtrip_sample
import random
import copy
import gc

def get_arccos_schedule(step, end_step, start_step, min_value, max_value):    
    """
    Calculate a value using arccos schedule between min_value and max_value.
    
    Args:
        step: Current step number
        end_step: Step number when schedule should reach max_value
        start_step: Step number when schedule should start increasing from min_value
        min_value: Minimum value of the schedule
        max_value: Maximum value of the schedule
        
    Returns:
        Scheduled value between min_value and max_value
    """
    if step < start_step:
        return min_value
            
    if step >= end_step:
        return max_value
        
    normalized_step = (step - start_step) / (end_step - start_step)
    arccos_value = 1 - (2 * math.acos(normalized_step) / math.pi)
    
    return min_value + (max_value - min_value) * arccos_value


class SynFlowWrapper(pl.LightningModule):
    """
    Wrapper class for the SynFlow model that handles training and inference.
    
    This class implements the training, validation and sampling logic for the SynFlow model.
    It inherits from PyTorch Lightning's LightningModule for structured training.
    """
    def __init__(
        self,
        config: OmegaConf,  # Configuration object containing model parameters
        args: Optional[Dict] = None  # Optional additional arguments
    ) -> None:
        """
        Initialize the SynFlow model wrapper.

        Args:
            config: Configuration object containing model parameters including model architecture,
                   training settings, and hyperparameters
            args: Optional dictionary of additional arguments for model configuration
        """
        super().__init__()
        self.save_hyperparameters(
            {
                "config": OmegaConf.to_container(config),
                "args": args or {},
            }
        )
        self.diffusion: Diffusion = get_vae_diffusion(config.model)
        self.schedule_sampler = get_schedule_sampler(config.schedule_sampler, self.diffusion)
        self.validation_step_outputs = None

        self.loss_weights = self.config.train.loss_weights


    @property
    def config(self) -> OmegaConf:
        """
        Get the model configuration.
        
        Returns:
            OmegaConf object containing model configuration
        """
        return OmegaConf.create(self.hparams["config"])

    @property
    def args(self) -> OmegaConf:
        """
        Get additional arguments.
        
        Returns:
            OmegaConf object containing additional arguments
        """
        return OmegaConf.create(self.hparams.get("args", {}))

    def setup(self, stage: str) -> None:
        """
        Setup the model for training or validation.
        
        Args:
            stage: Current stage ('fit' or 'test')
        """
        super().setup(stage)

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Either an optimizer or a dictionary containing optimizer and scheduler configuration
        """
        optimizer = get_optimizer(self.config.train.optimizer, self.diffusion)
        if "scheduler" in self.config.train:
            scheduler = get_scheduler(self.config.train.scheduler, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        return optimizer

    def training_step(
        self, 
        batch: DeltaGraphBatch,  # Input batch of graph data
        batch_idx: int  # Index of current batch
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch: Input batch of graph data containing molecule information
            batch_idx: Index of current batch

        Returns:
            Loss value for the training step
        """
        B, _ = batch['elements'].size()
        t, _ = self.schedule_sampler.sample(B, self.device)
        if self.config.train.mode == 'mix':
            mode = random.choice(['retro', 'forward'])
        else:
            mode = self.config.train.mode
        try:
            loss_dict = self.diffusion.get_loss(batch, t=t, mode=mode)
            loss_weights = copy.deepcopy(self.loss_weights)

            loss_sum = sum_weighted_losses(loss_dict, loss_weights)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.print(f"Skip iteration with OOM: {self.global_step} steps")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # return a dummy zero loss to safely skip this batch
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                raise

        if is_loss_nan_check(loss_sum):
            self.print(f"Skip iteration with NaN loss: {self.global_step} steps")
            loss_sum = torch.tensor(0.0, device=loss_sum.device, requires_grad=True)
            return loss_sum

        self.log("train/loss", loss_sum, on_step=True, prog_bar=True, logger=True, batch_size=B)
        self.log_dict(
            {f"train/loss_{k}": v for k, v in loss_dict.items()},
            on_step=True, logger=True, batch_size=B
        )
        return loss_sum
    

    def validation_step(
        self, 
        batch: DeltaGraphBatch,  # Input batch of graph data
        batch_idx: int  # Index of current batch
    ) -> Dict[str, Any]:
        """
        Perform a single validation step.

        Args:
            batch: Input batch of graph data containing molecule information
            batch_idx: Index of current batch

        Returns:
            Dictionary containing validation metrics including loss and prediction info
        """
        B, _ = batch['elements'].size()
        t, _ = self.schedule_sampler.sample(B, self.device)
        if self.config.train.mode == 'mix':
            mode = 'forward'
            loss_dict_forward = self.diffusion.get_loss(batch, t=t, mode=mode)
            pred_info_forward = self.sample(
                batch, 
                num_samples=self.config.train.sample_num, 
                sample_steps=self.config.train.sample_steps,
                mode=mode,
                batched_out=True
            )
            loss_sum_forward = sum_weighted_losses(loss_dict_forward, self.loss_weights)

            mode = 'retro'
            loss_dict_retro = self.diffusion.get_loss(batch, t=t, mode=mode)
            pred_info_retro = self.sample(
                batch, 
                num_samples=self.config.train.sample_num, 
                sample_steps=self.config.train.sample_steps,
                mode=mode,
                batched_out=True
            )
            loss_sum_retro = sum_weighted_losses(loss_dict_retro, self.loss_weights)
            
            loss_sum = loss_sum_forward + loss_sum_retro
            loss_dict = {**loss_dict_forward, **loss_dict_retro}
            pred_info = {**pred_info_forward, **pred_info_retro}

        else:
            mode = self.config.train.mode
            loss_dict = self.diffusion.get_loss(batch, t=t, mode=mode)

            pred_info = self.sample(
                batch, 
                num_samples=self.config.train.sample_num, 
                sample_steps=self.config.train.sample_steps,
                mode=mode,
                batched_out=True
            )

            loss_sum = sum_weighted_losses(loss_dict, self.loss_weights)

        self.log(
            "val/loss", loss_sum, 
            on_step=False, prog_bar=True, logger=True, sync_dist=True, batch_size=B
        )
        self.log_dict(
            {f"val/loss_{k}": v for k, v in loss_dict.items()}, 
            on_step=False, logger=True, sync_dist=True, batch_size=B
        )
        if 'loss_kl' in loss_dict:
            return {
                "loss": loss_sum,
                "pred_info": pred_info,
                "loss_kl": loss_dict.get('loss_kl', torch.tensor(0.0)),
            }
        else:
            return {
                "loss": loss_sum,
                "pred_info": pred_info
            }
    
    def on_validation_epoch_start(self) -> None:
        """
        Initialize validation metrics at the start of validation epoch.
        Sets up dictionaries to store prediction info and KL losses.
        """
        self.validation_step_outputs = {
            'pred_info': {},
            'kl_losses': [], 
        }
    
    def on_validation_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],  # Outputs from validation step
        batch: DeltaGraphBatch,  # Input batch
        batch_idx: int  # Batch index
    ) -> None:
        """
        Process validation batch results.

        Args:
            outputs: Outputs from validation step containing predictions and losses
            batch: Input batch of data
            batch_idx: Index of current batch
        """
        if 'pred_info' in outputs:
            self.validation_step_outputs['pred_info'].update(outputs['pred_info'])
        
        if 'loss_kl' in outputs:
            self.validation_step_outputs['kl_losses'].append(outputs['loss_kl'])

        gc.collect()
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate validation results at the end of validation epoch.
        Adjusts KL loss weights based on mean KL loss value.
        """
        if len(self.validation_step_outputs['kl_losses']) > 0:
            kl_losses = torch.tensor(self.validation_step_outputs['kl_losses'])
            kl_loss_mean = kl_losses.mean().item()
                        
            if kl_loss_mean < 0.5:
                self.loss_weights.kl *= 0.8
            elif kl_loss_mean > 1.0:
                self.loss_weights.kl *= 1.6            

    @torch.no_grad()
    def sample(
        self,
        batch: DeltaGraphBatch,  # Input batch of graph data
        num_samples: int = 1000,  # Number of samples to generate
        max_sample_chunk: int = 100,  # TODO: add this to config as max_sample_chunk
        sample_steps: int = 20,
        batched_out: bool = False,
        mode: str = 'retro',
        roundtrip: bool = False
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate samples from the model.

        Args:
            batch: Input batch of graph data containing molecule information
            num_samples: Number of samples to generate per input
            max_sample_chunk: Maximum batch size for sampling
            sample_steps: Number of steps to sample in diffusion sampler
            batched_out: Whether to return batched output
            mode: Mode to sample ('retro' or 'forward')
        Returns:
            Dictionary containing generated samples with predictions for bonds, aromas,
            charges, elements and corresponding ground truth values
        """
        B = batch['padding_masks'].size(0)
        batch_samples = expand_batch(batch, num_samples)
        preds = self.diffusion.sample(batch_samples, sample_steps=sample_steps, mode=mode)
        tgts, srcs = prepare_ground_truth(batch_samples, mode=mode, return_src=True)
        
        if not batched_out:
            preds_batched = {k: shape_back(v, B).cpu() for k, v in preds.items()}
            tgts_batched = {k: shape_back(v, B).cpu() for k, v in tgts.items()}
            srcs_batched = {k: shape_back(v, B).cpu() for k, v in srcs.items()}
        else:
            preds_batched = {k: v.cpu() for k, v in preds.items()}
            tgts_batched = {k: v.cpu() for k, v in tgts.items()}
            srcs_batched = {k: v.cpu() for k, v in srcs.items()}
        if roundtrip:
            return prepare_roundtrip_sample(preds_batched, tgts_batched, srcs_batched)
        
        output_samples = {}
        if 'id' in batch and batched_out:
            list_id, select_dim = batch_samples['id'], 0
        elif 'id' in batch and not batched_out:
            list_id, select_dim = batch['id'], 1
        elif 'id' not in batch and batched_out:
            list_id, select_dim = [i for i in range(B*num_samples)], 0
        else:
            list_id, select_dim = [i for i in range(B)], 1

        for i, id in enumerate(list_id):
            select = partial(torch.select, dim=select_dim, index=i)
            output_samples[f"{id}"] = {
                "pred_bonds": select(preds_batched['bonds']),
                "pred_aromas": select(preds_batched['aromas']),
                "pred_charges": select(preds_batched['charges']),
                "pred_elements": select(preds_batched['elements']),
                "pred_element_types": select(preds_batched['element_types']),
                "src_masks": select(preds_batched['masks']),
                "src_flags": select(preds_batched['flags']),
                "tgt_masks": select(tgts_batched['masks']),
                "true_bonds": select(tgts_batched['bonds']),
                "true_aromas": select(tgts_batched['aromas']),    
                "true_charges": select(tgts_batched['charges']),
                "true_elements": select(tgts_batched['elements']),
                "true_element_types": select(tgts_batched['element_types']),
                "input_bonds": select(srcs_batched['bonds']),
                "input_aromas": select(srcs_batched['aromas']),
                "input_charges": select(srcs_batched['charges']),
                "input_elements": select(srcs_batched['elements']),
                "input_element_types": select(srcs_batched['element_types']),
                "input_masks": select(srcs_batched['masks']),
                "input_flags": select(srcs_batched['flags']),
            }

        return output_samples

    def roundtrip_sample(self, batch: DeltaGraphBatch, num_samples: int = 1000, sample_steps: int = 20) -> Dict[str, Dict[str, torch.Tensor]]:
        return roundtrip_sample(self, batch, num_samples, sample_steps)