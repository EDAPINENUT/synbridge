import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
from synflow.chem.utils import evaluate_single_smiles
from tqdm import tqdm
from synflow.chem.constants import MAX_BONDS, MAX_DIFF
import multiprocessing as mp
from functools import partial
import signal
import atexit
import os
import wandb
import torch.distributed as dist
from rdkit import Chem
import numpy as np

class AUROCCallback(Callback):
    def __init__(self, num_classes: int, key_name='cls'):
        super().__init__()
        self.num_classes = num_classes
        self.key_name = key_name
        self.val_aurocs = torch.nn.ModuleList([
            BinaryAUROC() for _ in range(num_classes)
        ])

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = pl_module.validation_step_outputs
        if isinstance(outputs["pred_" + self.key_name], list):
            pred_cls = torch.cat(outputs["pred_" + self.key_name], dim=0)
            target_cls = torch.cat(outputs["target_" + self.key_name], dim=0)
        else:
            pred_cls = outputs["pred_" + self.key_name]
            target_cls = outputs["target_" + self.key_name]
        
        # Convert to probabilities if needed (assuming outputs are logits)
        pred_probs = torch.softmax(pred_cls, dim=1)
        
        # Convert to one-hot encoding for multi-class AUROC
        target_one_hot = F.one_hot(target_cls, num_classes=self.num_classes)
        
        auroc_scores = []
        for i in range(self.num_classes):
            if target_one_hot[:, i].sum() > 0: # ignore classes that have no positive samples
                self.val_aurocs[i].update(pred_probs[:, i], target_one_hot[:, i])
                auroc = self.val_aurocs[i].compute()
                auroc_scores.append(auroc)
                self.val_aurocs[i].reset()
        if len(auroc_scores) > 0:
            mean_auroc = torch.stack(auroc_scores).mean()
            trainer.logger.log_metrics(
                {"val/auroc_" + self.key_name: mean_auroc}, 
                step=trainer.global_step
            )
    
    def on_validation_start(self, trainer, pl_module):
        for auroc in self.val_aurocs:
            auroc.reset()


class ACCRecCallback(Callback):
    '''
    This callback is used to record the reconstruction accuracy of molecules in reactants or products.
    '''
    def __init__(self, key_name='sample', metric_name='val_accuracy_smiles'):
        super().__init__()
        self.key_name = key_name
        self.metric_name = metric_name
        self.accuracy_smiles = []
        
    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = pl_module.validation_step_outputs
        samples = outputs[self.key_name]

        for id, sample in tqdm(samples.items(), desc="Validating SMILES...", total=len(samples)):
            try:
                acc_smiles = evaluate_single_smiles(sample)
                self.accuracy_smiles.append(acc_smiles)
            except Exception as e:
                print(f"Error processing sample {id}: {e}")
                continue
        mean_accuracy_smiles = torch.tensor(self.accuracy_smiles).nanmean()

        pl_module.log(self.metric_name, mean_accuracy_smiles.to(pl_module.device), sync_dist=True)
        print(f"Epoch {trainer.current_epoch}, {self.metric_name}: {mean_accuracy_smiles.item()}")

            
    def on_validation_start(self, trainer, pl_module):
        self.accuracy_smiles = []

