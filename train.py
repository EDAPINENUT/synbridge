import os

import pytorch_lightning as pl
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import callbacks, loggers, strategies
import wandb
from synflow.data.deltagraph_dataset import DeltaGraphDataModule
from synflow.models.wrapper import SynFlowWrapper
from synflow.utils.misc import (
    get_experiment_name,
    get_experiment_version,
)
from synflow.utils.metrics import ACCRecCallback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

@hydra.main(version_base=None, config_path="./configs", config_name="difm_retro_usptomit")
def main(config: DictConfig):
    # Get command line overrides
    batch_size = config.train.batch_size
    devices = config.train.devices
    num_workers = config.train.num_workers
    num_nodes = config.train.num_nodes
    seed = config.train.seed
    log_dir = config.train.log_dir
    resume = config.train.resume
    wandb_project = config.train.wandb_project
    
    if batch_size % devices != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    batch_size_per_process = batch_size // devices

    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed)

    config_name = config.name
    exp_name = get_experiment_name(config_name, config.version)
    exp_ver = get_experiment_version()

    config_save_path = os.path.join(log_dir, exp_name, exp_ver, "config.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    OmegaConf.save(config, config_save_path)

    # Initialize wandb
    lightning_loggers = []
    if config.train.wandb_project is not None:
        lightning_loggers.append(
            loggers.WandbLogger(
                project=wandb_project,
                name=exp_name,
                version=exp_ver,
                log_model=True,
                save_code=True,
            ),
        )
    else:
        lightning_loggers.append(loggers.TensorBoardLogger(
            save_dir=log_dir,
            name=exp_name,
            version=exp_ver,
        ))

    # Dataloaders
    datamodule = DeltaGraphDataModule(
        config,
        batch_size=batch_size_per_process,
        num_workers=num_workers,
        **config.data,
    )

    # Model
    model = SynFlowWrapper(config)

    # Train
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategies.DDPStrategy(static_graph=True, find_unused_parameters=True),
        num_sanity_val_steps=config.train.num_sanity_val_steps,
        gradient_clip_val=config.train.max_grad_norm,
        log_every_n_steps=1,
        max_steps=config.train.max_iters,
        val_check_interval=config.train.val_freq,
        check_val_every_n_epoch=None,
        limit_val_batches=config.train.get('limit_val_batches', 8),
        callbacks=[
            ACCRecCallback(key_name='pred_info', metric_name='val_accuracy_smiles'),
            callbacks.ModelCheckpoint(
                dirpath=os.path.join(log_dir, exp_name, exp_ver), 
                save_last=True, 
                monitor="val_accuracy_smiles",
                mode="max", 
                save_top_k=3,
                filename="{epoch}-{step}-{val_accuracy_smiles:.4f}"
            ),
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=lightning_loggers,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume)

if __name__ == "__main__":
    main()
