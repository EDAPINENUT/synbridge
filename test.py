import os
import click
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm
from synflow.data.deltagraph_dataset import DeltaGraphDataModule
from synflow.models.wrapper import SynFlowWrapper
from synflow.chem.utils import result2mol, get_canonical_smiles
from multiprocessing import Pool, cpu_count

def evaluate_single_sample(args):
    sample, mode, topk = args
    sample_num = len(sample['src_masks'])
    acc_smiles = []
    eval_num = min(topk, sample_num)
    for i in range(eval_num):
        pred_bonds = sample['pred_bonds'][i]
        pred_aromas = sample['pred_aromas'][i]
        pred_charges = sample['pred_charges'][i]
        true_bonds = sample['true_bonds'][i]
        true_aromas = sample['true_aromas'][i]
        true_charges = sample['true_charges'][i]
        src_flags = sample['src_flags'][i]
        src_masks = sample['src_masks'][i]
        tgt_masks = sample['tgt_masks'][i]
        if 'elements' in sample:
            pred_elements = sample['elements'][i]
            true_elements = sample['elements'][i]
        else:
            pred_elements = sample['pred_elements'][i]
            true_elements = sample['true_elements'][i]
        _, pred_s, pred_valid = result2mol((pred_elements, src_masks, 
                                            pred_bonds, pred_aromas,
                                            pred_charges, src_flags))
        _, tgt_s, tgt_valid = result2mol((true_elements, tgt_masks, 
                                          true_bonds, true_aromas,
                                          true_charges, src_flags))
        if mode == 'forward':
            if tgt_s is None:
                return None
            elif pred_s is None:
                acc_cur = np.nan
                acc_smiles.append(acc_cur)
            else:
                tgt_split = tgt_s.split('.')
                pred_split = pred_s.split('.')
                for item in tgt_split:
                    if item not in pred_split:
                        acc_cur = 0.0
                        acc_smiles.append(acc_cur)
                        break
                else:
                    acc_cur = 1.0
                    acc_smiles.append(acc_cur)
            # if acc_cur == np.nan or acc_cur == 0.0:
            #     src_flags = sample['input_flags'][i]
            #     src_masks = sample['input_masks'][i]
            #     src_bonds = sample['input_bonds'][i]
            #     src_aromas = sample['input_aromas'][i]
            #     src_charges = sample['input_charges'][i]
            #     src_elements = sample['input_elements'][i]
            #     _, src_s, src_valid = result2mol((src_elements, src_masks, 
            #                                       src_bonds, src_aromas,
            #                                       src_charges, src_flags))
            #     with open('fail_analysis.txt', 'a') as f:
            #         f.write(f"{src_s}>>{tgt_s}\t {src_s}>>{pred_s} \t {acc_cur}\n")
        elif mode == 'retro':
            if tgt_s is None:
                return None
            elif pred_s is None:
                acc_smiles.append(0)
            else:
                tgt_s = get_canonical_smiles(tgt_s)
                pred_s = get_canonical_smiles(pred_s)
                if tgt_s is None:
                    return None
                if pred_s is None:
                    acc_smiles.append(0)
                    continue
                tgt_split = tgt_s.split('.')
                pred_split = pred_s.split('.')
                tgt_split = sorted(tgt_split)
                pred_split = sorted(pred_split)
                if tgt_split == pred_split:
                    acc_smiles.append(1.0)
                else:
                    acc_smiles.append(0.0)
    del sample, pred_bonds, pred_aromas, pred_charges, true_bonds, true_aromas, true_charges, src_flags, src_masks, tgt_masks, pred_elements, true_elements
    return acc_smiles



@click.command()
@click.option("--ckpt_path", type=click.Path(exists=True), help="Path to checkpoint file", 
              default="/logs/vaedifm_usptomit_uniform/2025_06_06__09_01_19-vaedifm_usptomit_uniform/2025_06_06__09_01_19/epoch=338-step=135000-val_accuracy_smiles=0.8535.ckpt")
@click.option("--config_path", type=click.Path(exists=True), 
              default="/logs/vaedifm_usptomit_uniform/2025_06_06__09_01_19-vaedifm_usptomit_uniform/2025_06_06__09_01_19/config.yaml")
@click.option("--batch_size", type=int, default=32) 
@click.option("--num_workers", type=int, default=4)
@click.option("--device", type=str, default="cuda")
@click.option("--sample_num", type=int, default=5)
@click.option("--sample_steps", type=int, default=100)
@click.option("--save_samples", is_flag=True, default=False)
@click.option("--num_workers_eval", type=int, default=cpu_count()//8) # cpu_count()//4
@click.option("--eval_mode", type=str, default='forward')
def main(
    ckpt_path: str,
    config_path: str,
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
    sample_num: int,
    sample_steps: int,
    save_dir: str | None = None,
    save_samples: bool = True,
    num_workers_eval: int = 1,
    eval_mode: str | None = None,
):
    # Load config
    config = OmegaConf.load(config_path)
    
    # Initialize model from checkpoint
    model = SynFlowWrapper.load_from_checkpoint(
        ckpt_path,
        config=config,
        strict=True
    )
    model = model.to(device)
    model.eval()

    # Initialize data module
    datamodule = DeltaGraphDataModule(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
        **config.data,
    )
    
    # Create output directory
    save_dir = (os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "samples") 
                if save_dir is None else save_dir)
    os.makedirs(save_dir, exist_ok=True)
    eval_mode = config.train.mode if eval_mode is None else eval_mode

    # Run sampling on test set
    datamodule.setup('test')
    # test_loader = datamodule.test_dataloader(indices=list(range(7000)))
    test_loader = datamodule.test_dataloader()
    print('Test set size:', len(test_loader.dataset))
    id = 0
    sample_all = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Sampling"):
            # Move batch to GPU if needed
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            cur_batch_size = batch['elements'].size(0)
            id_list = [i + id for i in range(cur_batch_size)]
            batch['id'] = id_list
            # Generate samples
            samples = model.sample(
                batch, sample_num, sample_steps=sample_steps, mode=eval_mode
            )
            id += cur_batch_size
            if save_samples:
                for key, value in samples.items():
                    torch.save(value, os.path.join(save_dir, f"{int(key):05d}.pt"))
            sample_all.update(samples)
    print(f"Generated {id} * {sample_num} = {id * sample_num} samples")


    # Evaluate samples
    k_list = [1, 3, 5]
    acc_smiles = {f'top{k}': [] for k in k_list}
    eval_tasks = [
        (sample, eval_mode, max(k_list)) for sample in list(sample_all.values())
    ]
    
    if num_workers_eval > 1:
        print(f"Starting parallel evaluation using {num_workers_eval} CPUs...")
        with Pool(processes=num_workers_eval) as pool:
            acc_all = list(tqdm(
                pool.imap(
                    evaluate_single_sample, 
                    eval_tasks
                ),
                total=len(sample_all),
                    desc="Evaluating samples"
                ))  
    else:
        acc_all = [evaluate_single_sample(task) 
                   for task in tqdm(eval_tasks, desc="Evaluating samples")]
    acc_all = [item for item in acc_all if item is not None]
    acc_all = np.array(acc_all)

    # Process results
    for k in k_list:
        acc_k = acc_all[:, :k]
        acc_k = acc_k[~np.all(np.isnan(acc_k), axis=1)]
        acc_k = np.any(acc_k, axis=1).astype(int)
        acc_smiles[f'top{k}'] = acc_k
        print(f"Top {k} accuracy: {np.mean(acc_smiles[f'top{k}'])}")
    # Write accuracy results to txt
    with open(os.path.join(save_dir, f"test_results_{eval_mode}.txt"), 'w') as f:
        for k in k_list:
            f.write(f"Top {k} accuracy: {np.mean(acc_smiles[f'top{k}'])}\n")

if __name__ == "__main__":
    main()