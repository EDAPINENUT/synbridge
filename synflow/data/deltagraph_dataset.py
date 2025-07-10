from typing import Callable
from .transform import get_transforms
import pytorch_lightning as pl
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .collate import (
    apply_collate,
    collate_1d_features,
    collate_2d_tokens,
    collate_padding_masks,
    collate_tokens,
)
from torch.utils.data import DataLoader
from typing import cast
from .deltagraph import create_data, DeltaGraphData, DeltaGraphBatch
import pickle

class DeltaGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = get_transforms(kwargs.pop("train_transforms", []))
        self.val_transforms = get_transforms(kwargs.pop("val_transforms", []))
        self.test_transforms = get_transforms(kwargs.pop("test_transforms", []))
        self.train_dataset_options = {**kwargs, "transforms": self.train_transforms}
        self.val_dataset_options = {**kwargs, "transforms": self.val_transforms}
        self.test_dataset_options = {**kwargs, "transforms": self.test_transforms}
        self.max_val_size = kwargs.pop("max_val_size", 2000)
        self.file_type = getattr(self.config.chem, "file_type", "pickle")
        self.reaction_data_dir = getattr(self.config.chem, "reaction_data_dir", None)
        self.reaction_file_tag = getattr(self.config.chem, "reaction_file_tag", None)

    def setup(self, stage: str | None = None) -> None:
        trainer = self.trainer

        if trainer is None and stage not in ["test", "val"]:
            raise RuntimeError("The trainer is missing.")

        if not os.path.exists(self.reaction_data_dir):
            raise FileNotFoundError(
                f"Reaction data not found: {self.reaction_data_dir}. "
            )
        
        file_names = {'train': os.path.join(self.reaction_data_dir, f'train_{self.reaction_file_tag}.{self.file_type}'),
                      'val': os.path.join(self.reaction_data_dir, f'val_{self.reaction_file_tag}.{self.file_type}'),
                      'test': os.path.join(self.reaction_data_dir, f'test_{self.reaction_file_tag}.{self.file_type}')}
        
        for split in ['train', 'val', 'test']:
            if not os.path.exists(file_names[split]):
                raise FileNotFoundError(
                    f"Reaction data not found: {file_names[split]}. "
                    "Please generate the reaction data before training."
                )
        
        if self.file_type == "pickle":
            data_class = DeltaGraphDataset

        self.train_dataset = data_class(file_names['train'], **self.train_dataset_options)
        self.val_dataset = data_class(file_names['val'], **self.val_dataset_options)
        self.test_dataset = data_class(file_names['test'], **self.test_dataset_options)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=Collater(),
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=Collater(),
        )
    
    def test_dataloader(self, indices=None):
        if indices is not None:
            from torch.utils.data import Subset
            test_subset = Subset(self.test_dataset, indices)
        else:
            test_subset = self.test_dataset
        return DataLoader(
            test_subset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=Collater(),
        )

class DeltaGraphDataset(Dataset):
    """
    src/tgt element, bond, charge, aroma, mask
    reactant, segment
    """
    def __init__(self, data_file: str, transforms: list[Callable] | None = None):
        self.data = pickle.load(open(data_file, "rb"))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        data = create_data(instance)
        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)
        return data
    
            
class Collater:
    def __init__(self, max_num_atoms: int = 192, max_num_mols: int = 16):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_num_mols = max_num_mols

        self.spec_atoms = {
            "elements": collate_tokens,
            "reactant_elements": collate_tokens,
            "reactant_element_types": collate_tokens,
            "reactant_bonds": collate_1d_features,
            "reactant_charges": collate_tokens,
            "reactant_masks": collate_padding_masks,
            "reactant_aromas": collate_tokens,
            "reactant_segments": collate_tokens,
            "reactant_flags": collate_tokens,
            "product_elements": collate_tokens,
            "product_element_types": collate_tokens,
            "product_bonds": collate_1d_features,
            "product_charges": collate_tokens,
            "product_masks": collate_padding_masks,
            "product_aromas": collate_tokens,
            "product_segments": collate_tokens,
            "padding_masks": collate_padding_masks,
        }

    def __call__(self, data_list: list[DeltaGraphData]) -> DeltaGraphBatch:
        data_list_t = cast(list[dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self.spec_atoms, data_list_t, max_size=self.max_num_atoms)
        }
        return cast(DeltaGraphBatch, batch)

if __name__ == "__main__":
    data_file = "/internfs/linhaitao/synflow/data/usptomit/train_data.pickle"
    dataset = DeltaGraphDataset(data_file)
    print(len(dataset))
    print(dataset[0])
