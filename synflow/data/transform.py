import torch
from typing import cast
from typing import Callable
from omegaconf import OmegaConf
import torch.nn.functional as F
import random


class RandomPermute:
    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def __call__(self, data: dict) -> dict:
        length = data[self.attr_name].shape[0]
        index = []
        index = list(range(length))
        reverse = [(x, y) for x, y in enumerate(index)]
        reverse.sort(key = lambda x: x[1])
        reverse = [i[0] for i in reverse]
        reverse = torch.tensor(reverse).unsqueeze(1).expand(-1, data['reactant_bonds'].shape[1])
        for key in data:
            if 'bond' in key or 'edge' in key:
                data[key] = data[key][index]
                data[key] = torch.gather(reverse, 0, data[key])
            else:
                data[key] = data[key][index]
        return data

TRANSFORM_DICT = {
    "random_permute": RandomPermute,
}


def get_transforms(transforms: list[dict]) -> list[Callable]:
    transform_list = []
    for t in transforms:
        t_dict = OmegaConf.to_container(t, resolve=True)
        if t_dict["type"] not in TRANSFORM_DICT:
            raise ValueError(f"Transform type {t_dict['type']} not found")
        else:
            transform_list.append(TRANSFORM_DICT[t_dict.pop("type")](**t_dict))
    return transform_list
