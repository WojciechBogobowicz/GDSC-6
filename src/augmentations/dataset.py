import random
from typing import Iterable, List, Optional

from datasets import Dataset

from .ast_aug import AstAug


class AugmentedDataset(Dataset):
    def __init__(self, *args, augs: Optional[List[AstAug]]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augs = [] if augs is None else augs

    @classmethod
    def from_dataset(cls, other, augs=None):
        aug = cls(other.data, augs=augs, info=other.info)
        return aug

    def __getitem__(self, index: int | slice | Iterable[int]):
        item = super().__getitem__(index)
        random_index = int(random.random() * len(self))

        aug = item
        for augmentation in self.augs:
            aug = augmentation(aug, linear_comb=super().__getitem__(random_index))
        return aug
