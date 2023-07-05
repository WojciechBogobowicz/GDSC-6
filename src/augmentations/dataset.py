from copy import deepcopy
from typing import Dict, Iterable, List, Optional

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
        if isinstance(index, str):
            return super().__getitem__(index)
        item = super().__getitem__(index)

        ret = item
        for augmentation in self.augs:
            ret = augmentation(ret)
        return ret

    def get(self, index: int | slice | Iterable[int], apply_aug=True):
        if apply_aug:
            return self[index]
        return super().__getitem__(index)


class Upsampler:
    def __init__(self, dataset: Dataset, augmentations: List[AstAug]):
        self.dataset = dataset
        self.augmentations = augmentations
        self.labels_count = {}
        self.upsampled = {}

        for data in self.dataset:
            label = data['label']
            if label in self.labels_count.keys():
                self.labels_count[label] += 1
            else:
                self.labels_count[label] = 1

    def upsample(self):
        ...



def upsample(dataset: Dataset, augmentations: List[AstAug], expand_labels: Optional[Dict] = None,
             n: int = 1, label_column='label', combine_aug: bool = False) -> Dataset:
    """
    A function that adds additional samples to the provided dataset.

    Args:
        - dataset (Dataset): base dataset for which the data are to be upsampled.
        - augmentations (List[AstAug]): list of augmentations to use when creating a new sample.
        - expand_labels (Dict): a dict where keys are the labels of classes to be upsampled and values are
         the numbers how many times data of this label is to be added. If not provided all classes are
         upsampled. Defaults to None.
        - n (int): if `expand_labels` is not provided augmentation is applied to all data rows `n` times.
         Defaults to 1.
        - label_column

    Note:
        All parameters of given `AstAug` holds which means the probability of a new sample being
        different than the original sample is equal to the probability of augmentation. If you wish for your
        upsampled data to be always different than the original sample consider setting the `p` parameter in
        augmentation to `1.0`.
    """
    ret = Dataset(dataset.data, info=dataset.info)
    if not expand_labels:
        labels = set(dataset[label_column])
        expand_labels = {label: n for label in labels}
    for data in dataset:
        label = data[label_column]
        if label in expand_labels.keys():
            for _ in range(expand_labels[label]):
                to_add = deepcopy(data)
                if combine_aug:
                    for augmentation in augmentations:
                        to_add = augmentation(to_add)
                    ret.add_item(to_add)
    return ret
