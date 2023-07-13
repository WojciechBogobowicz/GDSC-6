import collections
import random
from typing import Iterable, List, Optional

import pandas as pd
from datasets import Dataset, concatenate_datasets

from .ast_aug import AstAug


class AugmentedDataset(Dataset):
    def __init__(self, *args, augs: Optional[List[AstAug]]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augs = [] if augs is None else augs

    @classmethod
    def from_dataset(cls, other, augs=None):
        aug = cls(other.data, augs=augs, info=other.info)
        return aug

    @classmethod
    def from_iter(cls, dataset_iter, augs=None):
        ret = cls.from_dict({})
        for elem in dataset_iter:
            ret = ret.add_item(elem)
        print(len(ret))
        return ret

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


def upsample(dataset: Dataset, threshold):
    upsampled_dataset = dataset
    labels_count = collections.Counter(dataset['label'])

    upsampled_datasets = [upsampled_dataset]
    for label, count in labels_count.items():
        if count >= threshold:
            continue

        num_samples_to_add = threshold - count
        samples_to_duplicate = dataset.filter(lambda example: example['label'] == label)
        duplicate_indices = random.choices(list(range(len(samples_to_duplicate))), k=num_samples_to_add)
        # duplicated_samples = samples_to_duplicate.shuffle().select(range(num_samples_to_add))
        duplicated_samples = samples_to_duplicate.select(duplicate_indices)
        upsampled_datasets.append(duplicated_samples)
    # Shuffle the upsampled dataset
    upsampled_dataset = concatenate_datasets(upsampled_datasets)
    upsampled_dataset = upsampled_dataset.shuffle()
    return upsampled_dataset


def __get_upsampled_indices(dataset: Dataset, upsample_labels: dict):
    absent_labels = set(dataset['label']) - set(upsample_labels.keys())
    for absent_label in absent_labels:
        upsample_labels.update({absent_label: 0.0})
    pd_labels = pd.DataFrame({'label': dataset['label']}).reset_index()
    grouped = pd_labels.groupby(['label'])
    grouped_agg = grouped.count().reset_index()
    grouped_agg['aug count'] = grouped_agg.apply(lambda row: int(upsample_labels[row["label"]] * row['index']), axis=1)
    grouped_agg.set_index("label", inplace=True)

    augmented_groups = []
    for group_name, group in grouped:
        count = grouped_agg.loc[group_name, 'aug count']
        extra_labels = group.sample(n=count, replace=True)
        augmented_groups.append(pd.concat([group, extra_labels]))

    augmented_indices = pd.concat(augmented_groups)
    return list(augmented_indices['index'])
