from abc import ABC, abstractmethod

from datasets import Dataset


class AstAug(ABC):

    @abstractmethod
    def __call__(self, data, **kwargs):
        pass

    def __repr__(self) -> str:
        name = f'{self.__class__.__name__}('
        params = ', '.join([f'{key}={value}' for key, value in self.__dict__.items() if not key.startswith('_')])
        return name + params + ')'

    def attach_dataset(self, dataset: Dataset):
        pass
