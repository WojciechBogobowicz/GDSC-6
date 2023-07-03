from abc import ABC, abstractmethod


class AstAug(ABC):

    @abstractmethod
    def __call__(self, data):
        pass
