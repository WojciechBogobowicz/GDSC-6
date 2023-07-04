import random

import numpy as np

from .ast_aug import AstAug


class AudioAug(AstAug):
    pass


class NoiseAug(AudioAug):
    def __init__(self, p: float, noise_ratio: float = 0.01):
        self.noise_ratio = noise_ratio
        self.proba = p
        assert 0.0 <= self.proba <= 1.0, 'Augmentation probability should be between 0.0 and 1.0.'

    def __call__(self, data, **kwargs):
        if random.random() > self.proba:
            return data
        array = data['audio']['array']
        noise = np.random.randn(len(array))
        augmented_array: np.ndarray = array + noise * self.noise_ratio
        augmented_array = augmented_array.astype(type(array[0]))
        data['audio']['array'] = augmented_array
        return data


class ShiftAug(AudioAug):
    def __init__(self, p: float, len_percent: int, direction: str):
        self.proba = p
        self.percent = len_percent
        self.direction = direction
        assert 0.0 < self.percent < 1.0, 'Audio len percentage should be between 0.0 and 1.0.'
        assert 0.0 <= self.proba <= 1.0, 'Augmentation probability should be between 0.0 and 1.0.'
        assert self.direction in ['right', 'left', 'both'], 'Shift direction should have one of those values: \'right\', \'left\', \'both\'.'

    def __call__(self, data, **kwargs):
        if  random.random() > self.proba:
            return data
        array = data['audio']['array']
        max_shift = int(len(array) * self.percent)
        shift = random.randint(0, max_shift)

        if self.direction == 'left':
            shift = -shift
        elif self.direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(array, shift)
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        data['audio']['array'] = augmented_data
        return data


class DoubleAug(AudioAug):
    def __init__(self, p: float):
        self.proba = p
        assert 0.0 <= self.proba <= 1.0, 'Augmentation probability should be between 0.0 and 1.0.'

    def __call__(self, data, **kwargs):
        if random.random() > self.proba:
            return data
        arr = data['audio']['array']
        data['audio']['array'] = np.append(arr, arr)
        return data
