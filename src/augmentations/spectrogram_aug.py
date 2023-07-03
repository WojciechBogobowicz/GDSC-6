import random

import numpy as np
from datasets import Dataset

from .ast_aug import AstAug


class SpectrogramAug(AstAug):
    pass


class CutoutAug(SpectrogramAug):
    def __init__(self, p: float, freq_masking_percentage: float = 0.15, time_masking_percentage: float = 0.3):
        self.proba = p
        self.freq_maskig_percentage = freq_masking_percentage
        self.time_masking_percentage = time_masking_percentage
        assert 0.0 <= self.proba <= 1.0, 'Augmentation probability should be between 0.0 and 1.0.'
        assert 0.0 <= self.freq_maskig_percentage <= 1.0, 'Frequency masking percentage should be between 0.0 and 1.0.'
        assert 0.0 <= self.time_masking_percentage <= 1.0, 'Time masking percentage should be between 0.0 and 1.0.'

    def __call__(self, data, **kwargs):
        if random.random() > self.proba:
            return data
        spectrogram = np.array(data['input_values'])
        _, all_frames_num, all_freqs_num = spectrogram.shape

        spectrogram = self.__mask_frequency(spectrogram)
        spectrogram = self.__mask_time(spectrogram)

        data['input_values'] = spectrogram
        return data

        # freq_percentage = random.uniform(0.0, self.freq_maskig_percentage)
        # num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        # f0 = int(np.random.uniform(low=0.0, high=(all_freqs_num - num_freqs_to_mask)))
        # spectrogram[:, f0:(f0 + num_freqs_to_mask)] = 0

        # time_percentage = random.uniform(0.0, self.time_masking_percentage)
        # num_frames_to_mask = int(time_percentage * all_frames_num)
        # t0 = int(np.random.uniform(low=0.0, high=(all_frames_num - num_frames_to_mask)))
        # spectrogram[t0:(t0 + num_frames_to_mask), :] = 0


    def __mask_frequency(self, spectrogram: np.ndarray) -> np.ndarray:
        _, _, all_freqs_num = spectrogram.shape
        freq_percentage = random.uniform(0.0, self.freq_maskig_percentage)
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = int(np.random.uniform(low=0.0, high=(all_freqs_num - num_freqs_to_mask)))
        spectrogram[:, f0:(f0 + num_freqs_to_mask)] = 0
        return spectrogram

    def __mask_time(self, spectrogram: np.ndarray) -> np.ndarray:
        _, all_frames_num, _ = spectrogram.shape
        time_percentage = random.uniform(0.0, self.time_masking_percentage)
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = int(np.random.uniform(low=0.0, high=(all_frames_num - num_frames_to_mask)))
        spectrogram[t0:(t0 + num_frames_to_mask), :] = 0
        return spectrogram


class MixupAug(SpectrogramAug):
    def __init__(self, p: float, alpha: float = 1.0):
        self.proba = p
        self.alpha = alpha
        # self.dataset = dataset
        # labels_count = self.dataset.features['label'].num_classes
        # print('#' * 100, labels_count, '#' * 100)
        assert 0.0 <= self.proba <= 1.0, 'Augmentation probability should be between 0.0 and 1.0.'

    def __call__(self, data, linear_comb, **kwargs):
        if random.random() > self.proba:
            return data
        one_hot = np.zeros((1, 66), dtype=np.float64)
        one_hot[0, data['label']] = 1.0
        spectrogram = np.array(data['input_values'])

        one_hot_other = np.zeros((1, 66), dtype=np.float64)
        one_hot_other[0, linear_comb['label']] = 1.0
        spectrogram_other = np.array(linear_comb['input_values'])

        comb_fact = np.random.beta(self.alpha, self.alpha)

        ret = data.copy()
        ret['input_values'] = comb_fact * spectrogram + (1 - comb_fact) * spectrogram_other
        ret['label'] = comb_fact * one_hot + (1 - comb_fact) * one_hot_other
        return ret
