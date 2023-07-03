import random

import numpy as np

from .ast_aug import AstAug


class SpectrogramAug(AstAug):
    pass


class CutoutAug(SpectrogramAug):
    def __init__(self, p: float, freq_masking_percentage: float = 0.15, time_masking_percentage: float = 0.3):
        self.proba = p
        self.freq_maskig_percentage = freq_masking_percentage
        self.time_masking_percentage = time_masking_percentage

    def __call__(self, data):
        if random.random() > self.proba:
            return data
        spectrogram = np.array(data['input_values'])
        _, all_frames_num, all_freqs_num = spectrogram.shape

        freq_percentage = random.uniform(0.0, self.freq_maskig_percentage)
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = int(np.random.uniform(low=0.0, high=(all_freqs_num - num_freqs_to_mask)))
        spectrogram[:, f0:(f0 + num_freqs_to_mask)] = 0

        time_percentage = random.uniform(0.0, self.time_masking_percentage)
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = int(np.random.uniform(low=0.0, high=(all_frames_num - num_frames_to_mask)))
        spectrogram[t0:(t0 + num_frames_to_mask), :] = 0

        data['input_values'] = spectrogram
        return data


class MixupAug(SpectrogramAug):
    def __init__(self, p, alpha):
        self.proba = p
        self.alpha = alpha

    def __call__(self, data):
        return super().__call__(data)
