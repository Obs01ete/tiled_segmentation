from typing import Union
import numpy as np


class RollingAccuracy:
    def __init__(self, length_in_samples:Union[int, None]=1000):
        self._length_in_samples = length_in_samples
        self._buffer = None

    def add_batch(self, gt: np.ndarray, pred: np.ndarray) -> None:
        assert gt.shape == pred.shape
        assert gt.dtype == np.int64
        assert pred.dtype == np.int64

        if self._buffer is None:
            self._buffer = np.zeros((0, gt.shape[1], 2), dtype=gt.dtype)

        if self._length_in_samples is not None:
            throw_away = self._buffer.shape[0] + gt.shape[0] - self._length_in_samples
            throw_away = throw_away if throw_away >= 0 else 0
            if throw_away > 0:
                self._buffer = self._buffer[throw_away:]

        stacked = np.stack((gt, pred), axis=2)
        self._buffer = np.concatenate((self._buffer, stacked), axis=0)
        pass

    def get_accuracies(self):
        plate_accuracy = float('nan')
        symbol_accuracy = float('nan')

        if len(self._buffer > 0):
            gt = self._buffer[:, :, 0]
            pred = self._buffer[:, :, 1]
            matches = gt == pred
            symbol_matches = np.count_nonzero(matches)
            symbol_count = gt.shape[0] * gt.shape[1]
            symbol_accuracy = symbol_matches / symbol_count
            plate_matches = np.count_nonzero(np.all(matches, axis=1))
            plate_count = gt.shape[0]
            plate_accuracy = plate_matches / plate_count

        return {
            "plate_accuracy": plate_accuracy,
            "symbol_accuracy": symbol_accuracy
        }
