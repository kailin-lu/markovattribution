from typing import Dict, Any
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import inv
from scipy.sparse import dok_matrix, identity


class MarkovAttribution(object):
    """
    Description
    """

    def __init__(self, data: pd.DataFrame, path_col: str = 'path', conv_col: str = 'conversion',
                 forward: bool = False, order: int = 1, idxmax: int = 10):
        self.data = data
        self.path_col = path_col
        self.conv_col = conv_col
        self.forward = forward
        self.order = order
        self.idxmax = idxmax
        self.channels = self._channels()
        self.states = self._unique_states()
        self.transition_matrix = dok_matrix(
            (len(self.states), len(self.states)), dtype=np.float32)
        self.fitted = False
        self.p_conversion = None
        self.removal_effects = {}

    def _channels(self) -> set:
        """
        Returns a list of unique models
        """
        return set([c for path in self.data[self.path_col] for c in path.split('>')])

    def _unique_states(self) -> Dict[str, int]:
        """
        Returns a list of unique states
        """
        states = {
            'NULL': 0,
            'CONVERSION': 1,
            'START': 2
        }
        for i, c in enumerate(product(self.channels, repeat=self.order), start=3):
            states['>'.join(c)] = i
        return states

    def _sequences_simple(self, paths: pd.Series, conversions: pd.Series):
        """
        Sequence generator for simple model splits
        """
        for path, conv in zip(paths, conversions):
            path = path.split('>')
            path.insert(0, 'START')
            if conv == 1:
                path.append('CONVERSION')
            else:
                path.append('NULL')
            yield path

    def _sequences_forward(self, paths, conversions):
        """
        Sequence generator for forward model splits
        """
        for path, conv in zip(paths, conversions):
            path = path.split('>')
            path = [p + 'f_{i}' if i < self.idxmax else p + f'_>{self.idxmax}'
                    for i, p in enumerate(path)]
            path.insert(0, 'START')
            if conv == 1:
                path.append('CONVERSION')
            else:
                path.append('NULL')
            yield path

    def _sequences(self, split_train_test: bool = False, test_size: int = 0.2) -> Any:
        """
        Return sequence generator
        """
        if split_train_test:
            train_seq, test_seq, train_conv, test_conv = train_test_split(
                self.data[self.path_col], self.data[self.conv_col], test_size)

            if self.forward:
                train_seq = self._sequences_forward(train_seq, train_conv)
                test_seq = self._sequences_forward(test_seq, test_conv)
            else:
                train_seq = self._sequences_simple(train_seq, train_conv)
                test_seq = self._sequences_simple(test_seq, test_conv)
            return train_seq, test_seq

        else:
            if self.forward:
                return self._sequences_forward(self.data[self.path_col], self.data[self.conv_col])
            else:
                return self._sequences_simple(self.data[self.path_col], self.data[self.conv_col])

    @staticmethod
    def _merge(seq: list) -> str:
        return '>'.join(seq)

    def _update_transition_matrix(self, seq):
        """
        Update transition matrix with one sequence
        """
        first_step = self._merge(seq[1:self.order + 1])
        self.transition_matrix[self.states['START'], self.states[first_step]] += 1

        last_step = self._merge(seq[-self.order - 1:-1])
        self.transition_matrix[self.states[last_step], self.states[seq[-1]]] += 1

        # Middle steps if they exist
        if len(seq) > self.order + 2:
            for i, e in enumerate(seq[1:-self.order - 1], 1):
                current = self._merge(seq[i:i + self.order])
                target = self._merge(seq[i + 1:i + self.order + 1])
                self.transition_matrix[self.states[current], self.states[target]] += 1

    def _build_transition_matrix(self, sequences):
        """
        Build transition matrix
        """
        n_states = len(self.states)
        print(f'Fitting {n_states} x {n_states} transition matrix')
        for seq in sequences:
            self._update_transition_matrix(seq)

        self.transition_matrix[0, 0] = 1  # Null
        self.transition_matrix[1, 1] = 1  # Conversion

        self.transition_matrix = normalize(self.transition_matrix, 'l1', axis=1)

    def conversion_rate(self, tm=None) -> float:
        """
        Calculate long run conversion probability
        """
        if not self.fitted:
            self.fit()
        if tm is None:
            tm = self.transition_matrix
        R = tm[2:, :2]
        Q = tm[2:, 2:]
        F = inv(identity(Q.shape[0]) - Q)
        FR = F.dot(R)
        return FR[0, 1]  # p_conversion

    def predict(self):
        raise NotImplementedError

    def removal_effect_channel(self, channel: str) -> float:
        """
        Calculate removal effect of specified channel
        """
        assert channel in self.channels, 'channel not present in data'

        if channel in self.removal_effects:
            return self.removal_effects[channel]

        tm = self.transition_matrix.copy()
        channel_idx = [v for k, v in self.states.items() if channel in k]
        tm[channel_idx, :] = 0  # Remove counts starting from channel
        tm[channel_idx, 0] = 1  # Reassign everything channel goes to null
        p_conversion_removed = self.conversion_rate(tm=tm)
        removal_effect = 1 - p_conversion_removed / self.p_conversion

        self.removal_effects[channel] = removal_effect
        return removal_effect

    def all_removal_effects(self):
        """
        Calculate removal effect of all channels
        """
        for channel in self.channels:
            if channel in self.removal_effects:
                print('Channel: ', channel, self.removal_effects[channel])
            else:
                rem_effect = self.removal_effect_channel(channel)
                self.removal_effects[channel] = rem_effect
                print('Channel: ', channel, rem_effect)

    def fit(self, split_train_test: bool = False):
        """
        Build transition matrix
        """
        sequences = self._sequences(split_train_test)
        if split_train_test:
            self._build_transition_matrix(sequences[0])  # Build with training sequences only
        else:
            self._build_transition_matrix(sequences)  # Use all sequences
        self.fitted = True
        print('Fitted. Calculating conversion probability')
        self.p_conversion = self.conversion_rate()
