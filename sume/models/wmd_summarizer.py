# -*- coding: utf-8 -*-


"""Word Mover's Distance summarization."""


import collections
import logging
import multiprocessing
import time
from typing import (Callable, Hashable, Iterable, List, Mapping, Sequence, Set, TypeVar,
                    Union)

import fwmd
import numpy

from ..base import Reader

logger = logging.getLogger(__name__)


class WMDSummarizer(Reader):
    """Word Mover's Distance summarization model."""

    def __init__(self, embeddings: Callable[[str], numpy.ndarray],
                 input_directory: str, file_extension: str = '',
                 n_workers: int = multiprocessing.cpu_count()) -> None:
        """Construct a WMD summarizer.

        Args:
            model (FastText model): the model to use to compute word
              embeddings (with the .bin extension).
            args: args to pass on to the sume.base.Reader constructor.
            kwargs: kwargs to pass on to the sume.base.Reader constructor.
        """
        super().__init__(input_directory, file_extension=file_extension)

        logger.debug('initializing WMD summarizer')
        self.embeddings = embeddings
        self.n_workers = n_workers

        logger.debug('embedding words')
        self._embed()

        logger.debug('computing BOWs')
        self._compute_BOWs()

        logger.debug('computing document nBOW')
        self._doc_nBOW = self._compute_nBOW(range(len(self.sentences)))

    def _embed(self) -> None:
        """Compute word embeddings."""
        self.index_to_token: List[str] = []
        self.token_to_index: Mapping[str, int] = {}

        embeddings = []

        tokens = set()
        for sentence in self.sentences:
            for token in sentence.tokens:
                tokens.add(token)
        for i, token in enumerate(tokens):
            self.index_to_token.append(token)
            self.token_to_index[token] = i
            embeddings.append(self.embeddings(token))
        self.embeddings = numpy.vstack(embeddings)

    def _compute_BOWs(self) -> None:
        self._BOWs: List[collections.Counter] = []
        for i, sentence in enumerate(self.sentences):
            self._BOWs.append(collections.Counter({
                self.token_to_index[token]: sentence.tokens.count(token)
                for token in sentence.tokens
            }))

    def _compute_nBOW(self, indices: Iterable[int]) -> fwmd.NBOW:
        BOW = sum((self._BOWs[i] for i in indices), collections.Counter())
        word_indices, weights = zip(*BOW.items())
        weights = numpy.array(weights, dtype=numpy.float32)
        normalization = sum(len(self.sentences[i].tokens) for i in indices)
        return fwmd.NBOW(list(word_indices), weights / normalization)

    _T = TypeVar('_T', bound=Hashable)

    def _most_similar(self, keys: Sequence[_T],
                      indices_list: Sequence[Iterable[int]], k: int = 1
                      ) -> _T:
        query = self._doc_nBOW
        nBOWs = [self._compute_nBOW(indices) for indices in indices_list]
        logger.debug('computing most similar nBOW amongst {} nBOWs.'.format(
            len(nBOWs)))
        start = time.time()
        calc = fwmd.WMD(self.embeddings)
        nn = calc.nn(query, dict(zip(keys, nBOWs)), k=1, m_base='n', m_coeff=1,
                     n_workers=self.n_workers)
        logger.debug(
            'computed the {} most similar nBOWs in {:.2f} seconds'.format(
                k,
                time.time() - start))
        return nn[0][1]

    def greedy_approximation(self, summary_size: int = 100) -> Set[int]:
        """Greedy approximation for finding the best set of sentences.

        Args:
            summary_size (int): word length limit of the summary.

        Returns:
            (value, set) tuple (int, list): the value of the approximated
              objective function and the set of selected sentences as a tuple.

        """
        logger.debug('initializing the greedy approximation procedure')

        logger.debug('initializing the set of selected items')
        S: Set[int] = set()

        logger.debug('initializing the set of candidates')
        C = set(range(len(self.sentences)))

        summary_length = 0

        logger.debug('looping until the set of candidates is empty')
        while len(C) > 0:

            logger.debug('removing unsuitable items')
            C = set(c for c in C
                    if summary_length + self.sentences[c].length
                    <= summary_size)

            logger.debug('stopping if there are no candidates left')
            if not C:
                break

            logger.debug('computing the best candidate')
            keys = []
            indices_list = []
            for c in C:
                keys.append(c)
                indices_list.append(S | {c})
            c = self._most_similar(keys, indices_list)

            logger.debug('selecting sentence {}'.format(c))

            logger.debug('adding the best candidate to the selected items')
            S.add(c)
            summary_length += self.sentences[c].length

            logger.debug('removing the best candidate from the candidates')
            C.remove(c)

        return S
