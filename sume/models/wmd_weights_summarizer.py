# -*- coding: utf-8 -*-


"""Word Mover's Distance summarization."""


import collections
import itertools
import logging
import multiprocessing
from typing import Callable, Iterable, List, Mapping

import fwmd
import numpy

from ..base import Reader

logger = logging.getLogger(__name__)


class WMDWeightsSummarizer(Reader):
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

        docs = collections.defaultdict(set)

        for i, sentence in enumerate(self.sentences):
            docs[sentence.doc_id].add(i)

        calc = fwmd.WMD(self.embeddings)

        accumulated_costs = numpy.zeros(self.embeddings.shape[0])
        for a, b in itertools.combinations(docs.keys(), 2):
            wmd, costs = calc.wmd(self._compute_nBOW(docs[a]),
                                  self._compute_nBOW(docs[b]),
                                  return_costs=True)
            accumulated_costs += costs
        print(costs)

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

    def solve_ilp_problem(self) -> None:
        pass
