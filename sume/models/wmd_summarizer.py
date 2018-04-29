# -*- coding: utf-8 -*-

"""Word Mover's Distance summarization."""

import collections
import logging

import fastText
import numpy
import wmd

from ..base import Reader


logger = logging.getLogger(__name__)


class WMDSummarizer(Reader):
    """Word Mover's Distance summarization model."""

    def __init__(self,
                 model,
                 *args,
                 **kwargs):
        """Construct a WMD summarizer.

        Args:
            model (FastText model): the model to use to compute word
              embeddings (with the .bin extension).
            args: args to pass on to the sume.base.Reader constructor.
            kwargs: kwargs to pass on to the sume.base.Reader constructor.
        """
        super().__init__(*args, **kwargs)
        logger.debug('loading fastText model')
        self.model = fastText.load_model(model)
        logger.debug('loaded fastText model')
        logger.debug('embedding words')
        self._embed()
        logger.debug('embedded words')
        logger.debug('computing BOWs')
        self._compute_BOWs()
        logger.debug('computed BOWs')
        logger.debug('computing document nBOW')
        self._doc_nBOW = self._compute_nBOW((-1, range(len(self.sentences))))
        logger.debug('computed document nBOW')

    def _embed(self):
        """Compute word embeddings."""
        self.index_to_token = []
        self.token_to_index = {}

        embeddings = []

        tokens = set()
        for sentence in self.sentences:
            for token in sentence.tokens:
                tokens.add(token)
        for i, token in enumerate(tokens):
            self.index_to_token.append(token)
            self.token_to_index[token] = i
            embeddings.append(self.model.get_word_vector(token))
        self.embeddings = numpy.vstack(embeddings)

    def _compute_BOWs(self):
        self._BOWs = []
        for i, sentence in enumerate(self.sentences):
            self._BOWs.append(collections.Counter({
                self.token_to_index[token]: sentence.tokens.count(token)
                for token in sentence.tokens
            }))

    def _compute_nBOW(self, args):
        key, indices = args
        BOW = sum((self._BOWs[i] for i in indices), collections.Counter())
        word_indices, weights = zip(*BOW.items())
        weights = numpy.array(weights, dtype=numpy.float32)
        normalization = sum(len(self.sentences[i].tokens) for i in indices)
        return key, word_indices, weights / normalization

    def _most_similar(self, summaries):
        nBOWs = {self._doc_nBOW[0]: self._doc_nBOW}
        for key, indices, weights in map(self._compute_nBOW, summaries):
            nBOWs[key] = (key, indices, weights)
        calc = wmd.WMD(self.embeddings, nBOWs)
        return calc.nearest_neighbors(self._doc_nBOW[0],
                                      k=1,
                                      early_stop=1)[0][0]

    def greedy_approximation(self, summary_size=100):
        """Greedy approximation for finding the best set of sentences.

        Args:
            summary_size (int): word length limit of the summary.

        Returns:
            (value, set) tuple (int, list): the value of the approximated
              objective function and the set of selected sentences as a tuple.

        """
        # initialize the set of selected items
        S = set()

        # initialize the set of item candidates
        C = set(range(len(self.sentences)))

        summary_length = 0

        # main loop -> until the set of candidates is empty
        while len(C) > 0:

            # remove unsuitable items
            C = set(c for c in C
                    if summary_length + self.sentences[c].length
                    <= summary_size)

            # stop if no scores are to be computed
            if not C:
                break

            c = self._most_similar([(c, S | {c}) for c in C])

            S.add(c)
            summary_length += self.sentences[c].length

            # remove the selected sentence
            C.remove(c)

        return S
