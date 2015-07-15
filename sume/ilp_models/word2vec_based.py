# -*- coding: utf-8 -*-

""" Doc2Vec summarization methods.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
             Hugo Mougard (hugo.mougard@univ-nantes.fr)
    version: 0.1
    date: June 2015
"""

from sume.ilp_models import Summarizer

import operator
import re
import warnings

import numpy as np
from gensim import matutils


class Word2VecSummarizer(Summarizer):
    """Word2Vec summarization model.

    """
    def __init__(self,
                 input_directory,
                 model,
                 file_extension="txt",
                 mininum_sentence_length=5,
                 remove_citations=True,
                 remove_redundancy=True,
                 stemming=False):
        """
        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.
            model (Word2Vec model): the model to use to compute similarities
              between text segments.
        """
        super(self.__class__, self).__init__(
            input_directory,
            file_extension=file_extension,
            mininum_sentence_length=mininum_sentence_length,
            remove_citations=remove_citations,
            remove_redundancy=remove_redundancy)
        self.topic = []
        self.topic_embedding = None
        self.embeddings = {}
        self.model = model
        self._build_representations(stemming)
        self._filter_out_of_vocabulary()
        self._build_embeddings()

    def _build_representations(self, stemming):
        """Build the word representations for each sentence and for the topic.

           Args:
               stemming (bool): indicates whether stemming is applied, defaults
                 to False

        """
        for i, sentence in enumerate(self.sentences):

            # iterates over the sentence tokens and populates the concepts
            for token in sentence.tokens:

                # do not consider stopwords
                if token in self.stoplist:
                    continue

                # do not consider punctuation marks
                if not re.search('[a-zA-Z0-9]', token):
                    continue

                # add the stem to the concepts
                if stemming:
                    sentence.concepts.append(self.stemmer.stem(token.lower()))
                else:
                    sentence.concepts.append(token.lower())

            for token in self.sentences[i].concepts:
                self.topic.append(token)

    def _filter_out_of_vocabulary(self):
        """Filter out of vocabulary words."""
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].concepts = [u for u in sentence.concepts
                                          if u in self.model.vocab]

        self.topic = [u for u in self.topic if u in self.model.vocab]

    def _build_embeddings(self):
        """Build embeddings for the multi-document text and for individual
        sentences.

        """
        for s in self.sentences:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.embeddings[s] = np.array([self.model[t]
                                               for t in s.concepts])\
                                       .mean(axis=0)
                self.topic_embedding = matutils.unitvec(
                    np.array([self.model[t] for t in self.topic])
                    .mean(axis=0))

    def _average_cosinus_similarity(self, sentences):
        # here we need to compute a weighted average of sentence embeddings
        # to obtain a word embedding average
        sentences_embedding = matutils.unitvec(np.average(
            [self.embeddings[self.sentences[s]] for s in sentences],
            axis=0,
            weights=[len(self.sentences[s].concepts) for s in sentences]))
        return np.dot(self.topic_embedding, sentences_embedding)

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

        # initialize summary variables
        summary_weight = 0.0
        summary_length = 0

        # main loop -> until the set of candidates is empty
        while len(C) > 0:

            # remove unsuitable items
            C = set(c for c in C
                    if summary_length + self.sentences[c].length <=
                    summary_size)

            # stop if no scores are to be computed
            if not C:
                break

            sims = [(c, self._average_cosinus_similarity(S | {c})) for c in C]

            # select best candidate
            c, sim = max(sims, key=operator.itemgetter(1))

            S.add(c)
            summary_weight = sim
            summary_length += self.sentences[c].length

            # remove the selected sentence
            C.remove(c)

        return summary_weight, S
