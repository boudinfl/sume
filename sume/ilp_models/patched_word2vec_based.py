# -*- coding: utf-8 -*-

""" PatchedWord2Vec summarization methods.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
             Hugo Mougard (hugo.mougard@univ-nantes.fr)
    version: 0.1
    date: June 2015
"""

from sume.ilp_models import Summarizer
from sume.utils import infer_patched_word2vec

import operator
import re

import numpy as np
from gensim import matutils


class PatchedWord2VecSummarizer(Summarizer):
    """PatchedWord2Vec summarization model.

    """
    def __init__(self,
                 input_directory,
                 w2v_bin,
                 train_sequences,
                 file_extension="txt",
                 mininum_sentence_length=5,
                 remove_citations=True,
                 remove_redundancy=True,
                 dimensions=400,
                 window=7,
                 epochs=20,
                 min_count=1,
                 stemming=False):
        """
        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.
            w2v_bin (str): path to the patched word2vec binary.
            train_sequences (str): path to the train corpus

        """
        super(self.__class__, self).__init__(
            input_directory,
            file_extension=file_extension,
            mininum_sentence_length=mininum_sentence_length,
            remove_citations=remove_citations,
            remove_redundancy=remove_redundancy)
        self.w2v_bin = w2v_bin
        self.train_sequences = train_sequences
        self.dimensions = dimensions
        self.window = window
        self.epochs = epochs
        self.min_count = min_count
        self.topic = []
        self._build_representations(stemming)

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

    def _average_cosinus_similarities(self, summaries):
        sequences = [[concept
                      for sentence in summary
                      for concept in self.sentences[sentence].concepts]
                     for summary in summaries]
        sequences += [self.topic]
        embeddings = map(matutils.unitvec,
                         infer_patched_word2vec(self.w2v_bin,
                                                self.train_sequences,
                                                sequences,
                                                self.dimensions,
                                                self.window,
                                                self.epochs,
                                                self.min_count))
        summary_embeddings = embeddings[:-1]
        topic_embedding = embeddings[-1]
        return [np.dot(topic_embedding, summary_embedding)
                for summary_embedding in summary_embeddings]

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
            C = [c for c in C
                 if summary_length + self.sentences[c].length <=
                 summary_size]

            # stop if no scores are to be computed
            if not C:
                break

            sims = self._average_cosinus_similarities(S | {c} for c in C)

            # select best candidate
            i, sim = max(enumerate(sims), key=operator.itemgetter(1))
            best_c = C[i]

            S.add(best_c)
            summary_weight = sim
            summary_length += self.sentences[best_c].length

            # remove the selected sentence
            del C[i]

        return summary_weight, S
