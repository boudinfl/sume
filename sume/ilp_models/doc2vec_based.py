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

import numpy as np
from gensim import matutils


class Doc2VecSummarizer(Summarizer):
    """Doc2Vec summarization model using @gojomo's refactor (inference stage).

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
            model (Doc2Vec): Doc2Vec model to use to compute embeddings.
        """
        super(self.__class__, self).__init__(
            input_directory,
            file_extension=file_extension,
            mininum_sentence_length=mininum_sentence_length,
            remove_citations=remove_citations,
            remove_redundancy=remove_redundancy)
        self.model = model
        self.topic = []
        self.topic_embedding = None
        self._build_representations(stemming)
        self._filter_out_of_vocabulary()

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

    def _infer_vectors(self,
                      sequences,
                      alpha=0.1,
                      min_alpha=0.001,
                      epochs=5):
        """Inference step of paragraph2vec
        Args:
            sequences (list): the sequences to embed
        """
        return [self.model.infer_vector(s,
                                        alpha=alpha,
                                        min_alpha=min_alpha,
                                        steps=epochs)
                for s in sequences]

    def _average_cosinus_similarities(self, summaries):
        if self.topic_embedding is None:
            self.topic_embedding = matutils.unitvec(
                self._infer_vectors([self.topic])[0])
        sequences = [[concept
                      for sentence in summary
                      for concept in self.sentences[sentence].concepts]
                     for summary in summaries]
        raw_embeddings = self._infer_vectors(sequences)
        embeddings = map(matutils.unitvec, raw_embeddings)
        return [np.dot(self.topic_embedding, embedding)
                for embedding in embeddings]

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
