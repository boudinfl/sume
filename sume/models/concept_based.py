# -*- coding: utf-8 -*-
"""Concept-based ILP summarization methods."""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from sume import *

from collections import defaultdict, deque

import os
import re
import codecs
import random
import sys

import nltk

class ConceptBasedILPSummarizer(LoadFile):
    """Implementation of the concept-based ILP model for summarization.

    The original algorithm was published and described in:

      * Dan Gillick and Benoit Favre, A Scalable Global Model for Summarization,
        *Proceedings of the NAACL HLT Workshop on Integer Linear Programming for
        Natural Language Processing*, pages 10–18, 2009.
        
    """

    def __init__(self, input_directory):
        """Redefining initializer for ConceptBasedILPSummarizer.

        Args:
            input_directory (str): the directory from which text documents to
                be summarized are loaded.

        """
        super(ConceptBasedILPSummarizer, self).__init__(
                                                input_directory=input_directory)

        self.weights = {}
        """ weights container. """

        self.stoplist = nltk.corpus.stopwords.words('english')
        """ list of stopwords from nltk / english. """

        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        """ stemming algorithm from nltk / english. """

        self.concept_to_sentence = defaultdict(set)
        """ mapping from concepts to sentences. """

    def extract_ngrams(self, n=2):
        """Extract the ngrams of words from the input sentences.

        Args:
            n (int): the number of words for ngrams, defaults to 2
        """
        for i, sentence in enumerate(self.sentences):

            # for each ngram of words
            for j in range(len(sentence.tokens)-(n-1)):

                # initialize ngram container
                ngram = []

                # for each token of the ngram
                for k in range(j, j+n):
                    ngram.append(sentence.tokens[k].lower())

                # do not consider ngrams containing punctuation marks
                marks = [t for t in ngram if not re.search('[a-zA-Z0-9]', t)]
                if len(marks) > 0:
                    continue

                # do not consider ngrams composed of only stopwords
                stops = [t for t in ngram if t in self.stoplist]
                if len(stops) == len(ngram):
                    continue

                # stem the ngram
                ngram = [self.stemmer.stem(t) for t in ngram]

                # add the ngram to the concepts
                self.sentences[i].concepts.append(' '.join(ngram))

    def compute_document_frequency(self):
        """Compute the document frequency of each concept."""
        for i in range(len(self.sentences)):

            # for each concept
            for concept in self.sentences[i].concepts:

                # add the document id to the concept weight container
                if concept not in self.weights:
                    self.weights[concept] = set([])
                self.weights[concept].add(self.sentences[i].doc_id)

        # loop over the concepts and compute the document frequency
        for concept in self.weights:
            self.weights[concept] = len(self.weights[concept])

    def prune_concepts(self, method="threshold", value=3):
        """Prune the concepts for efficient summarization.

        Args:
            method (str): the method for pruning concepts that can be whether
                by using a minimal value for concept scores (threshold) or using
                the top-N highest scoring concepts (top-n), defaults to
                threshold.
            value (int): the value used for pruning concepts, defaults to 3.

        """
        # 'threshold' pruning method
        if method == "threshold":

            # iterates over the concept weights
            concepts = self.weights.keys()
            for concept in concepts:
                if self.weights[concept] < value:
                    del self.weights[concept]

        # 'top-n' pruning method
        elif method == "top-n":

            # sort concepts by scores
            sorted_concepts = sorted(self.weights,
                                     key=lambda x: self.weights[x],
                                     reverse=True)

            # iterates over the concept weights
            concepts = self.weights.keys()
            for concept in concepts:
                if concept not in sorted_concepts[:value]:
                    del self.weights[concept]

        # iterates over the sentences
        for i in range(len(self.sentences)):

            # current sentence concepts
            concepts = self.sentences[i].concepts

            # prune concepts
            self.sentences[i].concepts = [c for c in concepts
                                          if c in self.weights]

    def _compute_concept_to_sentences(self):
        """Compute the inverted concept to sentences dictionary. """

        for i, sentence in enumerate(self.sentences):
            for concept in sentence.concepts:
                self.concept_to_sentence[concept].add(i)

    def greedy_approximation(self, summary_size=100):
        """Greedy approximation of the ILP model.

        Args:
            summary_size (int): the maximum size in words of the summary,
              defaults to 100.

        Returns:
            (value, set) tuple (int, list): the value of the approximated
              objective function and the set of selected sentences as a tuple.

        """
        # initialize the inverted c2s dictionary if not already created
        if not self.concept_to_sentence:
            self._compute_concept_to_sentences()

        # initialize weights
        weights = {}

        # initialize the score of the best singleton
        best_singleton_score = 0

        # compute indices of our sentences
        sentences = range(len(self.sentences))

        # compute initial weights and fill the reverse index
        # while keeping track of the best singleton solution
        for i, sentence in enumerate(self.sentences):
            weights[i] = sum(self.weights[c] for c in set(sentence.concepts))
            if sentence.length <= summary_size\
               and weights[i] > best_singleton_score:
                best_singleton_score = weights[i]
                best_singleton = i

        # initialize the selected solution properties
        sel_subset, sel_concepts, sel_length, sel_score = set(), set(), 0, 0

        # greedily select a sentence
        while True:

            ###################################################################
            # RETRIEVE THE BEST SENTENCE
            ###################################################################

            # sort the sentences by gain and reverse length
            sort_sent = sorted(((weights[i] / float(self.sentences[i].length),
                                 -self.sentences[i].length,
                                 i)
                                for i in sentences),
                               reverse=True)

            # select the first sentence that fits in the length limit
            for sentence_gain, rev_length, sentence_index in sort_sent:
                if sel_length - rev_length <= summary_size:
                    break
            # if we don't find a sentence, break out of the main while loop
            else:
                break

            # if the gain is null, break out of the main while loop
            if not weights[sentence_index]:
                break

            # update the selected subset properties
            sel_subset.add(sentence_index)
            sel_score += weights[sentence_index]
            sel_length -= rev_length

            # update sentence weights with the reverse index
            for concept in set(self.sentences[sentence_index].concepts):
                if concept not in sel_concepts:
                    for sentence in self.concept_to_sentence[concept]:
                        weights[sentence] -= self.weights[concept]

            # update the last selected subset property
            sel_concepts.update(self.sentences[sentence_index].concepts)

        # check if a singleton has a better score than our greedy solution
        if best_singleton_score > sel_score:
            return best_singleton_score, set([best_singleton])

        # returns the (objective function value, solution) tuple
        return sel_score, sel_subset

