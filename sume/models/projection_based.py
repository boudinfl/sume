# -*- coding: utf-8 -*-

""" Maximum semantic volume model.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
    version: 0.1
    date: Sept 2015
"""

from sume.base import LoadFile

from collections import defaultdict

import os
import re
import sys

import nltk
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class MaximumSemanticVolumeSummarizer(LoadFile):
    """Implementation of the maximum semantic volume model for summarization.

    The original algorithm was published and described in:

      * Dani Yogatama, Fei Liu and Noah A. Smith, Extractive Summarization by 
        Maximizing Semantic Volume, *Proceedings of the 2015 Conference on 
        Empirical Methods in Natural Language Processing*, pages 1961-1966,
        2015.
        
    """
    def __init__(self, input_directory):
        """
        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.

        """
        self.input_directory = input_directory
        self.sentences = []
        self.weights = {}
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        self.embeddings = []
        self.c2s = defaultdict(set)

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
        """Compute the document frequency of each concept.

        """
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

    def compute_document_frequency(self):
        """Compute the document frequency of each concept.

        """
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

    def compute_c2s(self):
        """Compute the inverted concept to sentences dictionary. """

        for i, sentence in enumerate(self.sentences):
            for concept in sentence.concepts:
                self.c2s[concept].add(i)

    def compute_SVD_sentence_embeddings(self, K=500):
        """Compute the sentence embeddings using SVD decomposition.

        Args:
            K (int): the number of latent dimensions, defaults to 500.

        """
        # initialize the inverted c2s dictionary if not already created
        if not self.c2s:
            self.compute_c2s()

        # initialize the concept conatiner for ordered list of bigrams
        concepts = list(self.c2s)

        # initialize the S matrix (N x B)
        S = np.zeros((len(self.sentences), len(concepts)))

        # stack sentence vectors in columns to build the matrix S
        for i, sentence in enumerate(self.sentences):
            for j in range(len(concepts)):
                if concepts[j] in sentence.concepts:
                    S[i][j] = self.weights[concepts[j]]

        # compute the SVD
        U, s, V = np.linalg.svd(S, full_matrices=False)
        # print 'S', S.shape

        self.embeddings = U[:, :K]

        # print 'U', U.shape
        
        # plt.scatter(U[:,0], U[:,1])
        # plt.show()

    def greedy_approximation(self, summary_size=100):
        """Volume maximization function.

        """
        selected_sentences = set([])
        N, K = self.embeddings.shape

        # compute cluster centroid
        c = np.zeros(K)
        for i in range(K):
            c[i] = sum(self.embeddings[:,i]) / float(N)

        # find the sentence p that is the farthest from c
        distances_from_c = [np.linalg.norm(e-c) for e in self.embeddings]
        p = self.embeddings[np.argmax(distances_from_c)]

        # find the sentence q that is the farthest from p
        distances_from_p = [np.linalg.norm(e-p) for e in self.embeddings]
        q = self.embeddings[np.argmax(distances_from_p)]

        # plt.scatter(self.embeddings[:,0], self.embeddings[:,1], color='b')
        # plt.scatter(c[0], c[1], color='r')
        # plt.scatter(p[0], p[1], color='g')
        # plt.scatter(q[0], q[1], color='g')

        # plt.scatter(self.embeddings[87,0], self.embeddings[87,1], color='c')
        # plt.scatter(self.embeddings[2,0], self.embeddings[2,1], color='c')
        # plt.scatter(self.embeddings[155,0], self.embeddings[155,1], color='c')
        # plt.scatter(self.embeddings[151,0], self.embeddings[151,1], color='c')

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # print self.embeddings[0]

        # fig = plt.figure(1)
        # ax = Axes3D(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xs=self.embeddings[:,0], ys=self.embeddings[:,1], zs=self.embeddings[:,2], color='y')
        ax.scatter(c[0], c[1], c[2], color='r')
        ax.scatter(p[0], p[1], p[2], color='g')
        ax.scatter(q[0], q[1], q[2], color='g')
        ax.scatter(self.embeddings[87,0], self.embeddings[87,1], self.embeddings[87,2], color='k')
        ax.scatter(self.embeddings[2,0], self.embeddings[2,1], self.embeddings[2,2], color='k')
        ax.scatter(self.embeddings[155,0], self.embeddings[155,1], self.embeddings[155,2], color='k')
        ax.scatter(self.embeddings[151,0], self.embeddings[151,1], self.embeddings[151,2], color='k')

        # set([87, 2, 155, 151])

        plt.show()






        # Find the sentence






