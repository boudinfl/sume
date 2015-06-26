# -*- coding: utf-8 -*-

""" PatchedWord2Vec summarization methods.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
             Hugo Mougard (hugo.mougard@univ-nantes.fr)
    version: 0.1
    date: June 2015
"""

from sume.base import Sentence, untokenize
from sume.utils import infer_patched_word2vec

import codecs
import operator
import os
import re

import nltk
import numpy as np
from gensim import matutils


class PatchedWord2VecSummarizer:
    """PatchedWord2Vec summarization model.

    """
    def __init__(self,
                 input_directory,
                 w2v_bin,
                 train_sequences,
                 dimensions=400,
                 window=7,
                 epochs=20,
                 min_count=1):
        """
        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.
            w2v_bin (str): path to the patched word2vec binary.
            train_sequences (str): path to the train corpus

        """
        self.input_directory = input_directory
        self.w2v_bin = w2v_bin
        self.train_sequences = train_sequences
        self.dimensions = dimensions,
        self.window = window,
        self.epochs = epochs,
        self.min_count = min_count
        self.sentences = []
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        self.topic = []

    def read_documents(self, file_extension="txt"):
        """Read the input files in the given directory.

        Load the input files and populate the sentence list. Input files are
        expected to be in one tokenized sentence per line format.

        Args:
            file_extension (str): the file extension for input documents,
              defaults to txt.
        """
        for infile in os.listdir(self.input_directory):

            # skip files with wrong extension
            if not infile.endswith(file_extension):
                continue

            with codecs.open(self.input_directory + '/' + infile,
                             'r',
                             'utf-8') as f:

                # load the sentences
                lines = f.readlines()

                # loop over sentences
                for i in range(len(lines)):

                    # split the sentence into tokens
                    tokens = lines[i].strip().split(' ')

                    # add the sentence
                    if len(tokens) > 0:
                        sentence = Sentence(tokens, infile, i)
                        untokenized_form = untokenize(tokens)
                        sentence.untokenized_form = untokenized_form
                        sentence.length = len(untokenized_form.split(' '))
                        self.sentences.append(sentence)

    def prune_sentences(self,
                        mininum_sentence_length=5,
                        remove_citations=True,
                        remove_redundancy=True):
        """Prune the sentences.

        Remove the sentences that are shorter than a given length, redundant
        sentences and citations from entering the summary.

        Args:
            mininum_sentence_length (int): the minimum number of words for a
              sentence to enter the summary, defaults to 5
            remove_citations (bool): indicates that citations are pruned,
              defaults to True
            remove_redundancy (bool): indicates that redundant sentences are
              pruned, defaults to True

        """
        pruned_sentences = []

        # loop over the sentences
        for sentence in self.sentences:

            # prune short sentences
            if sentence.length < mininum_sentence_length:
                continue

            # prune citations
            first_token, last_token = sentence.tokens[0], sentence.tokens[-1]
            if remove_citations and \
               (first_token == u"``" or first_token == u'"') and \
               (last_token == u"''" or first_token == u'"'):
                continue

            # prune identical and almost identical sentences
            if remove_redundancy:
                is_redundant = False
                for prev_sentence in pruned_sentences:
                    if sentence.tokens == prev_sentence.tokens:
                        is_redundant = True
                        break

                if is_redundant:
                    continue

            # otherwise add the sentence to the pruned sentence container
            pruned_sentences.append(sentence)

        self.sentences = pruned_sentences

    def build_representations(self, stemming=False):
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

    def average_cosinus_similarities(self, summaries):
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

            sims = self.average_cosinus_similarities(S | {c} for c in C)

            # select best candidate
            i, sim = max(enumerate(sims), key=operator.itemgetter(1))
            best_c = C[i]

            S.add(best_c)
            summary_weight = sim
            summary_length += self.sentences[best_c].length

            # remove the selected sentence
            del C[i]

        return summary_weight, S
