# -*- coding: utf-8 -*-

""" Doc2Vec summarization methods.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
             Hugo Mougard (hugo.mougard@univ-nantes.fr)
    version: 0.1
    date: June 2015
"""

from sume.base import Sentence, untokenize

import os
import re
import codecs
import sys
import bisect

import nltk
from gensim.models import Doc2Vec

class Doc2VecSummarizer:
    """Doc2Vec summarization model.

    """
    def __init__(self, input_directory, file_extension="sentences"):
        """
        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.
            file_extension (str): the file extension for input documents,
              defaults to sentences.

        """
        self.input_directory = input_directory
        self.file_extension = file_extension
        self.sentences = []
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        self.topic = []

    def read_documents(self):
        """Read the input files in the given directory.

        Load the input files and populate the sentence list. Input files are
        expected to be in one tokenized sentence per line format.
        """
        for infile in os.listdir(self.input_directory):

            # skip files with wrong extension
            if infile[-len(self.file_extension):] != self.file_extension:
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
        """Build the word representations for each sentence

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

    def score_sentences(self, doc2vec_model, summary_size=100):
        """Greedy approximation for scoring the sentences using the Doc2Vec 
           model.

           Args:
               doc2vec_model (Doc2Vec model): the Doc2Vec trained model 

        """

        # load the model
        model = doc2vec_model #Doc2Vec.load(path_to_model)

        # filter the concepts according to the model
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].concepts = [u for u in sentence.concepts \
                                          if u in model.vocab]
            # populates the topic container
            for token in self.sentences[i].concepts:
                self.topic.append(token)

        # initialize the subset of selected sentences
        G = set([])

        # initialize the set of sentences
        U = set(range(len(self.sentences)))

        # initialize summary variables
        summary_weight = 0.0
        summary_length = 0.0
        summary_words = []

        while len(U) > 0:

            # initialize the score container
            scores = []

            # remove sentences that are too long
            remaining_sentences = U.copy()
            for i in remaining_sentences:
                if summary_length+self.sentences[i].length > summary_size:
                    U.remove(i)

            # stop if no scores are to be computed
            if len(U) == 0:
                break

            # initialize the score of each candidate sentence
            for i in U:

                # compute the summary similarity
                sim = model.n_similarity(self.topic,
                        self.sentences[i].concepts+summary_words)

                # compute the gain
                gain = (sim-summary_weight)
                gain /= float(summary_length+self.sentences[i].length)

                # add the score for the candidate sentence
                bisect.insort(scores, (gain, i, sim))

            # select best candidate
            gain, i , sim = scores[-1]

            # test if summary length is not exceeded
            if summary_weight+self.sentences[i].length <= summary_size:
                G.add(i)
                summary_weight = sim
                summary_length += self.sentences[i].length
                summary_words += self.sentences[i].concepts

            # remove the selected sentence 
            U.remove(i)

        return G






















