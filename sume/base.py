# -*- coding: utf-8 -*-

# sume
# Copyright (C) 2014, 2015, 2018 Florian Boudin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Base structures and functions for the sume module.

Base contains the Sentence, LoadFile and State classes.
"""

from collections import Counter

import codecs
import os
import re

import nltk


class State(object):
    """State class.

    Internal class used as a structure to keep track of the search state in
    the tabu_search method.

    Args:
        subset (set): a subset of sentences
        concepts (Counter): a set of concepts for the subset
        length (int): the length in words
        score (int): the score for the subset

    """

    def __init__(self):
        """Construct a State object."""
        self.subset = set()
        self.concepts = Counter()
        self.length = 0
        self.score = 0


class Sentence(object):
    """The sentence data structure.

    Args:
        tokens (list of str): the list of word tokens.
        doc_id (str): the identifier of the document from which the sentence
          comes from.
        position (int): the position of the sentence in the source document.

    """

    def __init__(self, tokens, doc_id, position):
        """Construct a sentence."""
        self.tokens = tokens
        """ tokens as a list. """

        self.doc_id = doc_id
        """ document identifier of the sentence. """

        self.position = position
        """ position of the sentence within the document. """

        self.concepts = []
        """ concepts of the sentence. """

        self.untokenized_form = ''
        """ untokenized form of the sentence. """

        self.length = 0
        """ length of the untokenized sentence. """


class Reader(object):
    """Reader class to process input documents."""

    def __init__(self,
                 input_directory):
        """Construct a base summarizer to inherit from.

        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.
            file_extension (str): the extension considered as input files by
              the reader.
            mininum_sentence_length (int): minimum length of sentences to keep
              in the input documents. 0 = keep everything.
            remove_citations (bool): whether to remove citations.
            remove_redundancy (bool): whether to remove redundant sentences.

        """
        self.input_directory = input_directory
        self.sentences = []
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')

    def read_documents(self, file_extension):
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

            with codecs.open(os.path.join(self.input_directory, infile),
                             'r',
                             'utf-8') as f:

                # loop over sentences
                for i, line in enumerate(f.readlines()):

                    # split the sentence into tokens
                    tokens = line.strip().split(' ')

                    # add the sentence
                    if tokens:
                        sentence = Sentence(tokens, infile, i)
                        untokenized_form = untokenize(tokens)
                        sentence.untokenized_form = untokenized_form
                        sentence.length = len(untokenized_form.split(' '))
                        self.sentences.append(sentence)

    def prune_sentences(self,
                        mininum_sentence_length=1,
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
               (first_token == "``" or first_token == '"') and \
               (last_token == "''" or first_token == '"'):
                continue

            # prune ___ said citations
            # if remove_citations and \
            #     (sentence.tokens[0]=="``" or sentence.tokens[0]=='"') and \
            #     re.search(r'(?i)(''|") \w{,30} (said|reported|told)\.$',
            #               sentence.untokenized_form):
            #     continue

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

    def extract_concepts(self, n=1, stemming=False):
        """Extract the ngrams of words from the input sentences.

        Args:
            n (int): the number of words for ngrams, defaults to 2
        """
        for i, sentence in enumerate(self.sentences):

            # for each ngram of words
            for j in range(len(sentence.tokens) - (n - 1)):

                # initialize ngram container
                ngram = []

                # for each token of the ngram
                for k in range(j, j + n):
                    ngram.append(sentence.tokens[k].lower())

                # do not consider ngrams containing punctuation marks
                marks = [t for t in ngram if not re.search(r'[a-zA-Z0-9]', t)]
                if len(marks) > 0:
                    continue

                # do not consider ngrams composed of only stopwords
                stops = [t for t in ngram if t in self.stoplist]
                if len(stops) == len(ngram):
                    continue

                # stem the ngram
                if stemming:
                    ngram = [self.stemmer.stem(t) for t in ngram]

                # add the ngram to the concepts
                self.sentences[i].concepts.append(' '.join(ngram))


def untokenize(tokens):
    """Untokenize a list of tokens.

    Args:
        tokens (list of str): the list of tokens to untokenize.

    Returns:
        a string

    """
    text = ' '.join(tokens)
    text = re.sub(r"\s+", r" ", text.strip())
    text = re.sub(r" ('[a-z]) ", "\g<1> ", text)
    text = re.sub(r" ([\.;,-]) ", "\g<1> ", text)
    text = re.sub(r" ([\.;,-?!])$", "\g<1>", text)
    text = re.sub(r" _ (.+) _ ", " _\g<1>_ ", text)
    text = re.sub(r" \$ ([\d\.]+) ", " $\g<1> ", text)
    text = text.replace(" ' ", "' ")
    text = re.sub(r"([\W\s])\( ", "\g<1>(", text)
    text = re.sub(r" \)([\W\s])", ")\g<1>", text)
    text = text.replace("`` ", "``")
    text = text.replace(" ''", "''")
    text = text.replace(" n't", "n't")
    text = re.sub(r'(^| )" ([^"]+) "( |$)', '\g<1>"\g<2>"\g<3>', text)

    # times
    text = re.sub(r'(\d+) : (\d+ [ap]\.m\.)', '\g<1>:\g<2>', text)

    text = re.sub(r'^" ', '"', text)
    text = re.sub(r' "$', '"', text)
    text = re.sub(r"\s+", " ", text.strip())

    return text
