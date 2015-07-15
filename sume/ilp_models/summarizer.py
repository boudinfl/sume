# -*- coding: utf-8 -*-

""" Base summarizer class.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
             Hugo Mougard (hugo.mougard@univ-nantes.fr)
    version: 0.2
    date: July 2015
"""

from sume.base import Sentence, untokenize

import os
import os.path
import codecs

import nltk

class Summarizer(object):
    """Concept-based ILP summarization model.

    Implementation of (Gillick & Favre, 2009) ILP model for summarization.
    """
    def __init__(self,
                 input_directory,
                 file_extension="txt",
                 mininum_sentence_length=5,
                 remove_citations=True,
                 remove_redundancy=True):
        """
        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.

        """
        self.input_directory = input_directory
        self.sentences = []
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        self._read_documents(file_extension)
        self._prune_sentences(mininum_sentence_length,
                             remove_citations,
                             remove_redundancy)

    def _read_documents(self, file_extension):
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

    def _prune_sentences(self,
                        mininum_sentence_length,
                        remove_citations,
                        remove_redundancy):
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

            # prune ___ said citations
            # if remove_citations and \
            #     (sentence.tokens[0]==u"``" or sentence.tokens[0]==u'"') and \
            #     re.search('(?i)(''|") \w{,30} (said|reported|told)\.$',
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
