# -*- coding: utf-8 -*-

""" Doc2Vec summarization methods.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
             Hugo Mougard (hugo.mougard@univ-nantes.fr)
    version: 0.1
    date: June 2015
"""

from sume.base import Sentence, untokenize
from sume.utils import Server
from sume.utils import infer

import bisect
import codecs
import operator
import os
import random
import re
import warnings

import nltk
import numpy as np
from gensim import matutils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.word2vec import Vocab


class Doc2VecSummarizer:
    """Doc2Vec summarization model.

    """
    def __init__(self, input_directory, model, method="awe"):
        """
        Args:
            input_directory (str): the directory from which text documents to
              be summarized are loaded.
            model (Doc2Vec model): the model to use to compute similarities
              between text segments.
            method          (str): the method to use to build embeddings. Also
              influences the way similarities are computed:
                - "awe" uses average word embeddings to represent sequences of
                  words;
                - "infer" uses doc2vec inference step for sentence and topic
                  embedding, and average sentence embedding for combination of
                  sentences (should eventually also be an inference step).

        """
        self.input_directory = input_directory
        self.sentences = []
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        self.topic = []
        self.topic_embedding = None
        self.embeddings = {}
        self.model = model
        self.method = method

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

    def filter_out_of_vocabulary(self):
        """Filter out of vocabulary words."""
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].concepts = [u for u in sentence.concepts
                                          if u in self.model.vocab]

        self.topic = [u for u in self.topic if u in self.model.vocab]

    def build_embeddings(self):
        """Build embeddings for the multi-document text and for individual
        sentences.

        Available methods are "awe" for Average Word Embeddings and
        "infer" for the paragraph2vec infer algorithm.

        """
        if self.method == "awe":
            for s in self.sentences:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.embeddings[s] = np.array([self.model[t]
                                                for t in s.concepts])\
                                          .mean(axis=0)
                    self.topic_embedding = matutils.unitvec(
                        np.array([self.model[t] for t in self.topic])
                        .mean(axis=0))

        elif self.method == "infer":
            sequences = [s.concepts for s in self.sentences]
            sequences += [self.topic]

            embeddings = infer(self.model, sequences)

            sentence_embs, topic_emb = embeddings[:-1], embeddings[-1]
            # put the obtained embeddings in our summarizer state
            self.topic_embedding = matutils.unitvec(topic_emb)
            for i, sentence in enumerate(self.sentences):
                self.embeddings[sentence] = sentence_embs[i]
        else:
            raise AssertionError('self.method should be "awe" or "infer".')

    def average_cosinus_similarity(self, sentences):
        # here we need to compute a weighted average of sentence embeddings
        # to obtain a word embedding average
        if self.method == "awe":
            sentences_embedding = matutils.unitvec(np.average(
                [self.embeddings[self.sentences[s]] for s in sentences],
                axis=0,
                weights=[len(self.sentences[s].concepts) for s in sentences]))
        # here just a sentence embedding average. Should be another infer step
        # eventually
        elif self.method == "infer":
            sentences_embedding = matutils.unitvec(np.mean(
                [self.embeddings[self.sentences[s]] for s in sentences],
                axis=0))
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
        S = set([])

        # initialize the set of item candidates
        C = set(range(len(self.sentences)))

        # initialize summary variables
        summary_weight = 0.0
        summary_length = 0.0
        summary_words = []

        # main loop -> until the set of candidates is empty
        while len(C) > 0:

            # initialize the score container
            scores = []

            # remove unsuitables items
            C = set([c for c in C
                    if summary_length + self.sentences[c].length <=
                     summary_size])

            # stop if no scores are to be computed
            if not C:
                break

            # initialize the score of each candidate sentence
            for i in C:

                # compute the summary similarity
                sim = self.model.n_similarity(
                    self.topic,
                    self.sentences[i].concepts + summary_words)

                # compute the gain
                gain = (sim-summary_weight)
                # gain /= float(summary_length+self.sentences[i].length)

                # add the score for the candidate sentence
                bisect.insort(scores, (gain, i, sim))

            # select best candidate
            gain, i, sim = scores[-1]

            # test if summary length is not exceeded
            if summary_weight+self.sentences[i].length <= summary_size:
                S.add(i)
                summary_weight = sim
                summary_length += self.sentences[i].length
                summary_words += self.sentences[i].concepts

            # remove the selected sentence
            C.remove(i)

        return summary_weight, S

    def greedy_approximation_fast(self, summary_size=100):
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

            sims = [(c, self.average_cosinus_similarity(S | {c})) for c in C]

            # select best candidate
            c, sim = max(sims, key=operator.itemgetter(1))

            S.add(c)
            summary_weight = sim
            summary_length += self.sentences[c].length

            # remove the selected sentence
            C.remove(c)

        return summary_weight, S

    def greedy_approximation_par(self, summary_size=100):
        """Greedy approximation for finding the best set of sentences.

        Args:
            model (Doc2Vec model): a Doc2Vec trained model.

        Returns:
            (value, set) tuple (int, list): the value of the approximated
              objective function and the set of selected sentences as a tuple.

        """

        server = Server(self)

        # initialize the set of selected items
        S = set()

        # initialize the set of item candidates
        C = set(range(len(self.sentences)))

        # initialize summary variables
        summary_weight = 0.0
        summary_length = 0.0

        # main loop -> until the set of candidates is empty
        while len(C) > 0:

            # remove unsuitable items
            C = set(c for c in C
                    if summary_length + self.sentences[c].length <=
                    summary_size)

            # stop if no scores are to be computed
            if not C:
                break

            sims = server.compute_sims(S, C)

            # select best candidate
            i, sim = max(sims, key=operator.itemgetter(1))

            S.add(i)
            summary_weight = sim
            summary_length += self.sentences[i].length

            # remove the selected sentence
            C.remove(i)
        server.exit()
        return summary_weight, S

    def fitness(self, tab):
        """Fitness function."""
        if tab.size == 0:
            return 0.0
        words = []
        for u in tab:
            words += self.sentences[u].concepts
        return self.model.n_similarity(self.topic, words)

    def fast_fitness(self, tab):
        """Fitness function."""
        if tab.size == 0:
            return 0.0
        return self.average_cosinus_similarity(tab)

    def differential_evolution(self,
                               NP=20,
                               gen_max=100,
                               CR=0.5,
                               F=1,
                               summary_size=100,
                               fast_fitness=False):
        """Approximate using a differential evolution."""

        # initialize dimension for arrays as number of sentences
        D = len(self.sentences)

        # initialize an array for computing summary size
        l = np.array([u.length for u in self.sentences])

        # initialize the container for the population
        P = []

        # initialize counter for generations
        count = 0

        # initialize cost container
        cost = []

        #######################################################################
        # STEP 1 : initialize initial population
        #######################################################################
        for i in xrange(NP):

            # generate an empty individual
            indiv = np.zeros(D)

            # initialize random element
            random_element = 0

            # modify random elements while summary size is not reached
            while np.sum(np.multiply(l, indiv)) <= summary_size:
                random_element = random.randint(0, D-1)
                indiv[random_element] = 1

            # modify last element
            indiv[random_element] = 0

            # add the individual to the initial population
            P.append(indiv)

            # add the cost of the individual
            if fast_fitness:
                cost.append(self.fast_fitness(np.flatnonzero(indiv)))
            else:
                cost.append(self.fitness(np.flatnonzero(indiv)))
        #######################################################################

        #######################################################################
        # MAIN LOOP
        #######################################################################
        while count < gen_max:

            # Loop through population
            for i in xrange(NP):

                ###############################################################
                # STEP 2 : Mutate/Recombine
                ###############################################################

                # Randomly pick 3 vectors all different from i
                a = random.randint(0, NP-1)
                while a == i:
                    a = random.randint(0, NP-1)

                b = random.randint(0, NP-1)
                while b == i and b == a:
                    b = random.randint(0, NP-1)

                c = random.randint(0, NP-1)
                while c == i and c == a and c == b:
                    c = random.randint(0, NP-1)

                # Randomly pick first parameter
                j = random.randint(0, D-1)

                # itilialize trial with P[i]
                trial = np.copy(P[i])

                # load D parameters into trial
                for k in xrange(D):

                    # Perfom D-1 binomial trials
                    if random.random() >= CR or k == D:
                        trial[k] = (P[c][k] + F * (P[a][k] - P[b][k])) % 2

                # Check consistency of trial
                while np.sum(np.multiply(l, trial)) > summary_size:
                    non_zeros = np.flatnonzero(trial)
                    random_element = random.choice(non_zeros)
                    trial[random_element] = 0

                ###############################################################
                # STEP 3 : Evaluate/Select
                ###############################################################
                if fast_fitness:
                    score = self.fast_fitness(np.flatnonzero(trial))
                else:
                    score = self.fitness(np.flatnonzero(trial))
                # print score, cost[i]
                if score > cost[i]:
                    P[i] = trial
                    cost[i] = score

            # end of generation, increment counter
            count += 1
        #######################################################################

        objective = np.max(cost)
        return (objective, set(np.flatnonzero(P[cost.index(objective)])))
        #######################################################################
