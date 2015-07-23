# -*- coding: utf-8 -*-

""" Concept-based ILP summarization methods.

    authors: Florian Boudin (florian.boudin@univ-nantes.fr)
             Hugo Mougard (hugo.mougard@univ-nantes.fr)
    version: 0.2
    date: May 2015
"""

from sume.base import Sentence, State, untokenize

from collections import defaultdict, deque

import os
import re
import codecs
import random
import sys

import nltk
import pulp


class ConceptBasedILPSummarizer:
    """Concept-based ILP summarization model.

    Implementation of (Gillick & Favre, 2009) ILP model for summarization.
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
        self.c2s = defaultdict(set)
        self.concept_sets = defaultdict(frozenset)
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')

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

    def compute_c2s(self):
        """Compute the inverted concept to sentences dictionary. """

        for i, sentence in enumerate(self.sentences):
            for concept in sentence.concepts:
                self.c2s[concept].add(i)

    def compute_concept_sets(self):
        """Compute the concept sets for each sentence."""

        for i, sentence in enumerate(self.sentences):
            for concept in sentence.concepts:
                self.concept_sets[i] |= {concept}

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
        if not self.c2s:
            self.compute_c2s()

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
                    for sentence in self.c2s[concept]:
                        weights[sentence] -= self.weights[concept]

            # update the last selected subset property
            sel_concepts.update(self.sentences[sentence_index].concepts)

        # check if a singleton has a better score than our greedy solution
        if best_singleton_score > sel_score:
            return best_singleton_score, set([best_singleton])

        # returns the (objective function value, solution) tuple
        return sel_score, sel_subset

    def tabu_search(self, summary_size=100, memory_size=5, iterations=30):
        """Greedy approximation of the ILP model with a tabu search
          meta-heuristic.

        Args:
            summary_size (int): the maximum size in words of the summary,
              defaults to 100.
            memory_size (int): the maximum size of the pool of sentences
              to ban at a given time, defaults at 5.
            iterations (int): the number of iterations to run, defaults at
              30.

        Returns:
            (value, set) tuple (int, list): the value of the approximated
              objective function and the set of selected sentences as a tuple.

        """
        if not self.c2s:
            raise AssertionError(
                "The solver's reverse index c2s is empty. "
                "Did you execute solver.compute_c2s()?")
        if not self.concept_sets:
            raise AssertionError(
                "The solver's concept sets dictionary is empty. "
                "Did you execute solver.compute_concept_sets()?")

        # initialize weights
        weights = {}

        # initialize the score of the best singleton
        best_singleton_score = 0

        # compute initial weights and fill the reverse index
        # while keeping track of the best singleton solution
        for i, sentence in enumerate(self.sentences):
            weights[i] = sum(self.weights[c] for c in set(sentence.concepts))
            if sentence.length <= summary_size\
               and weights[i] > best_singleton_score:
                best_singleton_score = weights[i]
                best_singleton = i

        best_subset, best_score = None, 0
        state = State()
        for i in xrange(iterations):
            queue = deque([], memory_size)
            # greedily select sentences
            state = self.select_sentences(summary_size,
                                          weights,
                                          state,
                                          queue)
            if state.score > best_score:
                best_subset = state.subset.copy()
                best_score = state.score
            to_tabu = set(random.sample(state.subset, 2))
            state = self.unselect_sentences(weights, state, to_tabu)
            queue.extend(to_tabu)

        # check if a singleton has a better score than our greedy solution
        if best_singleton_score > best_score:
            return best_singleton_score, set([best_singleton])

        # returns the (objective function value, solution) tuple
        return best_score, best_subset

    def select_sentences(self, summary_size, weights, state, tabu_set):
        """Greedy sentence selector.

        Args:
            summary_size (int): the maximum size in words of the summary,
              defaults to 100.
            weights (dictionary): the sentence weights dictionary. This
              dictionnary is updated during this method call (in-place).
            state (State): the state of the tabu search from which to start
              selecting sentences.
            tabu_set (iterable): set of sentences that are tabu: this
              selector will not consider them.

        Returns:
            state (State): the new state of the search. Also note that
              weights is modified in-place.

        """
        # greedily select a sentence while respecting the tabu
        while True:

            ###################################################################
            # RETRIEVE THE BEST SENTENCE
            ###################################################################

            # sort the sentences by gain and reverse length
            sort_sent = sorted(((weights[i] / float(self.sentences[i].length),
                                 -self.sentences[i].length,
                                 i)
                                for i in range(len(self.sentences))),
                               reverse=True)

            # select the first sentence that fits in the length limit
            for sentence_gain, rev_length, sentence_index in sort_sent:
                if sentence_index not in tabu_set \
                   and state.length - rev_length <= summary_size:
                    break
            # if we don't find a sentence, break out of the main while loop
            else:
                break

            # if the gain is null, break out of the main while loop
            if not weights[sentence_index]:
                break

            # update state
            state.subset |= {sentence_index}
            state.concepts.update(self.concept_sets[sentence_index])
            state.length -= rev_length
            state.score += weights[sentence_index]

            # update sentence weights with the reverse index
            for concept in set(self.concept_sets[sentence_index]):
                if state.concepts[concept] == 1:
                    for sentence in self.c2s[concept]:
                        weights[sentence] -= self.weights[concept]
        return state

    def unselect_sentences(self, weights, state, to_remove):
        """Sentence ``un-selector'' (reverse operation of the
          select_sentences method).

        Args:
            weights (dictionary): the sentence weights dictionary. This
              dictionnary is updated during this method call (in-place).
            state (State): the state of the tabu search from which to start
              un-selecting sentences.
            to_remove (iterable): set of sentences to unselect.

        Returns:
            state (State): the new state of the search. Also note that
              weights is modified in-place.

        """
        # remove the sentence indices from the solution subset
        state.subset -= to_remove
        for sentence_index in to_remove:
            # update state
            state.concepts.subtract(self.concept_sets[sentence_index])
            state.length -= self.sentences[sentence_index].length
            # update sentence weights with the reverse index
            for concept in set(self.concept_sets[sentence_index]):
                if not state.concepts[concept]:
                    for sentence in self.c2s[concept]:
                        weights[sentence] += self.weights[concept]
            state.score -= weights[sentence_index]
        return state

    def solve_ilp_problem(self,
                          summary_size=100,
                          solver='glpk',
                          excluded_solutions=[]):
        """Solve the ILP formulation of the concept-based model.

        Args:
            summary_size (int): the maximum size in words of the summary,
              defaults to 100.
            solver (str): the solver used, defaults to glpk.
            excluded_solutions (list of list): a list of subsets of sentences
              that are to be excluded, defaults to []

        Returns:
            (value, set) tuple (int, list): the value of the objective function
              and the set of selected sentences as a tuple.

        """
        # initialize container shortcuts
        concepts = self.weights.keys()
        w = self.weights
        L = summary_size
        C = len(concepts)
        S = len(self.sentences)

        # HACK Sort keys
        concepts = sorted(self.weights, key=self.weights.get, reverse=True)

        # formulation of the ILP problem
        prob = pulp.LpProblem(self.input_directory, pulp.LpMaximize)

        # initialize the concepts binary variables
        c = pulp.LpVariable.dicts(name='c',
                                  indexs=range(C),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')

        # initialize the sentences binary variables
        s = pulp.LpVariable.dicts(name='s',
                                  indexs=range(S),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')

        # OBJECTIVE FUNCTION
        prob += sum(w[concepts[i]] * c[i] for i in range(C))

        # CONSTRAINT FOR SUMMARY SIZE
        prob += sum(s[j] * self.sentences[j].length for j in range(S)) <= L

        # INTEGRITY CONSTRAINTS
        for i in range(C):
            for j in range(S):
                if concepts[i] in self.sentences[j].concepts:
                    prob += s[j] <= c[i]

        for i in range(C):
            prob += sum(s[j] for j in range(S)
                        if concepts[i] in self.sentences[j].concepts) >= c[i]

        # CONSTRAINTS FOR FINDING OPTIMAL SOLUTIONS
        for sentence_set in excluded_solutions:
            prob += sum([s[j] for j in sentence_set]) <= len(sentence_set)-1

        # solving the ilp problem
        if solver == 'gurobi':
            prob.solve(pulp.GUROBI(msg=0))
        elif solver == 'glpk':
            prob.solve(pulp.GLPK(msg=0))
        elif solver == 'cplex':
            prob.solve(pulp.CPLEX(msg=0))
            # prob.writeLP('test.lp')
        else:
            sys.exit('no solver specified')

        # retreive the optimal subset of sentences
        solution = set([j for j in range(S) if s[j].varValue == 1])

        # returns the (objective function value, solution) tuple
        return (pulp.value(prob.objective), solution)
