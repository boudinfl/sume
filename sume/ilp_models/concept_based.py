# -*- coding: utf-8 -*-

""" Concept-based ILP summarization methods.

    author: florian boudin (florian.boudin@univ-nantes.fr)
    version: 0.1
    date: Nov. 2014
"""

from sume.base import Sentence
from sume.base import untokenize

import os
import re
import codecs
import bisect
import sys

import nltk
import pulp

class ConceptBasedILPSummarizer:
    """Concept-based ILP summarization model. 

    Implementation of (Gillick & Favre, 2009) ILP model for summarization. 

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
        self.weights = {}
        self.stoplist = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')

    def read_documents(self):
        """Read the input files in the given directory.

        Load the input files and populate the sentence list. Input files are
        expected to be in one tokenized sentence per line format.
        """
        for infile in os.listdir(self.input_directory):

            # skip files with wrong extension
            if infile[-len(self.file_extension):] != self.file_extension:
                continue

            with codecs.open(self.input_directory+'/'+infile,'r','utf-8') as f:

                # load the sentences
                lines = f.readlines()

                # loop over sentences
                for i in range(len(lines)):

                    # split the sentence into tokens
                    tokens = lines[i].strip().split(' ')

                    # add the sentence
                    if len(tokens) > 0:
                        self.sentences.append(Sentence(tokens, infile, i))
                        self.sentences[-1].untokenized_form = untokenize(tokens)
                        self.sentences[-1].length = \
                            len(self.sentences[-1].untokenized_form.split(' '))

    def extract_ngrams(self, n=2):
        """Extract the ngrams of words from the input sentences.

        Args:
            n (int): the number of words for ngrams, defaults to 2
        """
        for i in range(len(self.sentences)):

            # current sentence
            sentence = self.sentences[i]

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
                if not self.weights.has_key(concept):
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
            if remove_citations and \
              (sentence.tokens[0]==u"``" or sentence.tokens[0]==u'"') and \
              (sentence.tokens[-1]==u"''" or sentence.tokens[0]==u'"'):
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
            method (str): the method for pruning concepts that can be whether by
              using a minimal value for concept scores (threshold) or using the
              top-N highest scoring concepts (top-n), defaults to threshold.
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
            sorted_concepts = sorted( self.weights, 
                                      key=lambda x : self.weights[x], 
                                      reverse=True )

            # iterates over the concept weights
            concepts = self.weights.keys()
            for concept in concepts:
                if not concept in sorted_concepts[:value]:
                    del self.weights[concept]

        # iterates over the sentences
        for i in range(len(self.sentences)):

            # current sentence concepts
            concepts = self.sentences[i].concepts

            # prune concepts
            self.sentences[i].concepts = [c for c in concepts \
                                          if self.weights.has_key(c)]

    def greedy_approximation(self, r=1.0, summary_size=100):
        """Greedy approximation of the ILP model.

        Args:
            r (int): the scaling factor, defaults to 1.
            summary_size (int): the maximum size in words of the summary, 
              defaults to 100.

        Returns:
            (value, set) tuple (int, list): the value of the approximated 
              objective function and the set of selected sentences as a tuple.

        """

        # initialize the set of selected sentences
        G = set([])

        # initialize the set of sentences
        U = set(range(len(self.sentences)))

        # initialize the concepts contained in G
        G_concepts = set([])

        # initialize the size of the sentences in G
        G_size = 0

        # greedily select a sentence
        while len(U) > 0:

            ####################################################################
            # COMPUTE THE GAINS FOR EACH SENTENCE IN U
            ####################################################################

            # compute the score of G
            f_G = sum([self.weights[i] for i in G_concepts])

            # convert U in a list
            U_list = list(U)

            # initialize the gain container
            gains = []

            # loop through the set of sentences
            for u in U_list:

                # do not consider sentences that are too long
                if G_size + self.sentences[u].length > summary_size:
                    gains.append(0)
                    continue

                # get the concepts from G union u
                G_u = G_concepts.union(self.sentences[u].concepts)

                # compute the score of G union u
                f_G_u = sum([self.weights[i] for i in G_u])

                # compute the gain for u
                gain = (f_G_u-f_G) / float(self.sentences[u].length)**r

                # add the gain for u in the container
                gains.append(gain)

            # find the maximum gain
            max_gain = max(gains)

            # find the indexes for the maximum gain
            ks = [i for i in xrange(len(U_list)) if gains[i] == max_gain]

            # select the shortest sentence
            sizes = [self.sentences[i].length for i in ks]
            k = U_list[ks[sizes.index(min(sizes))]]

            # compute the new size
            size = G_size + self.sentences[k].length

            # compute the score of G union k
            f_G_k = sum([self.weights[i] for i in \
                         G_concepts.union(self.sentences[k].concepts)])

            # if adding the sentence is permitted
            if size <= summary_size and f_G_k-f_G >= 0:

                # add the sentence to G
                G.add(k)

                # recompute the concepts for G
                G_concepts = []
                for i in G:
                    G_concepts.extend(self.sentences[i].concepts)
                G_concepts = set(G_concepts)

                # modify the size of G
                G_size = size 

            # remove sentence k from U
            U.remove(k)

        # compute the objective function for G
        f_G = sum([self.weights[i] for i in G_concepts])

        # find the singleton with the largest objective value
        obj_values = []
        for v in range(len(self.sentences)):
            obj = sum([self.weights[i] for i in self.sentences[v].concepts])
            if self.sentences[v].length <= summary_size:
                bisect.insort(obj_values, (obj, [v]))

        # return singleton if better
        if obj_values[-1][0] > f_G:
            return obj_values[-1]

        # returns the (objective function value, solution) tuple
        return f_G, G

    def greedy_approximation2(self, r=1.0, summary_size=100):
        """Greedy approximation of the ILP model.

        Args:
            r (int): the scaling factor, defaults to 1.
            summary_size (int): the maximum size in words of the summary, 
              defaults to 100.

        Returns:
            (value, set) tuple (int, list): the value of the approximated 
              objective function and the set of selected sentences as a tuple.

        """

        # initialize the set of sentence indeces
        sentence_indeces = set(range(0,len(self.sentences)))

        # initialize the set of selected sentence indeces
        selected_sentence_indeces = set()

        # initialize the concepts contained in G
        selected_concepts = set()

        # initialize the size of the sentences in G
        selected_length = 0

        # initialize the score of G
        selected_score = 0

        # greedily select a sentence
        while sentence_indeces:

            ####################################################################
            # COMPUTE THE GAINS FOR EACH SENTENCE IN U
            ####################################################################

            best_sentence_index = None
            best_sentence = None
            best_singleton = None
            best_singleton_score = 0
            best_gain = 0
            best_score_delta = 0

            # loop through the set of sentences
            for sentence_index in sentence_indeces:

                sentence = self.sentences[sentence_index]

                # do not consider sentences that are too long
                if selected_length + sentence.length > summary_size:
                    continue

                # compute the score difference if we add the sentence
                sentence_score_delta = sum(self.weights[c]
                                           for c in sentence.concepts
                                           if c not in selected_concepts)

                if sentence_score_delta > best_singleton_score:
                    best_singleton_score = sentence_score_delta
                    best_singleton = sentence_index

                # compute the normalization of the score difference by
                # the sentence length
                sentence_gain = sentence_score_delta / float(sentence.length)

                if sentence_gain == best_gain\
                   and sentence.length < best_sentence.length\
                   or sentence_gain > best_gain:
                    best_sentence_index = sentence_index
                    best_sentence = sentence
                    best_gain = sentence_gain
                    best_score_delta = sentence_score_delta

            # add the best sentence if its gain is non null
            if best_gain > 0:

                # add the sentence index to the selected set
                selected_sentence_indeces.add(best_sentence_index)

                # update the selected concepts
                selected_concepts |= set(best_sentence.concepts)

                # update the total length of the selected sentences
                selected_length += best_sentence.length

                # update the score of the selected sentences
                selected_score += best_score_delta

            # remove sentence index from indices
            sentence_indeces.remove(sentence_index)

        if best_singleton_score > selected_score:
            return best_singleton_score, set([best_singleton])

        # returns the (objective function value, solution) tuple
        return selected_score, selected_sentence_indeces


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

        #### HACK Sort keys
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
        prob += sum([ w[concepts[i]] * c[i] for i in range(C) ])

        # CONSTRAINT FOR SUMMARY SIZE
        prob += sum([s[j] * self.sentences[j].length for j in range(S)]) <= L

        # INTEGRITY CONSTRAINTS
        for i in range(C):
            for j in range(S):
                if concepts[i] in self.sentences[j].concepts:
                    prob += s[j] <= c[i]

        for i in range(C):
            prob += sum( [s[j] for j in range(S) \
                        if concepts[i] in self.sentences[j].concepts] ) >= c[i]

        # CONSTRAINTS FOR FINDING OPTIMAL SOLUTIONS
        for sentence_set in excluded_solutions:
            prob += sum([s[j] for j in sentence_set]) <= len(sentence_set)-1

        # solving the ilp problem
        if solver == 'gurobi':
            prob.solve(pulp.GUROBI(msg = 0))
        elif solver == 'glpk':
            prob.solve(pulp.GLPK(msg = 0))
        elif solver == 'cplex':
            prob.solve(pulp.CPLEX(msg = 0))
            # prob.writeLP('test.lp')
        else:
            sys.exit('no solver specified')

        # retreive the optimal subset of sentences
        solution = set([j for j in range(S) if s[j].varValue == 1])

        # returns the (objective function value, solution) tuple
        return (pulp.value(prob.objective), solution)
