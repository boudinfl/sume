# -*- coding: utf-8 -*-

""" Base structures and functions for the sume module

    author: florian boudin (florian.boudin@univ-nantes.fr)
    version: 0.1
    date: Nov. 2014
"""

import re
from collections import Counter

class State:
    """ State class

    Internal class used as a structure to keep track of the search state in 
    the tabu_search method.

    Args:
        subset (set): a subset of sentences
        concepts (Counter): a set of concepts for the subset
        length (int): the length in words
        score (int): the score for the subset
    """
    def __init__(self):
        self.subset = set()
        self.concepts = Counter()
        self.length = 0
        self.score = 0

class Sentence:
    """The sentence data structure.

    Args: 
        tokens (list of str): the list of word tokens.
        doc_id (str): the identifier of the document from which the sentence
          comes from.
        position (int): the position of the sentence in the source document.
    """
    def __init__(self, tokens, doc_id, position):

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

def untokenize(tokens):
    """Untokenizing a list of tokens. 

    Args:
        tokens (list of str): the list of tokens to untokenize.

    Returns:
        a string

    """
    text = u' '.join(tokens)
    text = re.sub(u"\s+", u" ", text.strip())
    text = re.sub(u" ('[a-z]) ", u"\g<1> ", text)
    text = re.sub(u" ([\.;,-]) ", u"\g<1> ", text)
    text = re.sub(u" ([\.;,-?!])$", u"\g<1>", text)
    text = re.sub(u" _ (.+) _ ", u" _\g<1>_ ", text)
    text = re.sub(u" \$ ([\d\.]+) ", u" $\g<1> ", text)
    text = text.replace(u" ' ", u"' ")
    text = re.sub(u"([\W\s])\( ", u"\g<1>(", text)
    text = re.sub(u" \)([\W\s])", u")\g<1>", text)
    text = text.replace(u"`` ", u"``")
    text = text.replace(u" ''", u"''")
    text = text.replace(u" n't", u"n't")
    text = re.sub(u'(^| )" ([^"]+) "( |$)', u'\g<1>"\g<2>"\g<3>', text)

    # times
    text = re.sub('(\d+) : (\d+ [ap]\.m\.)', '\g<1>:\g<2>', text)

    text = re.sub('^" ', '"', text)
    text = re.sub(' "$', '"', text)
    text = re.sub(u"\s+", u" ", text.strip())

    return text
