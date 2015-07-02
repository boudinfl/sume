# -*- coding: utf-8 -*-

from gensim.models.doc2vec import LabeledSentence
from gensim.models.word2vec import Vocab

from itertools import chain
from tempfile import NamedTemporaryFile

import codecs
import logging
import numpy as np
import subprocess
import sys


def infer_new_doc2vec(model,
                      sequences,
                      alpha=0.1,
                      min_alpha=0.001,
                      epochs=5):
    """Inference step of paragraph2vec
    Args:
        model (Doc2Vec model): the model to use to embed the sequences
        sequences (list): the sequences to embed
    """
    return [model.infer_vector(s,
                               alpha=alpha,
                               min_alpha=min_alpha,
                               steps=epochs)
            for s in sequences]

def infer_doc2vec(model, sequences):
    """Inference step of paragraph2vec
    Args:
        model (Doc2Vec model): the model to use to embed the sequences
        sequences (list): the sequences to embed

    Returns:
        embeddings (list): the embeddings of the sequences
          computed with the given model.
    """
    # based on https://gist.github.com/zseder/4201551d7f8608f0b82b
    # and https://groups.google.com/forum/#!topic/gensim/EFy1f0QwkKI
    # number of sentences used during model training
    n_train_sents = len([l for l in model.vocab
                         if l.startswith("SENT_")])

    # sequences we want to embed (e.g. sentences, sentence
    # combinations and multi-document texts).
    sequences = [LabeledSentence(sequence,
                                 ["SENT_%s" % (n_train_sents + i)])
                 for i, sequence in enumerate(sequences)]

    # vocabulary size before inference step
    n_vocab = len(model.vocab)

    # number of sequences to embed
    n_sequences = len(sequences)
    for i, sequence in enumerate(sequences):
        vocab_index = n_vocab + i
        label = sequence.labels[0]
        # create a vocabulary entry
        model.vocab[label] = Vocab(count=1,
                                   index=vocab_index,
                                   code=[0],
                                   sample_probability=1.)
        # create a reverse index entry
        model.index2word.append(label)
    # add rows to syn0 to be able to train the new sequences
    model.syn0 = np.vstack((
        model.syn0,
        np.empty((n_sequences, model.layer1_size),
                 dtype=np.float32)))
    # initialize them randomly
    for i in xrange(n_vocab, n_vocab + n_sequences):
        np.random.seed(
            np.uint32(model.hashfxn(
                model.index2word[i] + str(model.seed))))
        model.syn0[i] = (np.random.rand(model.layer1_size) - 0.5) \
            / model.layer1_size

    # train the model
    model.train_words = False
    model.train_lbls = True
    model.alpha = 0.025
    model.min_alpha = 0.025
    for epoch in xrange(10):
        model.train(sequences)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return [model["SENT_%s" % (n_train_sents + i)]
            for i in xrange(n_sequences)]


def infer_patched_word2vec(w2v_bin,
                           train_sequences,
                           sequences,
                           dimensions,
                           window,
                           epochs,
                           min_count):
    with NamedTemporaryFile() as w2v_input, \
         NamedTemporaryFile() as w2v_output, \
         codecs.open(train_sequences, 'r', 'utf-8') as fh_train:
        for i, sequence in enumerate(chain((' '.join(s) for s in sequences),
                                           fh_train.readlines())):
            w2v_input.write((u'_*%s %s\n' % (i, sequence)).encode('utf-8'))
        args = [w2v_bin,
                '-train', w2v_input.name,
                '-output', w2v_output.name,
                '-cbow', '0',
                '-size', str(dimensions),
                '-window', str(window),
                '-negative', '5',
                '-hs', '0',
                '-sample', '1e-4',
                '-threads', '40',
                '-binary', '0',
                '-iter', str(epochs),
                '-min-count', str(min_count),
                '-sentence-vectors', '1']
        code = subprocess.call(args)
        if code != 0:
            logging.error('The external call to word2vec returned a non zero '
                          'code. The args were:'
                          + str(args))
            sys.exit(1)
        i = 0
        result = []
        for line in w2v_output.readlines():
            if line.startswith('_*'):
                result.append(np.array(map(float, line.split(' ')[1:-1])))
                i += 1
            if i >= len(sequences):
                break
    return result
