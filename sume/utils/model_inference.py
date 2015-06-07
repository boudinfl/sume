# -*- coding: utf-8 -*-

from gensim.models.doc2vec import LabeledSentence
from gensim.models.word2vec import Vocab

import numpy as np


def infer(model, sequences):
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
