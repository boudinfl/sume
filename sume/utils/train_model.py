# -*- coding: utf-8 -*-

import codecs
import argparse

from gensim.models.doc2vec import LabeledLineSentence, LabeledSentence
from gensim.models import Doc2Vec


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sources', nargs='+', help='', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-d', '--dimension', default=100, type=int)
parser.add_argument('-m', '--min_count', default=5, type=int)
parser.add_argument('-t', '--threads', default=1, type=int)
parser.add_argument('-w', '--window', default=8, type=int)
parser.add_argument('-e', '--epoch', default=10, type=int)
parser.add_argument('--sample', default=int(10e12), type=int)
args = parser.parse_args()

sentences = []

for input_file in args.sources:
    print 'info - loading data source: ', input_file
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            sentences.append(LabeledSentence(line.strip().split(),
                                             ['SENT_%s' % len(sentences)]))
            if len(sentences) >= args.sample:
                break

print 'info - building model from', len(sentences), 'items'

model = Doc2Vec(size=args.dimension,        # dimensionality of the feature vectors
                window=args.window,         # maximum distance between the current and predicted word
                min_count=args.min_count,   # ignore words with total frequency lower than this
                workers=args.threads,       # number of threads
                alpha=0.025,                # initial learning rate
                min_alpha=0.025)            # use fixed learning rate

model.build_vocab(sentences)

for epoch in range(args.epoch):
    print "epoch", epoch
    model.train(sentences)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

param_output = args.output
param_output += '.d'+str(args.dimension)
param_output += '.w'+str(args.window)
param_output += '.m'+str(args.min_count)
param_output += '.e'+str(args.epoch)
param_output += '.word2vec'

model.save(param_output)
