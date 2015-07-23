#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Doc2Vec doc:
#   github.com/piskvorky/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

import argparse
import codecs
import os.path

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from multiprocessing import cpu_count
from os import makedirs
from random import shuffle

cores = cpu_count() * 2

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help='Model type', dest='model_type')

parser_w2v = subparsers.add_parser('w2v',
                                   help='Word2Vec')
parser_d2v = subparsers.add_parser('d2v',
                                   help='Doc2Vec')

# General arguments

parser.add_argument('-s', '--source', help='Training corpus', required=True)
parser.add_argument('-o',
                    '--output',
                    help='Output directory. The model will be named '
                    'automatically.',
                    required=True)
parser.add_argument('--dimension',
                    default=100,
                    type=int,
                    help='Dimensionality of the vectors.')
parser.add_argument('--window',
                    default=8,
                    type=int,
                    help='Size of the context window (used on both sides).')
parser.add_argument('--min-count',
                    default=1,
                    type=int,
                    help='Threshold of commonness to keep words.')
parser.add_argument('--workers',
                    default=cores,
                    type=int,
                    help='Number of workers to use.')
parser.add_argument('--negative-sampling',
                    default=5,
                    type=int,
                    help='Number of negative values to sample to learn the '
                    'softmax.')
parser.add_argument('--epoch',
                    default=20,
                    type=int,
                    help='Number of passes through the data.')
parser.add_argument('--alpha',
                    default=0.025,
                    type=float,
                    help='Starting learning rate.')
parser.add_argument('--min-alpha',
                    default=0.001,
                    type=float,
                    help='Final learning rate.')

# Doc2Vec arguments

parser_d2v.add_argument('--dbow',
                        action='store_true',
                        help='Use distributed bag of words instead of '
                        'distributed memory to compute paragraph vectors.')
parser_d2v.add_argument('--dm-concat',
                        action='store_true',
                        help='Use concatenation of the context vectors '
                        'instead of averaging.')

# Word2Vec arguments

parser_w2v.add_argument('--sg',
                        action='store_true',
                        help='Use the skip-gram model instead of the '
                        'continuous bag of words model.')

args = parser.parse_args()

sequences = []

print 'info - loading data source: ', args.source
with codecs.open(args.source, 'r', 'utf-8') as f:
    for i, line in enumerate(f):
        sequence = line.strip().split()
        if args.model_type == 'w2v':
            sequences.append(sequence)
        if args.model_type == 'd2v':
            sequences.append(TaggedDocument(sequence, [i]))

print 'info - building model from', i, 'items'

if args.model_type == 'd2v':
    if args.dbow:
        model = Doc2Vec(dm=0,
                        size=args.dimension,
                        window=args.window,
                        min_count=args.min_count,
                        workers=args.workers,
                        alpha=args.alpha,
                        min_alpha=args.min_alpha)
    else:
        model = Doc2Vec(dm=1,
                        dm_concat=1 if args.dm_concat else 0,
                        dm_mean=0 if args.dm_concat else 1,
                        size=args.dimension,
                        window=args.window,
                        min_count=args.min_count,
                        workers=args.workers,
                        alpha=args.alpha,
                        min_alpha=args.min_alpha)

else:
    model = Word2Vec(sg=1 if args.sg else 0,
                     size=args.dimension,
                     window=args.window,
                     min_count=args.min_count,
                     workers=args.workers,
                     alpha=args.alpha,
                     min_alpha=args.min_alpha)

model.build_vocab(sequences)

alpha_delta = (args.alpha - args.min_alpha) / args.epoch

for epoch in xrange(args.epoch):
    shuffle(sequences)
    print "epoch", epoch
    model.train(sequences)
    model.alpha -= alpha_delta  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

if args.model_type == 'd2v':
    if args.dbow:
        method = 'dbow'
    else:
        method = 'dmc' if args.dm_concat else 'dma'
else:
    method = 'sg' if args.sg else 'cbow'

param_output = '{input}-d{dimension}-w{window}-m{min_count}-e{epoch}-a{alpha}'\
               '-ma{min_alpha}-n{ns}-{method}.{model_type}'.format(
                   input=os.path.basename(args.source),
                   dimension=args.dimension,
                   window=args.window,
                   min_count=args.min_count,
                   epoch=args.epoch,
                   alpha=args.alpha,
                   min_alpha=args.min_alpha,
                   ns=args.negative_sampling,
                   method=method,
                   model_type=args.model_type)

output_dir = os.path.join(args.output, param_output)
output_files = os.path.join(output_dir, param_output)
if not os.path.isdir(output_dir):
    makedirs(output_dir)
model.save(output_files)
