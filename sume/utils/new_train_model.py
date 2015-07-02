# -*- coding: utf-8 -*-
# github.com/piskvorky/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

import argparse
import codecs
import os.path

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from multiprocessing import cpu_count
from os import makedirs
from random import shuffle

cores = cpu_count() * 2

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', help='Training corpus', required=True)
parser.add_argument('-o',
                    '--output',
                    help='Output directory. The model will be named '
                    'automatically.',
                    required=True)
parser.add_argument('--dimension',
                    default=100,
                    type=int,
                    help='Number of dimensions for the paragraph vectors.')
parser.add_argument('--window',
                    default=5,
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
parser.add_argument('--dbow',
                    action='store_true',
                    help='Use distributed bag of words instead of distributed '
                    'memory to compute paragraph vectors.')
parser.add_argument('--dm-concat',
                    action='store_true',
                    help='Use concatenation of the context vectors instead of '
                    'averaging.')
parser.add_argument('--sample',
                    default=int(10e12),
                    type=int,
                    help='Number of examples to limit the corpus to.')
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
args = parser.parse_args()

docs = []

print 'info - loading data source: ', args.source
with codecs.open(args.source, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            docs.append(TaggedDocument(line.strip().split(),
                                            [i]))
            if i >= args.sample:
                break

print 'info - building model from', i, 'items'

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

model.build_vocab(docs)

alpha_delta = (args.alpha - args.min_alpha) / args.epoch

for epoch in xrange(args.epoch):
    shuffle(docs)
    print "epoch", epoch
    model.train(docs)
    model.alpha -= alpha_delta  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

param_output = '{input}-d{dimension}-w{window}-m{min_count}-e{epoch}-a{alpha}'\
               '-ma{min_alpha}-n{ns}-s{sample}-{method}.doc2vec'.format(
                   input=os.path.basename(args.source),
                   dimension=args.dimension,
                   window=args.window,
                   min_count=args.min_count,
                   epoch=args.epoch,
                   alpha=args.alpha,
                   min_alpha=args.min_alpha,
                   ns=args.negative_sampling,
                   sample=args.sample,
                   method='dbow' if args.dbow else
                   'dmc' if args.dm_concat else 'dma')

output_dir = os.path.join(args.output, param_output)

output_files = os.path.join(output_dir, param_output)

if not os.path.isdir(output_dir):
    makedirs(output_dir)

model.save(output_files)
