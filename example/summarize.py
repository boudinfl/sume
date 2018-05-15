# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import unicode_literals

import sume

# create a summarizer, here a concept-based ILP model
s = sume.models.ConceptBasedILPSummarizer("data/")

# load documents with extension 'txt'
s.read_documents(file_extension="txt")

# compute the parameters needed by the model
# extract bigrams as concepts
s.extract_ngrams()

# compute document frequency as concept weights
s.compute_document_frequency()

# solve the ilp model
value, subset = s.greedy_approximation(summary_size=20)

# outputs the summary
print('\n'.join([s.sentences[j].untokenized_form for j in subset]))