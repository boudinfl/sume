# sume

The **sume** module is an automatic summarization library written in Python.

## Installation

The python module is available on PyPi. To install it, you can use:

    pip install sume

**sume** also requires a linear solver on the path, such as `glpsol` from GLPK.
Installation may vary. For a debian/ubuntu system, you can use:

    sudo apt-get install glpk-utils

## Description

**sume** contains the following extraction algorithms:

- Concept-based ILP model for summarization [(Gillick & Favre, 2009)][gf]

A typical usage of this module is:

    import nltk
    import sume.models.concept_based

    # directory from which text documents to be summarized are loaded. Input
    # files are expected to be in one tokenized sentence per line format.
    dir_path = "/tmp/"

    # download stopwords from nltk (required by sume summarizers).
    nltk.download('stopwords')

    # create a summarizer, here a concept-based ILP model
    s = sume.models.concept_based.ConceptBasedILPSummarizer(dir_path)

    # load documents (with extension '.txt' by default)
    s.read_documents()

    # compute the parameters needed by the model
    # extract stemmed bigrams as concepts
    s.extract_concepts(n=2, stemming=True)

    # compute document frequency as concept weights
    s.compute_document_frequency()

    # prune sentences that are shorter than 10 words, identical sentences and
    # those that begin and end with a quotation mark
    s.prune_sentences(mininum_sentence_length=10,
                      remove_citations=True,
                      remove_redundancy=True)

    # solve the ilp model
    value, subset = s.solve_ilp_problem()

    # outputs the summary
    print '\n'.join([s.sentences[j].untokenized_form for j in subset])

## Citing the sume module

If you use sume, please cite the following paper:

> [Florian Boudin, Hugo Mougard and Benoît Favre, Concept-based Summarization
using Integer Linear Programming: From Concept Pruning to Multiple Optimal
Solutions, *Proceedings of the 2015 Conference on Empirical Methods in
Natural Language Processing (EMNLP)*][bmf].

## Contributors

- Florian Boudin
- Hugo Mougard

[gf]: http://www.aclweb.org/anthology/W09-1802

[bmf]: http://aclweb.org/anthology/D15-1220
