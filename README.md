# sume

The sume module is an automatic summarization library written in Python. 

## Description

sume contains the following extraction algorithms :

  * Concept-based ILP model for summarization ([Gillick & Favre, 2009][GF09])

A typical usage of this module is:

    import sume

    # directory from which text documents to be summarized are loaded. Input
    # files are expected to be in one tokenized sentence per line format.
    dir_path = "/tmp/"

    # create a summarizer, here a concept-based ILP model
    s = sume.models.ConceptBasedILPSummarizer(dir_path)

    # load documents with extension 'txt'
    s.read_documents(file_extension="txt")

    # compute the parameters needed by the model
    # extract bigrams as concepts
    s.extract_ngrams()

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

## Contributors

* Florian Boudin
* Hugo Mougard

## References

[GF09]: Dan Gillick and Benoit Favre, A Scalable Global Model for Summarization,
        *Proceedings of the NAACL HLT Workshop on Integer Linear Programming for
        Natural Language Processing*, pages 10–18, 2009.
