import nltk
import sume.models.concept_based


def test_integration(shared_datadir):
    nltk.download('stopwords')
    # directory from which text documents to be summarized are loaded. Input
    # files are expected to be in one tokenized sentence per line format.
    dir_path = str((shared_datadir / 'cluster').resolve())

    # create a summarizer, here a concept-based ILP model
    s = sume.models.concept_based.ConceptBasedILPSummarizer(dir_path)

    # load documents with extension 'txt'
    s.read_documents(file_extension="txt")

    # compute the parameters needed by the model
    # extract bigrams as concepts
    s.extract_ngrams()

    # compute document frequency as concept weights
    s.compute_document_frequency()

    # prune sentences that are shorter than 10 words, identical sentences and
    # those that begin and end with a quotation mark
    s.prune_sentences(mininum_sentence_length=2,
                      remove_citations=True,
                      remove_redundancy=True)

    # solve the ilp model
    value, subset = s.solve_ilp_problem()

    # cluster is small, summary should contain all the sentences
    assert len(subset) == 9
