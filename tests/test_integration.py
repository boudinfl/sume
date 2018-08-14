# -*- coding: utf-8 -*-

# sume
# Copyright (C) 2014, 2015, 2018 Florian Boudin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path

import nltk

from sume.models.concept_based_ilp_summarizer import ConceptBasedILPSummarizer


def test_integration(shared_datadir: Path) -> None:
    nltk.download('stopwords')
    # directory from which text documents to be summarized are loaded. Input
    # files are expected to be in one tokenized sentence per line format.
    dir_path = str((shared_datadir / 'cluster').resolve())

    # create a summarizer, here a concept-based ILP model
    s = ConceptBasedILPSummarizer(dir_path, file_extension='.txt')

    # compute the parameters needed by the model
    # extract bigrams as concepts
    s.extract_concepts(n=2, stemming=True)

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
