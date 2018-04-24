from __future__ import unicode_literals

import sume.base


def test_load_file(shared_datadir):
    path = str((shared_datadir / 'cluster').resolve())
    lf = sume.base.LoadFile(path)
    lf.read_documents()
    assert len(lf.sentences) == 9


def test_untokenize():
    tokens_list = [
        ['I', 'untokenize', '.'],
        ['I', ' have', 'weird ', ' spaces ', '.'],
        ['Today', 'is', 'sunny', ',', 'and', 'warm', '.'],
        ['``', 'I', 'have', 'quotes', "''", ',', 'he', 'said', '.']
    ]
    sentences = [
        'I untokenize.',
        'I have weird spaces.',
        'Today is sunny, and warm.',
        "``I have quotes'', he said."
    ]
    for tokens, sentence in zip(tokens_list, sentences):
        assert sume.base.untokenize(tokens) == sentence
