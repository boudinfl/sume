import sume.base


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
