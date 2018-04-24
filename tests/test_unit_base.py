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
