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

"""Reconstruct the citations from the DUC/TAC files."""

import argparse
import codecs
import sys
from typing import List


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Extract the textual content from the DUC/TAC files.')
    parser.add_argument('input', help='input file path')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args()

    # open the input file
    with codecs.open(args.input, 'r', 'utf-8') as f:

        # read the lines from the input file
        lines = f.readlines()

        stacked_lines: List[str] = []
        in_citation = False

        openings = []
        endings = []

        # for each line
        for line in lines:

            line = line.strip()
            tokens = line.split(' ')

            for i in range(len(tokens)):
                token = tokens[i]
                if token == "``":
                    openings.append(token)
                if token == "''":
                    endings.append(token)
                if token == '"':
                    remp_char = '``'
                    if len(openings) > len(endings):
                        remp_char = "''"
                        endings.append(remp_char)
                    else:
                        openings.append(remp_char)
                    print('info - error with quotation marks at', sys.argv[1])
                    print('info - correcting, modifying with ', remp_char)
                    tokens[i] = remp_char
                    line = ' '.join(tokens)

            if len(openings) == len(endings):
                if in_citation:
                    stacked_lines[-1] = stacked_lines[-1] + ' ' + line
                else:
                    stacked_lines.append(line)
                in_citation = False

            else:
                if in_citation:
                    stacked_lines[-1] = stacked_lines[-1] + ' ' + line
                else:
                    stacked_lines.append(line)
                    in_citation = True

        # write the reconstructed file
        with codecs.open(args.output, 'w', 'utf-8') as w:
            w.write('\n'.join(stacked_lines))


if __name__ == '__main__':
    main()
