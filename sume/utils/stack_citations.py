# -*- coding: utf-8 -*-

"""Reconstruct the citations from the DUC/TAC files."""

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import codecs


def main():
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

        stacked_lines = []
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
