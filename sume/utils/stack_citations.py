#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import sys
import logging
import codecs

""" Reconstruct the citations from the DUC/TAC files.

    author: florian boudin (florian.boudin@univ-nantes.fr)
"""

logging.basicConfig(level=logging.INFO)

# open the input file
with codecs.open(sys.argv[1], 'r', 'utf-8') as f:

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
            if token == u"``":
                openings.append(token)
            if token == u"''":
                endings.append(token)
            if token == u'"':
                remp_char = u'``'
                if len(openings) > len(endings):
                    remp_char = u"''"
                    endings.append(remp_char)
                else:
                    openings.append(remp_char)
                logging.info('error with quotation marks at ' + sys.argv[1])
                logging.info('info - correcting, modifying with ' + remp_char)
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

    # separate the last punctuation marks
    for i, line in enumerate(stacked_lines):
        stacked_lines[i] = re.sub('(\w)([\.\!\?])$', '\g<1> \g<2>', line)
        stacked_lines[i] = re.sub("(\w)([\.\!\?]) (\"|'')$", '\g<1> \g<2> \g<3>', stacked_lines[i])
        if line == stacked_lines[i]:
            logging.info('no modification for sentence: ' + line)

    # write the reconstructed file
    with codecs.open(sys.argv[2], 'w', 'utf-8') as w:
        w.write('\n'.join(stacked_lines))