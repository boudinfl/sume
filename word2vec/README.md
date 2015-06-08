# Patched word2vec

## Description

word2vec patch to compute paragraph vectors by Tomas Mikolov. The
patch was published [on the word2vec mailing list][patch].

[patch]: https://groups.google.com/forum/#!msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ

## Installation

A Makefile is provided so that it's easier not to mess up the
compilation flags: just issue a `make` command.

## Usage

The documented use is:

    ./word2vec -train train.txt \
        -output vectors.txt \
        -cbow 0 \
        -size 100 \
        -window 10 \
        -negative 5 \
        -hs 0 \
        -sample 1e-4 \
        -threads 40 \
        -binary 0 \
        -iter 20 \
        -min-count 1 \
        -sentence-vectors 1

Take care not to play around with the options too much if you're not
sure what they do. As Tomas Mikolov [points out][remark] in the thread
where he published this patch, even main authors can make mistakes.

[remark]: https://groups.google.com/d/msg/word2vec-toolkit/Q49FIrNOQRo/kH1Ch0sWJwMJ

## License

See header of word2vec.c.
