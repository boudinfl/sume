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

"""Extract the textual content from the DUC/TAC files."""

import argparse
import codecs
import re


def remove_byline(text):
    """Remove the newswire byline from the textual content.

    Examples of headers are:
        WASHINGTON _
        NEW YORK _
        AMHERST, N.Y. _
        DAR ES SALAAM, Tanzania _
        LAUSANNE, Switzerland (AP) _
        SEOUL, South Korea (AP) _
        BRUSSELS, Belgium (AP) -
        INNSBRUCK, Austria (AP) --
        PORT-AU-PRINCE, Haiti (AP) _
        BEIJING &UR; &LR; _
    """
    text = re.sub(r'^[A-Z][\-\,\.\w\s]+ (\([A-Z]+\) )?(_|-|--) ', '', text)

    return text


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Extract the textual content from the DUC/TAC files.')
    parser.add_argument('input', help='input file path')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args()

    # open the input file
    with codecs.open(args.input, 'r', 'utf-8') as f:

        # read the entire file
        content = f.read()

        # extract the textual content
        m = re.search(r'(?is)<TEXT>(.+)</TEXT>', content)
        content = m.group(1)

        # remove the paragraph tags
        content = re.sub(r'(?i)</?p>', '', content)

        # remove annotation tags
        content = re.sub(r'(?i)<ANNOTATION>[^<]+</ANNOTATION>', '', content)

        # remove the HTML entities
        content = re.sub(r'(?i)&amp;', '&', content)
        content = re.sub(r'(?i)&quot;', '"', content)
        content = re.sub(r'(?i)&apos;', "'", content)
        content = re.sub(r'(?i)&lt;', "<", content)
        content = re.sub(r'(?i)&gt;', ">", content)
        content = re.sub(r'&\w+;', "", content)

        # remove extra spacing
        content = re.sub(r'\s+', ' ', content.strip())

        # remove byline from the first 80 characters
        header = remove_byline(content[:80])
        content = header + content[80:]

        # normalize the quotation marks
        content = re.sub(r'```', '"`', content)
        content = re.sub(r'``', '"', content)
        content = re.sub(r"'''", '\'"', content)
        content = re.sub(r"''", '"', content)
        content = re.sub(r"[”“]", '"', content)
        content = re.sub(r'(^|[ :;()])\"([^\"])', '\g<1>``\g<2>', content)
        content = re.sub(r'([^\"])\"($|[ :;()])', '\g<1>\'\'\g<2>', content)

        # count the quotation marks
        # opening_quotation_marks = re.findall(u'``', content)
        # ending_quotation_marks = re.findall(u"''", content)

        # write the extracted textual content into a file
        with codecs.open(args.output, 'w', 'utf-8') as w:
            w.write(content)


if __name__ == '__main__':
    main()
