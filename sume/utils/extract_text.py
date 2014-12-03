#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import sys
import codecs

""" Extract the textual content from the DUC/TAC files.

    author: florian boudin (florian.boudin@univ-nantes.fr)
"""

def remove_byline(text):
    """ Remove the newswire byline from the textual content.

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
    text = re.sub(u'^[A-Z][\-\,\.\w\s]+ (\([A-Z]+\) )?(_|-|--) ', '', text)

    return text

# open the input file
with codecs.open(sys.argv[1], 'r', 'utf-8') as f:

    # read the entire file
    content = f.read()

    # extract the textual content
    m = re.search(u'(?is)<TEXT>(.+)</TEXT>', content)
    content = m.group(1)

    # remove the paragraph tags
    content = re.sub(u'(?i)</?p>', '', content)

    # remove annotation tags
    content = re.sub(u'(?i)<ANNOTATION>[^<]+</ANNOTATION>', '', content)

    # remove the HTML entities
    content = re.sub(u'(?i)&amp;', '&', content)
    content = re.sub(u'(?i)&quot;', '"', content)
    content = re.sub(u'(?i)&apos;', "'", content)
    content = re.sub(u'(?i)&lt;', "<", content)
    content = re.sub(u'(?i)&gt;', ">", content)
    content = re.sub(u'&\w+;', "", content)

    # remove extra spacing
    content = re.sub(u'\s+', ' ', content.strip())

    # remove byline from the first 80 characters
    header = remove_byline(content[:80])
    content = header + content[80:]

    prev_content = content

    # normalize the quotation marks
    content = re.sub(u'```', '"`', content)
    content = re.sub(u'``', '"', content)
    content = re.sub(u"'''", '\'"', content)
    content = re.sub(u"''", '"', content)
    content = re.sub(u"[”“]", '"', content)
    content = re.sub(u'(^|[ :;()])\"([^\"])', '\g<1>``\g<2>', content)
    content = re.sub(u'([^\"])\"($|[ :;()])', '\g<1>\'\'\g<2>', content)

    # count the quotation marks
    # opening_quotation_marks = re.findall(u'``', content)
    # ending_quotation_marks = re.findall(u"''", content)

    # write the extracted textual content into a file
    with codecs.open(sys.argv[2], 'w', 'utf-8') as w:
        w.write(content)




