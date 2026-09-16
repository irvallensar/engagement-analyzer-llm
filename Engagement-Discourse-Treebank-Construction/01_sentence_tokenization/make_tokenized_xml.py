#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:16:04 2021

@author: masakieguchi
Previously called: sent_token_raw_to_xml.py

THis code will create sentence tokenized xmls
THis code should work for:
- TOEFL11 data
- ICNALE data

Should be applied to:
- FCE
- MICUSP
- 


"""

from bs4 import BeautifulSoup as bs
import lxml
import re
import csv
import glob

#stanza
import stanza

stnz = stanza.Pipeline(lang='en', processors='tokenize')


def preprocess(text):
    # spaces
    text = text.strip()
    while "-\n" in text:
        text = text.replace("-\n", " ")
    while "\n" in text:
        text = text.replace("\n", " ")
    while "\t" in text:
        text = text.replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    while '"{{ ' in text:
        text = text.replace('"{{ ', '"')
    while ' "}}' in text:
        text = text.replace(' "}}', '"')
    while "*%*]" in text:
        text = text.replace(' *%*]', ']')
    while ";" in text:
        text = text.replace(";", ".")
    while ":" in text:
        text = text.replace(":", ".")
    return (text)


def minor_preprocess(text):
    # spaces
    text = text.strip()
    text = text.replace("`` ", '"')
    text = text.replace(" ''", '"')
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" ?", "?")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = text.replace(" !", "!")
    text = text.replace("&amp;", "&")
    return (text)


def stanza_tokenize(text):
    output = []
    doc = stnz(text)
    for idx, sent in enumerate(doc.sentences):
        output.append(sent.text)
    return output


def replace_tags(text):
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace("<i>", '')
    text = text.replace("</i>", '')
    text = text.replace("<u>", '')
    text = text.replace("</u>", '')
    text = text.replace("<b>", '')
    text = text.replace("</b>", '')
    text = text.replace("<br/>", '\n')
    text = text.replace("<q>", '"{{')
    text = text.replace("</q>", '"}}')
    pattern = r"\<a.*?\>"
    text = re.sub(pattern, '[#REF ', text)
    text = text.replace("</a>", '*%*]')

    return (text)


def read_by_paragraphs(filename: str, encoding='utf-8', delim="\n"):
    with open(filename, "r", encoding=encoding) as file:
        paras = file.read().split(delim)
        return paras


template = '''
<text>
	<meta>
		<filename></filename>
	</meta>
	<body>
	</body>
</text>
'''
bs(template, features='xml')


def tokenize_rewrite(filename: str, stanza=True, metadata={}):

    ps = read_by_paragraphs(filename, delim="\n\n")  # "\n\n" for TOEFL dataset
    new_filename = filename.split("/")[-1]

    #print(new_filename)

    soup = bs(template, features='xml')  #create an empty xml

    filename = soup.meta.filename
    filename.string = str(new_filename[:-4])

    if new_filename in metadata:
        meta_section = soup.meta
        for name, data in metadata[new_filename].items():
            m = soup.new_tag(name)
            m.string = str(data)
            meta_section.append(m)

    #print(soup.prettify())

    original_body = soup.body

    for pid, para in enumerate(ps, start=1):  #for each paragraph
        p = soup.new_tag("p", pid=pid)
        #conduct preprocessing
        text = str(para)
        text = replace_tags(text)
        #text = p.text.replace("\n", "")
        #text = preprocess(text)
        text = minor_preprocess(text)

        # p.string = "" #empty the p so that we can add tokenized version
        if stanza:
            sents = stanza_tokenize(text)
        else:
            sents = text.split("\n")

        #stanza_tokenize will conduct neural parsing
        for idx, sent in enumerate(sents, start=1):
            #set a new tag for each sentence
            new_div = soup.new_tag("s", pid=str(pid), sid=str(idx))  #
            new_div.string = sent  #set the sentence as string of the new tag
            p.append(new_div)  #update the soup
        original_body.append(p)

    #soup.append(bodytag)
    return soup


tokenize_rewrite(test)

inputfiles = glob.glob(
    "1_corpora/2_learner_corpora/ICNALE_Edited Essays_2.1/EE_Unmerged_Unclassified/*_ORIG.txt"
)
output_dir = "1_corpora/2_learner_corpora/ICNALE_Edited Essays_2.1/sent_tokenized_xml/"

test = "/Users/masakieguchi/Desktop/W_TWN_SMK0_088_B2_0_ORIG.txt"

for n, file in enumerate(inputfiles, start=1):
    print("Processing {} out of {} files".format(n, str(len(inputfiles))))
    new_filename = file.split("/")[-1][:-4]
    with open(output_dir + new_filename + ".xml", 'w') as outf:
        outf.write(tokenize_rewrite(file).prettify())

## TOEFL 11
metadata = {}
with open(
        '/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/index-training-dev.csv',
        'r') as f:
    for l in f.readlines():
        file, prompt, l1, score = l.split(",")
        metadata[file] = {"prompt": prompt, "L1": l1, "score": score}

inputfiles = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized/*.txt"
)
output_dir = "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized_xml/"

for n, file in enumerate(inputfiles, start=1):
    print("Processing {} out of {} files".format(n, str(len(inputfiles))))
    new_filename = file.split("/")[-1][:-4]
    with open(output_dir + new_filename + ".xml", 'w') as outf:
        outf.write(
            tokenize_rewrite(file, stanza=False, metadata=metadata).prettify())

# MICUSP

inputfiles = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_corpus_sampling/data/MICUSP_scraped_1.0_20211018/raw_xmls/*.xml"
)
output_dir = "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_corpus_sampling/data/MICUSP_scraped_1.0_20211018/sent_test/"

test = "/Users/masakieguchi/Desktop/W_TWN_SMK0_088_B2_0_ORIG.txt"

for n, file in enumerate(inputfiles, start=1):
    print("Processing {} out of {} files".format(n, str(len(inputfiles))))
    new_filename = file.split("/")[-1][:-4]
    with open(output_dir + new_filename + ".xml", 'w') as outf:
        outf.write(tokenize_rewrite(file).prettify())
