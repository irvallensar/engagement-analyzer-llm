#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:06:10 2021

@author: masakieguchi
"""

from bs4 import BeautifulSoup as bs
import lxml
import re
import csv
import glob

#stanza
import stanza

example = "The capabilities of a processing pipeline always depend on the components, their models and how they were trained. For example, a pipeline for named entity recognition needs to include a trained named entity recognizer component with a statistical model and weights that enable it to make predictions of entity labels. "
doc = nlp(example)
for sent in doc.sents:
    print(sent.text)


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


def preprocess(text):
    # spaces
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

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
    text = text.replace("&amp;", "&")
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
    text = text.replace("&quot;", '"')
    text = text.replace("&apos;", "'")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    return (text)


def read_MICUSP(xml_file: str):
    output = []
    with open(xml_file, "r") as file:
        # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        bs_content = bs(content, "lxml")
        body = bs_content.find("div", class_='dynacloud')
        for p in body.find_all('p'):
            text = p.text.replace("\n", "")
            text = preprocess(text)
            text = minor_preprocess(text)
            output.append(text)
    return output


stnz = stanza.Pipeline(lang='en', processors='tokenize')


def stanza_tokenize(text):
    output = []
    doc = stnz(text)
    for idx, sent in enumerate(doc.sentences):
        output.append(sent.text)
    return output


def tokenize_rewrite_MICUSP(xml_file: str):
    with open(xml_file, "r") as file:
        # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        bs_content = bs(content, "lxml")  #parse into soup

        ### unique to MICUSP from here
        body = bs_content.find(
            "div",
            class_='dynacloud')  #find the body of the paper by dynacloud

        for pid, p in enumerate(body.find_all('p'),
                                start=1):  #for each paragraph
            #conduct preprocessing
            text = str(p)
            #print(text)
            text = replace_tags(text)
            #text = p.text.replace("\n", "")
            text = preprocess(text)
            #print(text)

            p.string = ""  #empty the p so that we can add tokenized version
            #stanza_tokenize will conduct neural parsing
            for idx, sent in enumerate(stanza_tokenize(text), start=1):
                #set a new tag for each sentence
                new_div = bs_content.new_tag("s", pid=str(pid),
                                             sid=str(idx))  #
                new_div.string = sent  #set the sentence as string of the new tag
                p.append(new_div)  #update the soup
    return bs_content


print(tokenize_rewrite_MICUSP(test_file).prettify())
# =============================================================================
# conduct fixing from here.
# =============================================================================
xmlfiles = glob.glob("data/MICUSP_scraped_1.0_20211018/raw_xmls/*.xml")
output_dir = "data/MICUSP_scraped_1.0_20211018/sent_test/"

for ids, xml in enumerate(xmlfiles):
    filename = xml.split('/')[-1]
    print("Processing {} out of {}: {}".format(str(ids), str(len(xmlfiles)),
                                               filename))

    with open(output_dir + filename, 'w') as f:
        f.write(tokenize_rewrite_MICUSP(xml).prettify())

test_file = "data/MICUSP_scraped_1.0_20211018/raw_xmls/BIO.G0.27.1.xml"
test_file = "data/MICUSP_scraped_1.0_20211018/raw_xmls/ENG.G0.54.1.xml"
test_file = "data/MICUSP_scraped_1.0_20211018/raw_xmls/HIS.G1.04.1.xml"
test_file = "data/MICUSP_scraped_1.0_20211018/raw_xmls/BIO.G0.01.1.xml"
test_file = "data/MICUSP_scraped_1.0_20211018/raw_xmls/SOC.G3.10.3.xml"
test_file = "data/MICUSP_scraped_1.0_20211018/raw_xmls/SOC.G1.10.2.xml"

testtext = tokenize_rewrite_MICUSP(test_file)
for x in testtext:
    print(x)
