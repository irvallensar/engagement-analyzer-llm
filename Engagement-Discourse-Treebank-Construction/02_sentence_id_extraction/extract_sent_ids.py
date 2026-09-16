#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:06:10 2021

@author: masakieguchi

the script randomly samples sentence ids from xml file, using the 
sid and pid attributes from xml
"""

import random
from bs4 import BeautifulSoup as bs
import lxml
import re
import csv
import glob
import copy
import json

test_ICNALE = '1_corpora/2_learner_corpora/ICNALE_Edited Essays_2.1/sent_tokenized_xml/W_THA_PTJ0_008_A2_0_ORIG.xml'
test_MICUSP = '1_corpora/1_reference/MICUSP_scraped_1.0_20211018/sent_tokenized_xml/BIO.G0.10.2.xml'


def extract_pid_sid(filename, id_only=True):
    holder = []
    fileid = filename.split("/")[-1]

    with open(filename, "r") as f:
        # Read each line in the file, readlines() returns a list of lines
        content = f.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        soup = bs(content, "lxml")
        print(soup)

        if 'fce-released' in filename:
            answer1 = soup.answer1
            if answer1 != None:
                for s in soup.find_all('s'):
                    pid = s['pid']
                    sid = s['sid']
                    #holder.append(fileid + "_answer1_" + pid + "_" + sid)

            answer2 = soup.answer2
            if answer2 != None:
                for s in soup.find_all('s'):
                    pid = s['pid']
                    sid = s['sid']
                    if id_only:
                        holder.append(fileid + "_answer2_" + pid + "_" + sid)
                    else:
                        holder.append((fileid + "_answer2_" + pid + "_" + sid,
                                       s.text.strip()))

        else:
            for s in soup.find_all('s'):
                pid = s['pid']
                sid = s['sid']
                if id_only:
                    holder.append(fileid + "_" + pid + "_" + sid)
                else:
                    holder.append(
                        (fileid + "_" + pid + "_" + sid, s.text.strip()))
    return (holder)


def extract_BAWE_sentid(filename, id_only=True):
    holder = []
    fileid = filename.split("/")[-1]

    with open(filename, "r") as f:
        # Read each line in the file, readlines() returns a list of lines
        content = f.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        soup = bs(content, "lxml")
        for s in soup.find_all('s'):
            sentid = s['n']
            if ";pn" in str(sentid):  #avoid extracting footnotes
                continue
            if id_only:
                holder.append(fileid + "_" + sentid)
            else:
                holder.append((fileid + "_" + sentid, s.text.strip()))
    return (holder)


def make_list(files: list, id_only=True):
    list_sents = []
    for file in files:
        if "BAWE" in file:
            list_sents.extend(extract_BAWE_sentid(file, id_only=id_only))
        else:
            list_sents.extend(extract_pid_sid(file, id_only=id_only))

    return (list_sents)


def save_random(ids: list, save_dir: str, save_name: str, seed=1993):

    with open(save_dir + save_name + ".json", 'w') as f:
        json.dump(ids, f, indent=2)

    ids.sort()
    random.seed(seed)
    random.shuffle(ids)

    with open(save_dir + save_name + "_shuffled.json", 'w') as f:
        json.dump(ids, f, indent=2)


# =============================================================================
# random shuffle of ICNALE sentences
# =============================================================================
icnale_files = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/ICNALE_Edited Essays_2.1/sent_tokenized_xml/*.xml"
)
icnale_sent_list = make_list(icnale_files, False)

save_random(
    icnale_sent_list,
    "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_corpus_sampling/sentence_ids/",
    'ICNALE_20220327')

# =============================================================================
# random shuffle of MICUSP sentences
# =============================================================================
micusp_files = glob.glob(
    "corpus_data/MICUSP_scraped_1.0_20211018/sent_tokenized_xml2/*.xml")
micusp_sent_list = make_list(micusp_files, id_only=False)
save_random(micusp_sent_list, "sentence_ids/", 'MICUSP_20220427')

len(micusp_sent_list)

# =============================================================================
# random shuffle of BAWE corpos
# =============================================================================
bawe_files = glob.glob("corpus_data/BAWE/CORPUS_UTF-8/*.xml")
bawe_sent_list = make_list(bawe_files, id_only=False)
save_random(bawe_sent_list, "sentence_ids/", 'BAWE_20220427')

# =============================================================================
# random shuffle of TOEFL11 sentences
# =============================================================================

toefl11 = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized_xml/*.xml"
)
toefl_sent_list = make_list(toefl11, id_only=False)
save_random(
    toefl_sent_list,
    "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_corpus_sampling/sentence_ids/",
    'TOEFL11_20220327')

FCE = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/fce-released-dataset/sentence_segmented_xml/*.xml"
)
FCE_sent_list = make_list(FCE, id_only=False)
save_random(
    FCE_sent_list,
    "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_corpus_sampling/sentence_ids/",
    'FCE_answer2_20220327')

### From extract sent from xml.py


def extract_sents(filename: str, sentids: dict):
    holder = []
    fileid = filename.split("/")[-1]

    with open(filename, "r") as f:
        # Read each line in the file, readlines() returns a list of lines
        content = f.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        soup = bs(content, "lxml")
        for s in soup.find_all('s'):
            pid = s['pid']
            sid = s['sid']
            key = fileid + "_" + pid + "_" + sid

            if key in sentids:
                holder.append((key, s.text.strip()))

    return (holder)


def main(files: list, sentids: dict):
    list_sents = []
    for file in files:
        if "BAWE" in file:
            continue
            #list_sents.extend(extract_BAWE_sents(file, sentids))
        else:
            list_sents.extend(extract_sents(file, sentids))
    return (list_sents)


# =============================================================================
# ICNALE n = 200 to start with
# =============================================================================

with open(
        "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_annotation_data_prep/shuffled_sent_id/shuffled_ICNALE_sent_list.json",
        'r') as f:
    sent_list = json.load(f)

# entire sample
icnale_dict = dict.fromkeys(sent_list)
len(icnale_dict)

icnale_dict = dict.fromkeys(sent_list[:200])
len(icnale_dict)
icnale_files = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/ICNALE_Edited Essays_2.1/sent_tokenized_xml/*.xml"
)
icnale_sents = main(icnale_files, icnale_dict)

#W_PHL_SMK0_123_B1_1_ORIG
icnale_dict = dict.fromkeys(sent_list[200:400])
len(icnale_dict)

icnale_files = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/ICNALE_Edited Essays_2.1/sent_tokenized_xml/*.xml"
)
icnale_sents = main(icnale_files, icnale_dict)

# =============================================================================
# MICUSP n = 400 to start with
# =============================================================================

with open(
        "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_annotation_data_prep/shuffled_sent_id/shuffled_MICUSP_sent_list.json",
        'r') as f:
    micusp_list = json.load(f)

#entire sample

micusp_dict = dict.fromkeys(micusp_list)
len(micusp_dict)

micusp_dict = dict.fromkeys(micusp_list[:400])
len(micusp_dict)

micusp_dict = dict.fromkeys(micusp_list[400:800])
len(micusp_dict)

micusp_files = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/1_reference/MICUSP_scraped_1.0_20211018/sent_tokenized_xml/*.xml"
)
micusp_sents = main(micusp_files, micusp_dict)
len(micusp_sents)

# =============================================================================
# TOEFL n = 400 to start with
# =============================================================================

with open(
        "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_annotation_data_prep/shuffled_sent_id/shuffled_TOEFL_sent_list.json",
        'r') as f:
    toefl_list = json.load(f)

#entire sample
toefl_dict = dict.fromkeys(toefl_list)
len(toefl_dict)

toefl_dict = dict.fromkeys(toefl_list[:400])
len(toefl_dict)

# micusp_dict = dict.fromkeys(micusp_list[400:800])
# len(micusp_dict)

toefl_files = glob.glob(
    "/Users/masakieguchi/Dropbox/0_Projects/1_corpora/2_learner_corpora/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized_xml/*.xml"
)
toefl_sents = main(toefl_files, toefl_dict)
len(toefl_sents)

# =============================================================================
# BAWE
# =============================================================================

with open(
        "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_annotation_data_prep/shuffled_sent_id/shuffled_BAWE_sent_list.json",
        'r') as f:
    bawe_list = json.load(f)

bawe_dict = dict.fromkeys(bawe_list)
len(bawe_dict)

# =============================================================================
# creating XML for annotation
# =============================================================================


def make_batch_list(sent_list: list, batch_size: int = 20):
    """ Returns two-level nested list, each for a batch of filenames.
	
	Parameters
	----------
	filenames : list
		The entire text filenames.
	batch_size : int, optional
		The number of documents for each batch. The default is 20

	Returns
	-------
	nested list of filenames.

	"""
    batches = []
    batch_count = 1
    current_batch = []
    for x, sent in enumerate(sent_list, start=1):
        if batch_count < batch_size:
            current_batch.append(sent)
            batch_count += 1
            print(x)
        else:
            current_batch.append(sent)
            batches.append(current_batch)
            print(x)
            batch_count = 1
            current_batch = []
            print('Reset')
    if len(current_batch) > 0:
        batches.append(current_batch)  #this is the leftovers
    return (batches)


template = '''
<sentences>
</sentences>
'''

saving_dir = "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_annotation_data_prep/TOEFL_data/"


def output_xml(sent_list, name, batch: int = 20):

    with open(saving_dir + name + ".xml", "w") as outf:

        soup = bs(template, features='xml')  #create an empty xml
        sentences = soup.sentences

        for sentid, sent in sent_list:
            s_tag = soup.new_tag("s",
                                 id=sentid,
                                 engmt_main="",
                                 engmt_sec="",
                                 tunit="")
            s_tag.string = sent
            print(s_tag)
            sentences.append(s_tag)

        outf.write(soup.prettify())


def output_txt(sent_list, name, batch: int = 20):
    with open(saving_dir + name + "_sent.txt", "w") as outf:
        tsv_writer = csv.writer(outf, delimiter="\t")
        for sentid, sent in sent_list:
            tsv_writer.writerow([str(sent)])


def write_output(batches_list, corpus, start=1):
    for n, batch in enumerate(batches_list, start=start):
        output_xml(batch, "{}_{}".format(corpus, n))
        output_txt(batch, "{}_{}".format(corpus, n))


# ICNALE
icnale_batch = make_batch_list(icnale_sents)
write_output(icnale_batch, "ICNALE", 11)

## Micusp
micusp_batch = make_batch_list(micusp_sents)
write_output(micusp_batch, "MICUSP", 21)

## TOEFL
batch = make_batch_list(toefl_sents)
write_output(batch, "TOEFL", 1)
