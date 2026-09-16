import random
from bs4 import BeautifulSoup as bs
import lxml
import re
import csv
import glob
import copy
import json

import spacy

from scr.utils import make_batch_list


def extract_sents(filename: str, sentids: dict):
    holder = []
    fileid = filename.split("/")[-1]

    with open(filename, "r") as f:
        # Read each line in the file, readlines() returns a list of lines
        content = f.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        soup = bs(content, "lxml")

        if 'fce-released' in filename:
            answer1 = soup.answer1
            if answer1 != None:
                for s in soup.find_all('s'):
                    pid = s['pid']
                    sid = s['sid']
                    key = fileid + "_answer1_" + pid + "_" + sid
                    if key in sentids:
                        #holder.append((key, s.text.strip()))
                        pass
            answer2 = soup.answer2
            if answer2 != None:
                for s in soup.find_all('s'):
                    pid = s['pid']
                    sid = s['sid']
                    key = fileid + "_answer2_" + pid + "_" + sid

                    if key in sentids:
                        holder.append((key, s.text.strip()))

        else:
            for s in soup.find_all('s'):
                pid = s['pid']
                sid = s['sid']
                key = fileid + "_" + pid + "_" + sid

                if key in sentids:
                    holder.append((key, s.text.strip()))

    return (holder)


def extract_BAWE_sentid(filename):
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

            holder.append(fileid + "_" + sentid)
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


def sentence_batch(Corpus_name, batch_size):
    corpus_dirs = {
        'BAWE': 'corpus_data/BAWE/CORPUS_UTF-8',
        'MICUSP': "corpus_data/MICUSP_scraped_1.0_20211018/sent_tokenized_xml2",
        'FCE_answer2': 'corpus_data/fce-released-dataset/sentence_segmented_xml',
        'TOEFL11':
        'corpus_data/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized_xml',
        'ICNALE': 'corpus_data/ICNALE_Edited Essays_2.1/sent_tokenized_xml'
    }

    date_suffix = {
        'BAWE': '20220427', 'MICUSP': '20220427',
        'FCE_answer2': '20220327', 'TOEFL11': '20220327', 'ICNALE': '20220327'
    }
    with open("output/sentence_ids/{}_{}_shuffled.json".format(Corpus_name, date_suffix[Corpus_name]),
              'r') as f:
        sent_list = json.load(f)

    #sentid_dict = dict.fromkeys(sentid_list)
    #corpus = glob.glob(corpus_dirs[Corpus_name] + "/*.xml")
    batch = make_batch_list(sent_list, batch_size=batch_size)

    with open("output/sentence_ids/batches/{}_batch.json".format(Corpus_name),
              'w') as f:
        json.dump(batch, f, indent=2)


if __name__ == "__main__":
    sentence_batch("BAWE", 750)
    sentence_batch("MICUSP", 750)
    sentence_batch("FCE_answer2", 150)
    sentence_batch("TOEFL11", 200)
    sentence_batch("ICNALE", 150)
