import glob
import json
from os import dup
import random
import pandas as pd
import csv
import json
import re

from scr.utils import make_batch_list


def load_core_sents(
        core_sent_dir: str = 'batches_cx_specific/annotation_files_rawids',
        json_file='output/annotation_batches/1_annotation_data.json'
):
    filenames = glob.glob('{}/*.txt'.format(core_sent_dir))
    filenames.sort()

    # with open('batches_cx_specific/sentence_ids.json', 'r') as f:
    #     sid_sent_list = json.load(f)
    with open(json_file, 'r') as f:
        sid_sent_list = json.load(f)

    sid_sent_list.sort(key=lambda x: x[0])

    core_sentids = {}
    for sid, sent in sid_sent_list:
        core_sentids[sid] = sent
    return core_sentids


## databases
def load_all_sentences():
    with open("output/sentence_ids/BAWE_20220427.json", 'r') as f:
        BAWE = json.load(f)

    with open("output/sentence_ids/FCE_answer2_20220327.json", 'r') as f:
        FCE = json.load(f)

    with open("output/sentence_ids/ICNALE_20220327.json", 'r') as f:
        ICNALE = json.load(f)

    with open("output/sentence_ids/MICUSP_20220427.json", 'r') as f:
        MICUSP = json.load(f)

    with open("output/sentence_ids/TOEFL11_20220327.json", 'r') as f:
        TOEFL = json.load(f)

    all_sent_ids = BAWE + FCE + ICNALE + MICUSP + TOEFL

    sent_db = {}
    for sid, sent in all_sent_ids:
        sent_db[sid] = sent

    return sent_db


## first 2000 sentences
def dict_already_seen_sents(txtdir: str = '20220520/annotation_files_rawids'):
    firstbatches = glob.glob('{}/*.txt'.format(txtdir))
    firstbatches.sort()

    firstbatch_sentids = {}
    for file in firstbatches:
        with open(file, 'r') as f:
            for line in f.readlines():
                sentid, sent = line.split('\t')

                firstbatch_sentids[sentid] = sent
    return firstbatch_sentids


p1 = [None, None, "the", "the", "X", "the", "the", "the", None]
p1 = [4, 4, 1, 2, 0, 3, 4, 5, 6]

len(p1)


def optimal_window(pattern, k=2):
    cnum = int((len(pattern) + 1) / 2)
    if not None in pattern:
        cnum = int((len(pattern) + 1) / 2)
        return (pattern[cnum - (k + 1):cnum + k])

    else:
        none_ind_prev = [
            index for (index, item) in enumerate(pattern)
            if item == None and index < cnum
        ]
        none_ind_foll = [
            index for (index, item) in enumerate(pattern)
            if item == None and index > cnum
        ]

        newwindow = pattern[max(none_ind_prev) + 1:min(none_ind_foll)]
        print(newwindow)


optimal_window(p1, k=1)


### Tring to extract prev and next from already selected sentences.
def extract_contexts(sent_db: dict,
                     core_sentids: dict,
                     already_seen_sentids: dict,
                     fixed_window=True,
                     k=4):
    duplicates = []
    sent_sequence = {}

    for fullsid, sent in core_sentids.items():
        pattern1 = r"(.*?)\.xml_(|answer\d_)(\d+)_(\d+)"
        pattern2 = r"(.*?)\.xml_(|answer\d)s(\d+)\.(\d+)\;p(\d+)\.(\d+)"
        match = re.match(pattern1, fullsid)
        match2 = re.match(pattern2, fullsid)

        seen = {}
        if match:
            docid, task, pid, sid = match.group(1), match.group(
                2), match.group(3), int(match.group(4))

            ## Old algorithm used for Batch 2 contextual
            if fixed_window:
                if sid == 1:
                    sent1 = docid + ".xml_" + task + pid + "_" + str(sid + 1)
                    sent2 = docid + ".xml_" + task + pid + "_" + str(sid + 2)
                    sentids = (fullsid, sent1, sent2)

                else:
                    sent1 = docid + ".xml_" + task + pid + "_" + str(sid - 1)
                    sent2 = docid + ".xml_" + task + pid + "_" + str(sid + 1)
                    sentids = (sent1, fullsid, sent2)
            else:
                ## New algorithm
                window_ids = []

                for x in range(sid - k, sid + k):
                    sentid = docid + ".xml_" + task + pid + "_" + str(x)
                    try:
                        if sent_db[sentid]:
                            window_ids.append(sentid)
                    except KeyError:
                        window_ids.append(None)
                        continue
                continue

        elif match2:  #patterns for BAWE corpus
            docid, task, sid, smax, pid, pmax = match2.group(1), match2.group(
                2), int(match2.group(3)), match2.group(4), match2.group(
                    5), match2.group(6)
            # print(fullsid)
            # print(sid, smax, pid, pmax)

            if sid == 1:
                sent1 = "{}.xml_s{}.{};p{}.{}".format(docid, str(sid + 1),
                                                      smax, pid, pmax)
                sent2 = "{}.xml_s{}.{};p{}.{}".format(docid, str(sid + 2),
                                                      smax, pid, pmax)
                sentids = (fullsid, sent1, sent2)

            else:
                sent1 = "{}.xml_s{}.{};p{}.{}".format(docid, str(sid - 1),
                                                      smax, pid, pmax)
                sent2 = "{}.xml_s{}.{};p{}.{}".format(docid, str(sid + 1),
                                                      smax, pid, pmax)
                sentids = (sent1, fullsid, sent2)
        else:
            sentids = (fullsid)

        for sentid in sentids:  #Checking if any of the sentences are already in the dataset
            if sentid in already_seen_sentids:
                duplicates.append(sentid)

        holder = []
        for sentid in sentids:
            try:
                holder.append(sent_db[sentid])
            except KeyError:
                continue

        sent_sequence[fullsid] = "\n".join(holder)
    return sent_sequence, duplicates


import os


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def save_batches(sent_sequence: dict,
                 saving_dir: str = 'batches_cx_spec_w_contexts/textfiles'):
    make_dir(saving_dir)

    for sid, sent in sent_sequence.items():
        with open('{}/{}.txt'.format(saving_dir, sid), 'w') as f:
            f.write(sent)


def main():

    sent_database = load_all_sentences()

    core_sents = load_core_sents(
        core_sent_dir="annotation_files_rawids")
    already_seen_sents = dict_already_seen_sents(
        txtdir='batches_cx_spec_w_contexts/annotation_files_rawids')

    contextual_sequence, duplicates = extract_contexts(sent_database,
                                                       core_sents,
                                                       already_seen_sents)

    save_batches(contextual_sequence, 'batches_first_round/with_contexts')

    with open('batches_first_round/with_contexts/__duplicates_ids.tsv',
              'w') as f:
        for d in duplicates:
            f.write(d)
            f.write("\n")


if __name__ == "__main__":
    main()
