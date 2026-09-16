#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import csv
import json
from scr.utils import make_batch_list

import spacy

nlp = spacy.load('en_core_web_trf', disable=['ner'])


def preprocess(sent: str):
    sent = sent.replace("\n", " ")
    sent = sent.replace("  ", " ")
    return sent


def spacy_tokenize(sent: str):
    '''This creates spacy tokenization for the training data.'''
    sent = preprocess(sent)
    doc = nlp(sent)
    tokenized = [t.text for t in doc]
    return tokenized


def spacy_tokenize_tsv(sent: str,
                       sent_id: str,
                       prev_last_char: int,
                       pos=False,
                       dep=False,
                       dep_list=None):
    sent = preprocess(sent)
    doc = nlp(sent)
    holder = []
    last_char = copy.deepcopy(prev_last_char)
    for t in doc:
        tokenids = "{}-{}".format(sent_id, str(t.i + 1))  #+1
        charids = "{}-{}".format(t.idx + last_char,
                                 t.idx + len(t.text) + last_char)
        if pos and dep:
            if dep_list == None:
                holder.append([
                    tokenids, charids, t.text, t.tag_, t.pos_, "_", "_", "_",
                    t.dep_, 'basic', '{}-{}'.format(sent_id, t.head.i + 1)
                ])
            else:
                if t.dep_ in dep_list:
                    holder.append([
                        tokenids, charids, t.text, t.tag_, t.pos_, "_", "_",
                        "_", t.dep_, 'basic',
                        '{}-{}'.format(sent_id, t.head.i + 1)
                    ])
                else:
                    holder.append([
                        tokenids, charids, t.text, t.tag_, t.pos_, "_", "_",
                        "_", "_", "_", "_"
                    ])
        elif pos:
            holder.append(
                [tokenids, charids, t.text, t.tag_, t.pos_, "_", "_", "_"])
        else:
            holder.append([tokenids, charids, t.text, "_", "_", "_"])  #

    last_char += t.idx + len(t.text) + 1
    return (holder, last_char)



tsv_head = '''#FORMAT=WebAnno TSV 3.3
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement
#T_SP=webanno.custom.ModalSenseDisambiguation|ModalSense'''

tsv_head_pos = '''#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement
#T_SP=webanno.custom.ModalSenseDisambiguation|ModalSense'''

tsv_head_pos_dep = '''#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement
#T_SP=webanno.custom.ModalSenseDisambiguation|ModalSense
#T_RL=de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency|DependencyType|flavor|BT_de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS'''


def write_webanno_tsv(batch,
                      outfile,
                      header: str = tsv_head,
                      pos=False,
                      dep=False,
                      dep_list=None):
    outfile.write(header)
    outfile.write("\n")
    last_char = 0
    for sid, sent_info in enumerate(batch, start=1):
        meta, sent = sent_info
        sent = preprocess(sent)
        # Some sentences are empty, and has to be escaped
        if len(sent) < 1:
            print(sent)
            outfile.write("\n\n#Sentence.id={}".format(meta))
            #this has caused a display problem in webanno
            # outfile.write("\n#Text={}\t{}".format(sent, "SKIPANNOTATION"))
            # token_list, last_char = spacy_tokenize_tsv(
            #     "{}\t{}".format(sent, "SKIPANNOTATION"), sid, last_char, pos,
            #     dep, dep_list)
            outfile.write("\n#Text={}".format("SKIPANNOTATION"))
            token_list, last_char = spacy_tokenize_tsv(
                "{}".format("SKIPANNOTATION"), sid, last_char, pos, dep,
                dep_list)

        else:
            outfile.write("\n\n#Sentence.id={}".format(meta))
            outfile.write("\n#Text={}".format(sent))
            token_list, last_char = spacy_tokenize_tsv(sent, sid, last_char,
                                                       pos, dep, dep_list)

        for t in token_list:
            outfile.write("\n")
            outfile.write("\t".join(t) + "\t")


file = 'output/annotation_batches/1_annotation_data.json'

# the following does not use csv anymore because of the quoting issue in the output file.
with open(file, 'r') as f:
    sent_list = json.load(f)

    batches = make_batch_list(sent_list, batch_size=20)
    ch = 'A'
    ch = bytes(ch, 'utf-8')

    for fid, batch in enumerate(batches, start=0):
        if fid % 10 == 0 and fid != 0:
            # converting character to byte
            ch = bytes([ch[0] + 1])

        with open("annotation_files_tsv/batch1_{}{}.tsv".format(
                ch.decode("utf-8"),
                str(fid)[-1]),
                  "w",
                  encoding='utf-8') as f:
            write_webanno_tsv(batch, f, tsv_head)

        with open("annotation_files_tsv_pos/batch1_{}{}_pos.tsv".format(
                ch.decode("utf-8"),
                str(fid)[-1]),
                  "w",
                  encoding='utf-8') as f:
            write_webanno_tsv(batch, f, tsv_head_pos, pos=True, dep=False)

        with open("annotation_files_tsv_pos_dep/batch1_{}{}_pos.tsv".format(
                ch.decode("utf-8"),
                str(fid)[-1]),
                  "w",
                  encoding='utf-8') as f:
            write_webanno_tsv(batch, f, tsv_head_pos_dep, pos=True, dep=True)

        with open("annotation_files_tsv_pos_dep2/batch1_{}{}_pos.tsv".format(
                ch.decode("utf-8"),
                str(fid)[-1]),
                  "w",
                  encoding='utf-8') as f:
            write_webanno_tsv(batch,
                              f,
                              tsv_head_pos_dep,
                              pos=True,
                              dep=True,
                              dep_list=[
                                  "ROOT",
                                  "advcl",
                                  "appos",
                                  "relcl",
                                  "acl",
                                  "ccomp",
                                  'nsubj',
                                  "nsubjpass",
                                  "cc",
                                  "mark",
                                  "parataxis",
                                  "prep",
                                  "xcomp",
                                  "conj",
                                  "dep",
                              ])

        # id + tokenized text
        with open('annotation_files_rawids/batch1_{}{}_withids.txt'.format(
                ch.decode("utf-8"),
                str(fid)[-1]),
                  'w',
                  encoding='utf-8') as f:
            #tsv_writer = csv.writer(f, delimiter = "\t", escapechar='\\', quoting = csv.QUOTE_NONE)
            for id, sent in batch:
                # tsv_writer.writerow([id,str(spacy_tokenize(sent))])
                f.write("\t".join(
                    [str(id), str(" ".join(spacy_tokenize(sent)))]))
                f.write("\n")

        # tokenized text only
        with open('annotation_files_raw/batch1_{}{}.txt'.format(
                ch.decode("utf-8"),
                str(fid)[-1]),
                  'w',
                  encoding='utf-8') as f:
            #tsv_writer = csv.writer(f, delimiter = "\t", escapechar='\\', quoting = csv.QUOTE_NONE)
            for id, sent in batch:
                # f.write(str(spacy_tokenize(sent)))
                f.write(str(" ".join(spacy_tokenize(sent))))
                f.write("\n")
            # try:
            #	tsv_writer.writerow([str(spacy_tokenize(sent))])
            # except csv.Error:
            #	tsv_writer.writerow([" "])
