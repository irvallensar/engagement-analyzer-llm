import glob
import re
import pprint as pp
import os

from scripts.tsv_related.tsv2dict import conll2dict

filenames = glob.glob("data/2_test_set/*.tsv")

header = '''#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement|Hierarchy
#T_SP=webanno.custom.Supplementary|Supplementaryrhetoricalmoves'''

separater = '''#Text=--- Para SEP ---'''
skip = '''#Text=SKIPANNOTATION'''

for file in filenames:

    para = []
    holder = []

    text = open(file, 'r').read()

    chunks = re.split("\n\n|\n\n\n\n", text)

    for chunk in chunks[1:]:

        if skip in chunk:
            continue
        elif separater not in chunk:
            holder.append(chunk)
        else:
            if len(holder) > 0:
                para.append(holder)
            holder = []

    if len(holder) > 0:
        para.append(holder)

    # print(chunk)

    for i, p in enumerate(para, start=1):
        shortname = os.path.split(file)[-1]
        with open(f"data/3_test_set_separated/{shortname}" + str(i) + ".tsv",
                  'w') as f:
            f.write(header)
            f.write("\n\n")
            f.write("\n\n".join(p))

    three_sents = []

    for i, p in enumerate(para, start=1):
        holder = []
        for ix, s in enumerate(p, start=1):
            if ix % 3 == 1 and len(holder) > 0:
                three_sents.append(holder)
                holder = []
            holder.append(s)

        three_sents.append(holder)

    for cid, c in enumerate(three_sents):
        print(cid)
        print(c)
        print()
        shortname = os.path.split(file)[-1]
        with open(
                f"data/3_test_set_three_sents/{shortname}" + str(cid) + ".tsv",
                'w') as f:
            f.write(header)
            f.write("\n\n")
            f.write("\n\n".join(c))
