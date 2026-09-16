import re
import csv
import json
import os

import glob
import random


def conll2list(conllfile):
    holder = []
    for chunk in conllfile.split('\n\n'):
        temp = []
        for line in chunk.split("\n"):
            if len(line) < 1:
                continue
            elif "#" in line[0]:
                #print(line)
                pass
            else:
                temp.append(line)
        if len(temp) > 0:  #skip line with no sentences
            holder.append(temp)
    return holder


def line2dict(line: str, cols):
    l = {}
    items = line.strip().split("\t")

    for c, label in zip(cols, items):
        l[c] = label
    return l


col_def = ['tid', 'chid', 'token', 'tag', 'pos', 'clause', 'engmt', 'modal']

line2dict("1-1\t0-2\tIt\tPRP\tPRON\tMAIN[1]\t_\t_\t", col_def)


def conll2dict(conllfile: str):
    '''Takes conll file as string
    outputs 
    holder[chunkid] =  {'sentId': sent_id, 'Text': text, 'lines': temp}
    '''
    features = []
    cols = [
        'tid',
        'chid',
        'word',
    ]
    holder = {}
    for chunkid, chunk in enumerate(conllfile.split('\n\n')):
        temp = []
        sent_id = "NA"
        for line in chunk.split("\n"):
            if len(line) < 1:
                continue
            elif "#Sentence.id" in line:
                sent_id = re.match(r'#Sentence\.id=(.+)', line).group(1)
                print(sent_id)
            elif "#Text=" in line:
                text = re.match(r'#Text=(.+)', line).group(1)

            elif "#T_SP=" in line:  #for webanno tsv format
                if 'POS' in line:
                    features.append('POS')
                    cols.extend(['tag', 'pos'])
                elif 'Clause' in line:
                    features.append('Clause')
                    cols.extend(['clause'])
                elif 'Engagement' in line:
                    features.append('Engagement')
                    cols.extend(['engmt'])
                elif 'ModalSense' in line:
                    features.append('Modal')
                    cols.extend(['modal'])
            elif "#" in line[0]:
                pass
            else:
                temp.append(line2dict(line, cols))
        if len(temp) > 0:  #skip line with no sentences
            holder[chunkid] = {'sentId': sent_id, 'Text': text, 'lines': temp}
    return holder


def sent2iob(sent_lines: list,
             feature: str = 'engmt',
             col_size: int = 5,
             ignore_tags=['T-UNIT', 'QUOTED']):
    holder = []
    seen_tags = {}

    for l in sent_lines:
        tags = l[feature].split("|")
        if len(tags) > 1:
            print(tags)

        iob_tags = []

        for tag in tags:
            if any([re.search(t, tag) for t in ignore_tags]):
                continue
            pat = r"(.*)\[(\d+)\]"
            match = re.match(pat, tag)

            if match:  ## multiword tags
                if tag not in seen_tags:
                    seen_tags[tag] = None
                    ## strip number here
                    iob_tags.append("B-{}".format(match.group(1)))
                else:
                    ## strip number here
                    iob_tags.append("I-{}".format(match.group(1)))

            else:
                if tag not in ["_", "*"]:
                    iob_tags.append("B-{}".format(tag))
                else:
                    iob_tags.append("O")

        #Then store as string
        n_tags = len(iob_tags)
        for i in range(0, col_size - n_tags):
            iob_tags.append("O")

        holder.append("\t".join([l['word']] + iob_tags))
    return holder


# pat = r"(.*)\[(\d+)\]"
# match = re.match(pat, 'DENY[1]')

# if match:
#     print("yes")

# file1 = "data/Annotated_files_ALM/completed_archive/download-20220616/1C-Batch/2C_completed/batch1_C6_pos_ALM.tsv"


def tsv2sentlist(filename: str, feature: str = 'engmt'):

    holder = []

    conll_file = conll2dict(open(filename, 'r').read())
    for sid, sent in conll_file.items():
        holder.append(sent2iob(sent['lines'], feature))

    return (holder)


def write_file(output_name: str, data: list):
    with open(output_name, 'w') as f:

        for sent in data:
            for line in sent:
                f.write(line)
                f.write('\n')
            f.write('\n')


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


# The output would be
def Run(split: float = 0.9,
        feature: str = "engmt",
        input_dir='data/data_to_convert/20220903_firstbatch_AtoI',
        save_dir='data/iob_data/20220903_AtoI_not-adjudicated'):

    filenames = glob.glob('{}/*.tsv'.format(input_dir))

    holder = []
    for file in filenames:
        holder.extend(tsv2sentlist(file, feature))

    # random.seed(1993)
    random.seed(400)
    random.shuffle(holder)

    size = len(holder)

    train, dev = holder[0:int(size * split)], holder[int(size * split):]
    print("Training data: {} sentences; \n Develop data: {} sentences".format(
        len(train), len(dev)))

    write_file('{}/train.iob'.format(save_dir), train)
    write_file('{}/dev.iob'.format(save_dir), dev)


if __name__ == "__main__":
    input_dir = 'data/data_to_convert/20221007_clause'
    save_dir = 'data/iob_data/20221007_clause'
    make_dir(save_dir)
    Run(split=0.85, feature='clause', input_dir=input_dir, save_dir=save_dir)
