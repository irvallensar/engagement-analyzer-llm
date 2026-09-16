import pprint as pp
import copy
import os
import glob
import re
from scripts.tsv_related.tsv2dict import conll2dict


def make_dir(path) -> None:
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def dir_renamer(path: str, dir_name: str) -> str:

    path_elem = path.split(os.path.sep)
    new_path_elem = copy.deepcopy(path_elem)
    new_path_elem[1] = dir_name
    new_path = os.path.join(*new_path_elem)
    return (new_path)


def clean_tag(tag_list: list,
              seen_tags: dict,
              replacement: dict,
              replace=False) -> list:
    '''
    This is a re-implementation of Aaron's code to replace empty tag with zero.
    '''
    new_list: list = []

    for tag in tag_list:
        tag = tag.strip()
        tag = tag.replace('TEXT SEQUENCING', 'TEXT-SEQUENCING')

        if tag in ['_', "*"]:
            continue

        elif "Sentence" in tag:
            continue

        elif "[" in tag:
            tag = re.sub(r'\[\d+\]', '', tag)
            if tag not in seen_tags:
                seen_tags[tag] = 1
            else:
                seen_tags[tag] += 1
                if replace:
                    tag = replacement[tag]
            new_list.append(tag)
        else:
            new_list.append(tag)

    return (new_list)


# def shorten_tag(cln_tag: list, seen_tags: dict, labels: dict ={}):
#     holder: list = []

#     for tag in cln_tag:
#         if tag not in seen_tags:


def dict2linearTag(dict_connl: dict,
                   layer: list = ['word', 'engmt', 'spl'],
                   hierarchy: bool = False):
    '''
    
    '''
    holder: list = []
    replacement = {
        'ATTRIBUTE': "ATTR",
        'ENTERTAIN': "ENT",
        'MONOGLOSS': "MONO",
        'CONCUR': "CCR",
        'COUNTER': "COUNT",
        'DENY': "DNY",
        'ENDORSE': "ENDS",
        'PRONOUNCE': 'PRON',
        "ADDITIVE": "ADD",
        "CITATION": "CTN",
        "COMPARATIVE": "COMP",
        "EXEMPLIFYING": "EX",
        "EXPOSITORY": "EXP",
        "GOAL-ANNOUNCING": "GOAL",
        "JUSTIFYING": "JUSTIFY",
        "QUOTED": "QT",
        "SOURCES": "SRS",
        "SUMMATIVE": "SUMS",
        "TEXT-SEQUENCING": "TEXTSQ",
        "ENDOPHORIC": "ENDPHRC",
        "_": "",
        "*": ""
    }

    hierarchyTag = {"Primary": "1", "Secondary": "2", "_": "", "*": ""}

    for chunkid, sentinfo in dict_connl.items():
        chunk_holder: list = []
        sentId = sentinfo['sentId']
        Text = sentinfo['Text']
        seen: dict = {
        }  # list of seen tags to decide to present full or short tags

        for line in sentinfo['lines']:
            token = []
            word = line['word']
            tag = line['tag']
            # iterate columns
            for col, v in line.items():
                if col == "word":
                    word = v
                elif col in layer:
                    # print(col, v)
                    tag_list = v.split("|")
                    # print(v, tag_list)
                    cln_tag = clean_tag(tag_list,
                                        seen_tags=seen,
                                        replacement=replacement,
                                        replace=True)

                    hie = clean_tag(line['hierarchy'].split("|"),
                                    seen_tags={},
                                    replacement={},
                                    replace=False)

                    if hierarchy:
                        if col == "engmt":
                            cln_tag = [
                                eng + hierarchyTag[h]
                                for eng, h in zip(cln_tag, hie)
                            ]

                    if len(cln_tag) > 0:
                        token.extend(cln_tag)

                else:  #skip if we dont have that layer
                    continue

            if len(token) > 0:
                tag_seq = "+".join([tag] + sorted(token))
            else:
                tag_seq = tag + "+Ø"  #re-implementing Aaron's zero formatting
            chunk_holder.append("_".join([word, tag_seq]))
        holder.append(" ".join(chunk_holder))
        print()
    return holder


def write_txt(sent_list: list[str], new_filename: str):

    new_filename = new_filename.replace('.tsv', '.txt')
    with open(new_filename, 'w') as f:
        for line in sent_list:
            f.write(line)
            f.write('\n')


### run the following
# filename = 'data/adjudicated/Batch1/D-batch/batch_1_D0_01.tsv'
filenames = glob.glob("data/0_EDT_three_para/**/*.tsv", recursive=True)

for filename in filenames:
    new_filename_rawID = dir_renamer(filename, 'linearTag/20230125/RealID/')
    new_filename_tsv = dir_renamer(filename, 'linearTag/20230125/tsvID/')
    make_dir(os.path.split(new_filename_rawID)[0])
    make_dir(os.path.split(new_filename_tsv)[0])

    conll_file = conll2dict(open(filename, 'r').read())
    pp.pprint(conll_file, sort_dicts=False)

    sent_list: list[str] = dict2linearTag(conll_file,
                                          layer=['word', 'engmt', 'spl'],
                                          hierarchy=True)
    text_id = conll_file[1]['sentId']
    print(text_id)
    new_filename = os.path.join(
        os.path.split(new_filename_rawID)[0], text_id + '.txt')
    write_txt(sent_list, new_filename)

    write_txt(sent_list, new_filename_tsv)
