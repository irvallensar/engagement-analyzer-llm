import re
import csv
import json
import os

import glob
import random

from scripts.tsv_related.tsv2dict import conll2dict

DOC_DELIMITER = "-DOCSTART- -X- O O\n"


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


def line2dict(line: str,
              cols,
              fullcols: list = [
                  'tid', 'chid', 'token', 'tag', 'pos', 'clause', 'engmt',
                  'hierarchy', 'spl'
              ]):
    l = {}
    items = line.strip().split("\t")
    if len(items) == 8:
        items.append("_")

    for c, label in zip(cols, items):
        l[c] = label

    for layer in fullcols:
        if layer not in l:
            l[layer] = "_"
    return l


col_def = [
    'tid', 'chid', 'token', 'tag', 'pos', 'clause', 'engmt', 'hierarchy', 'spl'
]

line2dict(
    "2-25\t161-166\twould\tMD\t\tT-UNIT[5]|SUBORDINATE[8]\tENTERTAIN\tSecondary\tJUSTIFYING[18]\t",
    col_def)

# def conll2dict(conllfile: str, skip_list=[" "]):
#     '''Takes conll file as string
#     outputs
#     holder[chunkid] =  {'sentId': sent_id, 'Text': text, 'lines': temp}
#     '''
#     features = []
#     # cols = [
#     #     'tid', 'chid', 'word', 'tag', 'pos', 'clause', 'engmt', 'hierarchy',
#     #     'spl'
#     # ]
#     cols = [
#         'tid',
#         'chid',
#         'word',
#     ]
#     holder = {}
#     for chunkid, chunk in enumerate(conllfile.split('\n\n')):
#         temp = []
#         sent_id = "NA"
#         for line in chunk.split("\n"):
#             if len(line) < 1:
#                 continue
#             elif "#Sentence.id" in line:
#                 sent_id = re.match(r'#Sentence\.id=(.+)', line).group(1)
#                 # print(sent_id)
#             elif "#Text=" in line:
#                 text = re.match(r'#Text=(.+)', line).group(1)

#             elif "#T_SP=" in line:  #for webanno tsv format
#                 if 'POS' in line:
#                     features.append('POS')
#                     if "tag" not in cols:
#                         cols.extend(['tag', 'pos'])
#                 elif 'Clause' in line:
#                     features.append('Clause')
#                     if 'clause' not in cols:
#                         cols.extend(['clause'])
#                 elif 'Engagement' in line:
#                     features.append('Engagement')
#                     if 'engmt' not in cols:
#                         cols.extend(['engmt'])
#                     features.append('Hierarchy')
#                     if 'hierarchy' not in cols:
#                         cols.extend(['hierarchy'])
#                     features.append('Supplementary')
#                     if 'spl' not in cols:
#                         cols.extend(['spl'])
#                 elif 'ModalSense' in line:
#                     features.append('Modal')
#                     cols.extend(['modal'])
#             elif "#" in line[0]:
#                 pass
#             else:
#                 if line.split('\t')[0] in skip_list:
#                     continue
#                 else:
#                     temp.append(line2dict(line, cols))
#         if len(temp) > 0:  #skip line with no sentences
#             holder[chunkid] = {'sentId': sent_id, 'Text': text, 'lines': temp}
#     return holder


def _combine_eng_hierarchy(_eng: list, _hie: list):
    res = []
    for eng, hie in zip(_eng, _hie):
        # print(eng, hie)
        engInd = re.match(r"^(.+)\[(\d+)\]$", eng)
        hieInd = re.match(r"^(.+)\[(\d+)\]$", hie)
        if engInd:
            assert engInd.group(2) == hieInd.group(2)
            eng_tag = engInd.group(1)

            hie_tag = "P" if hieInd.group(1) == "Primary" else "S"
            new_eng = "{}_{}[{}]".format(eng_tag, hie_tag, engInd.group(2))
            res.append(new_eng)
        elif eng not in ["_", "*"]:
            hie_tag = "P" if hie == "Primary" else "S"
            new_eng = "{}_{}".format(eng, hie_tag)
            res.append(new_eng)
        else:
            res.append(eng)
    return res


def _cln_tag(tag: str, engage_hierarchy: bool = False):
    replacement = {
        'GOAL-ANNOUNCING': "GOAL_ANNOUNCING",
        'TEXT SEQUENCING': "TEXT_SEQUENCING",
        'ATTRIBUTE': 'ATTRIBUTION',
        'ENDORSE': 'ATTRIBUTION'
    }

    if tag in replacement:
        tag = replacement[tag]
    if not engage_hierarchy:
        tag = re.sub(r"_(S|P)$", "", tag)
    return tag


def sent2iob(
        sent_lines: list,
        features: list = ['engmt', 'spl'],
        col_size: int = 5,
        ignore_tags=[
            'T-UNIT',
            'QUOTED',
            "Sentence",
            "ADDITIVE",
            "SUMMATIVE",
            "EXEMPLIFYING",
            "EXPOSITORY",
            "GOAL-ANNOUNCING",
            "TEXT SEQUENCING",
            "MONOGLOSS_S",
            # "ENDOPHORIC",
        ],
        engage_hierarchy=True,
        switch=False):
    holder = []
    seen_tags = {}
    # print(ignore_tags)

    for l in sent_lines:
        # print(l)
        tags = []
        for feature in features:
            # tag = l[feature]
            # if tag != "_":
            if feature == 'engmt':
                # before 2022.11.30, the code added hierarchy if specified
                # I have refactored code to strip the hierarchy after assigning them later in the code.
                # if engage_hierarchy:
                #     _eng = l[feature].split("|")
                #     _hie = l['hierarchy'].split("|")
                #     ltag = _combine_eng_hierarchy(_eng, _hie)
                # else:
                #     ltag = l[feature].split("|")

                _eng = l[feature].split("|")
                _hie = l['hierarchy'].split("|")
                ltag = _combine_eng_hierarchy(_eng, _hie)
            else:
                ltag = l[feature].split("|")
            tags.extend(ltag)

        # if len(tags) > 1:
        #     print(tags)

        iob_tags = []

        for tag in tags:
            if any([t in tag for t in ignore_tags]):
                print(f"Deleted: {tag}")
                continue
            else:
                pat = r"(.*)\[(\d+)\]"
                match = re.match(pat, tag)

                if match:  ## multiword tags
                    cln_tag = _cln_tag(match.group(1),
                                       engage_hierarchy=engage_hierarchy)
                    if tag not in seen_tags:
                        seen_tags[tag] = None
                        ## strip number here
                        iob_tags.append("B-{}".format(cln_tag))
                    else:
                        ## strip number here
                        iob_tags.append("I-{}".format(cln_tag))

                else:
                    cln_tag = _cln_tag(tag, engage_hierarchy=engage_hierarchy)
                    if tag not in ["_", "*"]:
                        iob_tags.append("B-{}".format(cln_tag))
                    else:
                        iob_tags.append("O")

        #Then store as string
        n_tags = len(iob_tags)
        for i in range(0, col_size - n_tags):
            iob_tags.append("O")

        if switch:
            iob_tags[0], iob_tags[1] = iob_tags[1], iob_tags[0]
        # print(iob_tags)
        holder.append("\t".join([l['word']] + iob_tags))
    return holder


# pat = r"(.*)\[(\d+)\]"
# match = re.match(pat, 'DENY[1]')

# if match:
#     print("yes")

# file1 = "data/Annotated_files_ALM/completed_archive/download-20220616/1C-Batch/2C_completed/batch1_C6_pos_ALM.tsv"


def tsv2sentlist(
    filename: str,
    features: list = ['engmt', 'spl'],
    engage_hierarchy=True,
    switch: bool = False,
    ignore_tags=[[
        'T-UNIT',
        'QUOTED',
        "Sentence",
        "ADDITIVE",
        "SUMMATIVE",
        "EXEMPLIFYING",
        "EXPOSITORY",
        "GOAL-ANNOUNCING",
        "TEXT SEQUENCING",
        "MONOGLOSS_S",
        # "ENDOPHORIC",
    ]]):

    holder = []

    conll_file = conll2dict(open(filename, 'r').read())
    for sid, sent in conll_file.items():
        holder.append(
            sent2iob(sent['lines'],
                     features,
                     engage_hierarchy=engage_hierarchy,
                     switch=switch,
                     ignore_tags=ignore_tags))

    return (holder)


def write_file_by_sent(output_name: str,
                       data: list[list],
                       doc_delim: str = DOC_DELIMITER):
    with open(output_name, 'w', encoding='utf-8') as f:

        for doc in data:
            check = ['\tO\tO\tO\tO\tO' in line for line in doc]
            # if True in check:
            #     print(doc)
            #     print(type(doc))
            #     print(check)
            for sent in doc:
                check2 = ['\tO\tO\tO\tO\tO' in line for line in sent]
                # if False in check2:
                #     f.write("\n".join(sent))
                # else:
                #     print(check2)
                #     print(sent)
                f.write("\n".join(sent))
                f.write('\n\n')
            f.write(doc_delim)


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def make_iobs(
    filepath: list,
    features: list,
    engage_hierarchy=True,
    alternate=False,
    ignore_tags=[[
        'T-UNIT',
        'QUOTED',
        "Sentence",
        "ADDITIVE",
        "SUMMATIVE",
        "EXEMPLIFYING",
        "EXPOSITORY",
        "GOAL-ANNOUNCING",
        "TEXT SEQUENCING",
        "MONOGLOSS_S",
        # "ENDOPHORIC",
    ]]):
    holder = []
    for n, file in enumerate(filepath):
        if alternate:
            if (n % 2) == 0:
                switch = True
            else:
                switch = False
        else:
            switch = False
        sents = tsv2sentlist(file,
                             features,
                             engage_hierarchy=engage_hierarchy,
                             switch=switch,
                             ignore_tags=ignore_tags)
        # print(sents)
        holder.append(sents)
    return holder


def count_tags(data: list):

    holder = {}
    sent_count = 0
    for n, doc in enumerate(data, start=1):
        for sent in doc:
            sent_count += 1
            for line in sent:
                tags = line.split("\t")[1:]
                for tag in tags:
                    if tag.startswith("B-"):
                        if tag not in holder:
                            holder[tag] = 1
                        else:
                            holder[tag] += 1
    return sent_count, holder


def print_tag_counts(sentno: int, tag_counts: dict, dataset: str):
    print(f"============ {dataset} tag counts =============")
    print(f"No. of sentences: \t {sentno}")

    for tag, n in sorted(tag_counts.items()):
        print(f"{tag}\t\t{n}")
    print()


# The output would be
def Run(
        split: float = 0.9,
        features: list = ["engmt"],
        engage_hierarchy=False,
        input_dir='data/data_to_convert/20220903_firstbatch_AtoI',
        save_dir='data/iob_data/20220903_AtoI_not-adjudicated',
        seed=1993,
        ignore_tags=[
            'T-UNIT',
            'QUOTED',
            "Sentence",
            "ADDITIVE",
            "SUMMATIVE",
            "EXEMPLIFY",
            "EXPOSITORY",
            "GOAL-ANNOUNCING",
            "TEXT SEQUENCING",
            "MONOGLOSS_S",
            # "ENDOPHORIC",
        ],
        shuffle=True):

    filenames = glob.glob('{}/*.tsv'.format(input_dir), recursive=True)

    filenames.sort()

    if shuffle:
        random.seed(seed)
        rand_num = [random.randint(1, 10000) for i in range(100)]
        for n in rand_num:
            random.seed(n)
            random.shuffle(filenames)

    no_doc = len(filenames)
    print("A total number of tsvs: {}".format(str(no_doc)))
    trf, devf = filenames[0:int(no_doc * split)], filenames[int(no_doc *
                                                                split):]

    train = make_iobs(trf,
                      features=features,
                      engage_hierarchy=engage_hierarchy,
                      alternate=False,
                      ignore_tags=ignore_tags)
    dev = make_iobs(devf,
                    features=features,
                    engage_hierarchy=engage_hierarchy,
                    alternate=False,
                    ignore_tags=ignore_tags)

    print("Training data: {} files; \n Develop data: {} files".format(
        len(train), len(dev)))

    train_sent, train_tags = count_tags(train)
    print_tag_counts(train_sent, train_tags, 'Training')
    dev_sent, dev_tags = count_tags(dev)
    print_tag_counts(dev_sent, dev_tags, 'Dev')

    write_file_by_sent('{}/train.iob'.format(save_dir), train)

    write_file_by_sent('{}/dev.iob'.format(save_dir), dev)


def main(input_dir: str,
         output_dir: str,
         train_ratio: float,
         shuffle: bool = True):
    make_dir(output_dir)
    Run(
        split=train_ratio,
        features=['engmt', "spl"],
        engage_hierarchy=False,
        input_dir=input_dir,
        save_dir=output_dir,
        seed=1234,
        ignore_tags=[
            'T-UNIT',
            'QUOTED',
            "Sentence",
            "MONOGLOSS_S",
            "ADDITIVE",
            # "SUMMATIVE",
            # "EXEMPLIFYING",
            # "EXPOSITORY",
            # "TEXT SEQUENCING",
            # 2023/01/21
            # "GOAL-ANNOUNCING",
            # "COMPARATIVE",
            # 2023/01/09
            # "ENDOPHORIC",
        ],
        shuffle=shuffle)


if __name__ == "__main__":
    # input_dir = 'data/data_to_convert/20221007_clause'
    # save_dir = 'data/iob_data/20221007_clause'
    # input_dir = 'data/tentative_/**/**'
    # input_dir = 'data/adjudicated/**'
    # save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/assets/20230111'
    # main(input_dir, save_dir, .85)

    # This is previous command to convert create whole document texts into iob formats

    # input_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/engagement-corpus/data/annotated_for_reliability/test_set/ALM/'
    # save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/Test_set/20230110/ALM'
    # main(input_dir, save_dir, 1, shuffle=False)

    # input_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/engagement-corpus/data/annotated_for_reliability/test_set/RW/'
    # save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/Test_set/20230110/RW'
    # main(input_dir, save_dir, 1, shuffle=False)

    # input_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/engagement-corpus/data/2_test_set/'
    # save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/Test_set/reviewed2'
    # main(input_dir, save_dir, 1, shuffle=False)

    input_dir = 'data/0_EDT_three/**'
    save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/assets/EDT_three_most_tags'
    main(input_dir, save_dir, .80, shuffle=True)

    input_dir = 'data/0_EDT_three_para/**'
    save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/assets/EDT_para_most_tags'
    main(input_dir, save_dir, .80, shuffle=True)

    # save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Eng_Clause_span_finder/assets/20221208_cl_citation'
    # input_dir = 'data/adjudicated/**/**'
    # # save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/assets/20221208_eng_spl_reduced'
    # save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Eng_Clause_span_finder/assets/20221208_cl_citation'
    # make_dir(save_dir)
    # Run(split=0.8,
    #     features=['clause', "spl"],
    #     engage_hierarchy=False,
    #     input_dir=input_dir,
    #     save_dir=save_dir,
    #     seed=1234,
    #     ignore_tags=[
    #         "Sentence", "ADDITIVE", "MONOGLOSS_S", "ENDOPHORIC",
    #         "EXEMPLIFYING", "EXPOSITORY", "SUMMATIVE", "GOAL-ANNOUNCING",
    #         "JUSTIFYING", "TEXT SEQUENCING", "COMPARATIVE", "ENDOPHORIC",
    #         "SOURCES", "QUOTED"
    #     ])