from scripts.post_process.tsv2iob2 import Run, make_iobs, write_file_by_sent
import random
import glob
import numpy as np
import os

import copy

features = ['engmt', 'spl']
input_dir = 'data/0_EDT_three/**'
# save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/assets/20221117_eng_w_test'
save_dir = '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder/assets/5_fold_20230124'

filenames = glob.glob('{}/*.tsv'.format(input_dir), recursive=True)
no_doc = len(filenames)
print("A total number of tsvs: {}".format(str(no_doc)))

seed = 808
random.seed(seed)
rand_num = [random.randint(1, 10000) for i in range(500)]

for i in rand_num:
    random.seed(i)
    random.shuffle(filenames)


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


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def make_kfold(filenames: list, k: int = 10):
    splits = list(np.array_split(filenames, k))
    holder = []
    for x in range(len(splits)):
        train = copy.deepcopy(splits)
        print(len(train))
        dev = train.pop(x)
        print(len(train))
        train = [file for fold in train for file in fold]
        holder.append({'train': train, "dev": dev})

    return holder


include_test = True
tenf = make_kfold(filenames, 5)
make_dir(save_dir)

for nth, fold in enumerate(tenf, start=1):

    trainf = list(fold['train'])
    devf = list(fold['dev'])

    split = len(devf) // 2
    testf = devf[split:]
    devf = devf[:split]

    for x in trainf:
        print(x)
        print(type(x))

    train = make_iobs(
        trainf,
        features=features,
        engage_hierarchy=False,
        alternate=False,
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
            "COMPARATIVE",
        ])
    dev = make_iobs(
        devf,
        features=features,
        engage_hierarchy=False,
        alternate=False,
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
            "COMPARATIVE",
        ])

    test = make_iobs(
        testf,
        features=features,
        engage_hierarchy=False,
        alternate=False,
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
            "COMPARATIVE",
        ])

    # make_dir("{}/fold_{}".format(save_dir, nth))

    train_sent, train_tags = count_tags(train)
    print_tag_counts(train_sent, train_tags, 'Training')
    dev_sent, dev_tags = count_tags(dev)
    print_tag_counts(dev_sent, dev_tags, 'Dev')

    write_file_by_sent('{}/train{}.iob'.format(save_dir, nth), train)
    write_file_by_sent('{}/dev{}.iob'.format(save_dir, nth), dev)
    test_sent, test_tags = count_tags(test)
    print_tag_counts(test_sent, test_tags, 'Dev')
    write_file_by_sent('{}/test{}.iob'.format(save_dir, nth), test)
