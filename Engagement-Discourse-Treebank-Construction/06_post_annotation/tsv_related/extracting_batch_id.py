import glob
import re
import json
import os
import shutil

files = glob.glob(
    '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_corpus_sampling/batches_first_round/annotation_files_rawids/*.txt'
)


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


holder = {}
for file in files:
    batch_id = re.search(r'batch1_(.+)_withids\.txt', file).group(1)

    with open(file, 'r') as f:
        for n, line in enumerate(f.readlines(), start=1):
            id, sent = line.split("\t")
            print(id)
            print(n)
            holder[id] = {'batch': batch_id, 'sentn': n}

tsvs = glob.glob('tsv_data/batches_first_round_w_context_integrated_cl2/*.tsv')
make_dir('tsv_data/batch1_w_context/')

for tsv in tsvs:
    name = os.path.split(tsv)[-1].replace('.tsv', '')
    print(name)

    print(holder[name])
    if holder[name]['sentn'] < 10:
        n = "0" + str(holder[name]['sentn'])
    else:
        n = str(holder[name]['sentn'])
    newname = 'tsv_data/batch1_w_context/batch_1_{}_{}.tsv'.format(
        holder[name]['batch'], n)
    shutil.copy(tsv, newname)

tsvs = glob.glob(
    'tsv_data/batches_first_round_w_context_cl2/annotation_files_rawids/*')

path_to_save = 'tsv_data/batch1_w_context/raw_txt'
suffix_to_replace = '_withids.txt'
make_dir(path_to_save)

for tsv in tsvs:
    name = os.path.split(tsv)[-1].replace(suffix_to_replace, '')
    print(name)

    print(holder[name])
    if holder[name]['sentn'] < 10:
        n = "0" + str(holder[name]['sentn'])
    else:
        n = str(holder[name]['sentn'])
    newname = '{}/batch_1_{}_{}.tsv'.format(path_to_save,
                                            holder[name]['batch'], n)
    shutil.copy(tsv, newname)