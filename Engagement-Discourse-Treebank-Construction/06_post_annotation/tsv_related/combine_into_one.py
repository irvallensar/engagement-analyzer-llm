import glob
import re
import csv

files = glob.glob('tsv_data/batch1_w_context/raw_txt/*')

files.sort()

for file in files:
    batch = re.search(r'batch_1_(.+)_\d+\.tsv', file)
    print(batch.group(1))
    with open(file, 'r') as f:
        content = f.read()
    with open('tsv_data/batch1_w_context/Batch1_combined.txt', 'a') as a:
        tsv_writer1 = csv.writer(a, dialect='excel-tab', delimiter='\t')
        lst = ["Batch1-{}".format(batch.group(1))
               ] + content.strip().split('\t')
        tsv_writer1.writerow(lst)
    with open(
            'tsv_data/batch1_w_context/Batch1_{}_combined.txt'.format(
                batch.group(1)), 'a') as outf:
        tsv_writer = csv.writer(outf, dialect='excel-tab', delimiter='\t')
        lst = content.strip().split('\t')
        tsv_writer.writerow(lst)
