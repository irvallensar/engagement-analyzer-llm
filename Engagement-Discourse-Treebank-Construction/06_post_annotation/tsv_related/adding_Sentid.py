import glob
import re
import os
import nltk

fileTochange = glob.glob(
    'data/annotated_for_reliability/Batch1/ME/*/batch*.tsv')


#define Jaccard Similarity function
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


saving_dir = 'tsv_data/annotation_files_integrated'

for file in fileTochange:
    tail = os.path.split(file)[-1]
    pattern = r'(batch1_.{2})_.+\.tsv'
    match = re.match(pattern, tail)

    batch_name = match.group(1)
    if match:
        print(match.group(1))

    with open(file, 'r') as f:
        target = f.read()
        t_chunks = target.split('\n\n')

    with open(
            "/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/1_corpus_sampling/batches_first_round/annotation_files_tsv_pos/{}_pos.tsv"
            .format(batch_name), 'r') as f:
        sourcef = f.read()
        s_chunks = sourcef.split('\n\n')

    with open(saving_dir + '/{}'.format(tail), 'w') as outf:

        for s, t in zip(s_chunks, t_chunks):
            # for l in s.split('\n'):
            #     print(l)
            sentid = re.match(r'\s{0,2}(#Sentence\.id=.+)\n', s)
            print('\nSOURCE:')
            print(s)
            print("\nTarget:")
            print(t)

            if t.startswith('#FORMAT='):
                outf.write(t)
                outf.write('\n\n\n')

            elif t.startswith('#Sentence') or t.startswith("\n#Sentence"):
                outf.write(t.strip())
                outf.write('\n\n')

            elif t.startswith('#Text') or t.startswith("\n#Text"):
                outf.write(sentid.group(1))
                outf.write('\n')
                outf.write(t.strip())
                outf.write('\n\n')
                # print(sentid.group(1))
                # print(t)

        # print(s[:100], t[:100])
