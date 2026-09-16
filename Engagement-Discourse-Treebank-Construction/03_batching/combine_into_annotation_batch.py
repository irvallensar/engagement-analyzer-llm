import json
import glob
import random
'''
This code is used to sample annotation data from each dataset.
'''

## Loading datasets
with open('output/sentence_ids/batches/BAWE_batch.json', 'r') as f:
    bawe = json.load(f)

with open('output/sentence_ids/batches/MICUSP_batch.json', 'r') as f:
    micusp = json.load(f)

with open('output/sentence_ids/batches/FCE_answer2_batch.json', 'r') as f:
    fce = json.load(f)

with open('output/sentence_ids/batches/ICNALE_batch.json', 'r') as f:
    icnale = json.load(f)

with open('output/sentence_ids/batches/TOEFL11_batch.json', 'r') as f:
    toefl = json.load(f)

combined = []
for idx, x in enumerate(zip(bawe, micusp, fce, icnale, toefl), start=1):
    holder = []
    size = []
    for list in x:
        #print(list)
        size.append(len(list))
        holder += list
    #print(size)
    random.seed(1993)
    random.shuffle(holder)
    #print(len(holder))
    combined.append(holder)

# save each batch to
for idx, batch in enumerate(combined, start=1):
    with open("output/annotation_batches/{}_annotation_data.json".format(idx),
              'w') as f:
        json.dump(batch, f, indent=2)
