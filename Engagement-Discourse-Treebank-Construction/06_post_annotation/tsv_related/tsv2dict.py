import re
import csv
import json
import os


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


# col_def = [
#     'tid', 'chid', 'token', 'tag', 'pos', 'clause', 'engmt', 'hierarchy', 'spl'
# ]

# line2dict(
#     "2-25\t161-166\twould\tMD\t\tT-UNIT[5]|SUBORDINATE[8]\tENTERTAIN\tSecondary\tJUSTIFYING[18]\t",
#     col_def)


def conll2dict(conllfile: str, skip_list=[" "]):
    '''Takes conll file as string
    outputs 
    holder[chunkid] =  {'sentId': sent_id, 'Text': text, 'lines': temp}
    '''
    features = []
    # cols = [
    #     'tid', 'chid', 'word', 'tag', 'pos', 'clause', 'engmt', 'hierarchy',
    #     'spl'
    # ]
    cols = [
        'tid',
        'chid',
        'word',
    ]
    holder = {}
    for chunkid, chunk in enumerate(conllfile.split('\n\n')):
        temp = []
        sent_id = "NA"
        text = "NA"
        for line in chunk.split("\n"):
            if len(line) < 1:
                continue
            elif "#Sentence.id" in line:
                sent_id = re.match(r'#Sentence\.id=(.+)', line).group(1)
                # print(sent_id)
            elif "#Text=" in line:
                text = re.match(r'#Text=(.+)', line).group(1)

            elif "#T_SP=" in line:  #for webanno tsv format
                if 'POS' in line:
                    features.append('POS')
                    if "tag" not in cols:
                        cols.extend(['tag', 'pos'])
                elif 'Clause' in line:
                    features.append('Clause')
                    if 'clause' not in cols:
                        cols.extend(['clause'])
                elif 'Engagement' in line:
                    features.append('Engagement')
                    if 'engmt' not in cols:
                        cols.extend(['engmt'])
                    features.append('Hierarchy')
                    if 'hierarchy' not in cols:
                        cols.extend(['hierarchy'])
                elif 'Engmt_layer' in line:
                    cols.extend(['Engmt_layer'])
                    features.append('dpl_hierarchy')
                elif 'Supplementary' in line:
                    features.append('Supplementary')
                    if 'spl' not in cols:
                        cols.extend(['spl'])
                elif 'ModalSense' in line:
                    features.append('Modal')
                    cols.extend(['modal'])
            elif "#" in line[0]:
                pass
            else:
                if line.split('\t')[0] in skip_list:
                    continue
                else:
                    temp.append(line2dict(line, cols))
        if len(
                temp
        ) > 0 and text != "--- Para SEP ---":  #skip line with no sentences
            holder[chunkid] = {'sentId': sent_id, 'Text': text, 'lines': temp}
    return holder
