import os
import glob
import re
import pprint as pp
from scripts.matching_webannotsv import conll2dict

annotated_tsv = glob.glob(
    'tsv_data/annotated_Batch1/20220903_firstbatch_sentid_reconstructed/*.tsv')

annotated_tsv.sort()
annotated_tsv


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def extract_tsv_features(format_chunk: str,
                         colids: list = ['tid', 'cid', 'token']):
    colids = colids
    lines = format_chunk.split('\n')
    for l in lines:
        for attr in l.split("|")[1:]:
            if attr not in colids:
                colids.append(attr)
    return colids


def extract_sentence_annotation(sentence_chunk: str, colids: list):
    lines = sentence_chunk.split('\n')
    sentinfo = {}

    for l in lines:  #iterate tokens
        l = l.strip()
        # first we need to separate metadata from token lines
        if len(l) < 1:
            continue
        elif "#Sentence.id" in l:
            sent_id = re.search(r'#Sentence\.id=(.+)', l).group(1)
            sentinfo['Sentence.id'] = sent_id
            #print(sent_id)
        elif "#Text=" in l:
            text = re.search(r'#Text=(.+)', l).group(1)
            sentinfo['Text'] = text
        elif "#" in l[0]:
            #print(line)
            pass
        ## tokens are processed below
        else:
            if "lines" not in sentinfo:
                sentinfo["lines"] = []

            holder = {}
            for col, val in zip(colids, l.strip().split('\t')):
                #print(col, val)
                holder[col] = val
            sentinfo["lines"].append(holder)
    return sentinfo


def webannotsv2dict(text: str):
    chunks = text.split('\n\n')
    # print(len(chunks))
    colids = ['tid', 'cid', 'token']

    chunks_dict = {}

    for c in chunks:
        ## Breakdown format headers for indexing
        if c.startswith('#FORMAT='):
            lines = c.split('\n')
            for l in lines:
                for attr in l.split("|")[1:]:
                    if attr not in colids:
                        colids.append(attr)

        ## Using indexes to make a dictionary
        elif c.startswith('#Sentence') or c.startswith("\n#Sentence"):

            # lines = c.split('\n')
            # sentinfo = {}

            # for l in lines:  #iterate tokens
            #     l = l.strip()
            #     # first we need to separate metadata from token lines
            #     if len(l) < 1:
            #         continue
            #     elif "#Sentence.id" in l:
            #         sent_id = re.match(r'#Sentence\.id=(.+)', l).group(1)
            #         sentinfo['Sentence.id'] = sent_id
            #         #print(sent_id)
            #     elif "#Text=" in l:
            #         text = re.match(r'#Text=(.+)', l).group(1)
            #         sentinfo['Text'] = text
            #     elif "#" in l[0]:
            #         #print(line)
            #         pass
            #     ## tokens are processed below
            #     else:
            #         if "lines" not in sentinfo:
            #             sentinfo["lines"] = []

            #         holder = {}
            #         for col, val in zip(colids, l.strip().split('\t')):
            #             #print(col, val)
            #             holder[col] = val
            #         sentinfo["lines"].append(holder)
            sentinfo = extract_sentence_annotation(c, colids)
            chunks_dict[sentinfo['Sentence.id']] = sentinfo
    print(len(chunks_dict))
    return (chunks_dict)


def load_sentbased_annotations(annotated_tsv: list[str]):

    database = {}

    for file in annotated_tsv:
        print(file)
        with open(file, 'r') as f:
            text = f.read()
            database |= webannotsv2dict(text)
    return database


db = load_sentbased_annotations(annotated_tsv)

# pp.pprint(a['0340a.xml_s1.1;p36.76'], sort_dicts=False)

# len(db)

# db.keys()

### Go through the non-annotated data

## if the sentence in the annotated data is found, replace the chunk with the old one,
## you need to align the columns.
# If possible, treat JUSTIFY and CITATION in the supplementary layer.


def check_match(chunk: str, sent_db: dict, print_res: bool = False):

    sent_id = re.search(r'#Sentence\.id=(.+)', chunk).group(1)
    Text = re.search(r'#Text=(.+)', chunk).group(1).strip()

    if sent_id in sent_db:

        anno_tsv_text = sent_db[sent_id]['Text'].strip()
        if print_res:
            print(sent_id)
            print(Text)
            print(anno_tsv_text)
            print(Text == anno_tsv_text)
            print()

        return Text == anno_tsv_text  #check if the sentence is the same
    else:
        if print_res:
            print(sent_id)
            print("+++++++++++++++++++++++")
        return False


import copy


def clean_empty_tag(tag: str, empties=['*']):
    if tag in empties:
        return ("_")
    else:
        return (tag)


def separate_tags(tag: str):
    tag_list = tag.split("|")
    eng = []
    spl = []
    for tag in tag_list:
        tag = clean_empty_tag(tag)

        if "JUSTIFY" in tag:
            spl.append(tag.replace('JUSTIFY', 'JUSTIFYING'))
        elif "CITATION" in tag:
            spl.append(tag)
        else:
            eng.append(tag)

    if len(eng) == 0:
        eng.append("_")
    if len(spl) == 0:
        spl.append("_")

    return "|".join(eng), "|".join(spl)


"|".join(["CITATION"])


def replace_tsv_content(chunk: str,
                        sent_db: dict,
                        feature_list: list,
                        cols_to_update: list = ['Clausetype', 'Engagement']):

    sent_id = re.search(r'#Sentence\.id=(.+)', chunk).group(1)
    Text = re.search(r'#Text=(.+)', chunk).group(1).strip()

    annoted_lines = sent_db[sent_id]['lines']
    target_sent_chunk = extract_sentence_annotation(chunk, feature_list)
    rev_target_sent_chunk = copy.deepcopy(target_sent_chunk)

    for tl, annol in zip(rev_target_sent_chunk['lines'], annoted_lines):

        for col in cols_to_update:
            if "JUSTIFY" in annol[col] or 'CITATION' in annol[col]:
                eng_tags, spl_tags = separate_tags(annol[col])

                tl['Engagement'] = eng_tags
                tl['Hierarchy'] = "_"
                tl['Supplementaryrhetoricalmoves'] = spl_tags
            else:
                tl[col] = annol[col]
    # print(rev_target_sent_chunk)
    return rev_target_sent_chunk


def dict2webannotsv(chunk_dict: dict, features: list):

    lines = []

    for l in chunk_dict['lines']:
        holder = [l[col] for col in features]
        lines.append('\t'.join(holder))

    annotations = "\n".join(lines)
    # print(annotations)

    full_chunk = '''#Sentence.id={}\n#Text={}\n{}'''.format(
        chunk_dict['Sentence.id'], chunk_dict['Text'], annotations)
    return full_chunk


def convert(raw_tsv, print_replaced=False):
    holder = []
    with open(raw_tsv, 'r') as f:
        text = f.read()
        chunks = text.split('\n\n')
        #print(chunks)

        # check if the sentence should be swapped
        colids = []
        for c in chunks:
            if c.startswith('#FORMAT='):
                holder.append(
                    c + '\n'
                )  #this is to add extra line character in the annotation
                colids = extract_tsv_features(c)

            elif c.startswith('#Sentence.id') or c.startswith(
                    "\n#Sentence.id"):
                c = c.strip()
                replace = check_match(c, db)

                if replace:
                    s_chunk_dict = replace_tsv_content(
                        c, db, colids, cols_to_update=['Engagement'])
                    full_chunk = dict2webannotsv(s_chunk_dict, colids)
                    if print_replaced:

                        print("##########")
                        print(full_chunk)
                        print('#######')
                    holder.append(full_chunk)
                else:
                    holder.append(c)
    return (holder)


def write_new_tsv(chunk_list: list, save_dir: str, tail: str):

    with open(save_dir + "/" + tail, 'w') as f:
        f.write("\n\n".join(chunk_list))


raw_tsvs = glob.glob(
    'tsv_data/batches_first_round_w_context_cl2/annotation_files_tsv_pos_withRealID/*.tsv'
)
raw_tsvs.sort()

make_dir('tsv_data/batches_first_round_w_context_integrated_cl2')

for r_tsv in raw_tsvs:
    tail = os.path.split(r_tsv)[-1]
    c_list = convert(r_tsv)
    write_new_tsv(c_list,
                  'tsv_data/batches_first_round_w_context_integrated_cl2',
                  tail)

# convert('tsv_data/batches_first_round_w_context_cl2/annotation_files_tsv_pos_withRealID/PSY.G1.11.1.xml_24_7.tsv', print_replaced = True)