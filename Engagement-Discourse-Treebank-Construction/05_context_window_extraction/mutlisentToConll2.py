import glob
import random
import os
import spacy

nlp = spacy.load(
    '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Eng_Clause_span_finder/packages/en_engagement_RoBERTa_combined-cl_0.2.5/en_engagement_RoBERTa_combined/en_engagement_RoBERTa_combined-cl_0.2.5'
)

nlp.component
# nlp = spacy.load('en_core_web_trf', disable=['ner'])

from construction_specific_samples.scripts.making_annotation_file3 import write_webanno_tsv, preprocess, spacy_tokenize, spacy_tokenize_tsv

tsv_head = '''#FORMAT=WebAnno TSV 3.3
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement
#T_SP=webanno.custom.ModalSenseDisambiguation|ModalSense'''

tsv_head_pos = '''#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement|Hierarchy
#T_SP=webanno.custom.Supplementary|Supplementaryrhetoricalmoves'''

tsv_head_pos_dep = '''#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement|Hierarchy
#T_SP=webanno.custom.Supplementary|Supplementaryrhetoricalmoves
#T_RL=de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency|DependencyType|flavor|BT_de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS'''


def sentenceTowebanno_tsv(text,
                          sid,
                          outfile,
                          header: str = tsv_head,
                          pos=False,
                          dep=False,
                          dep_list=None):
    outfile.write(header)
    outfile.write("\n")
    last_char = 0

    sent = preprocess(text)
    # Some sentences are empty, and has to be escaped
    if len(sent) < 1:
        print(sent)
        outfile.write("\n\n#Sentence.id={}".format(sid))
        #this has caused a display problem in webanno
        # outfile.write("\n#Text={}\t{}".format(sent, "SKIPANNOTATION"))
        # token_list, last_char = spacy_tokenize_tsv(
        #     "{}\t{}".format(sent, "SKIPANNOTATION"), sid, last_char, pos,
        #     dep, dep_list)
        outfile.write("\n#Text={}".format("SKIPANNOTATION"))
        token_list, last_char = spacy_tokenize_tsv(
            "{}".format("SKIPANNOTATION"), sid, last_char, pos, dep, dep_list)

    else:
        outfile.write("\n\n#Sentence.id={}".format(sid))
        outfile.write("\n#Text={}".format(sent))
        token_list, last_char = spacy_tokenize_tsv(sent, sid, last_char, pos,
                                                   dep, dep_list)

    for t in token_list:
        outfile.write("\n")
        outfile.write("\t".join(t) + "\t")


def sentencesTowebanno_tsv(batch,
                           meta,
                           outfile,
                           header: str = tsv_head,
                           pos=False,
                           dep=False,
                           dep_list=None):
    outfile.write(header)
    outfile.write("\n")
    last_char = 0
    last_span_id = 1

    for sid, sent in enumerate(batch, start=1):
        sent = preprocess(sent)
        # Some sentences are empty, and has to be escaped
        if len(sent) < 1:
            print(sent)
            outfile.write("\n\n#Sentence.id={}".format(meta))
            #this has caused a display problem in webanno
            # outfile.write("\n#Text={}\t{}".format(sent, "SKIPANNOTATION"))
            # token_list, last_char = spacy_tokenize_tsv(
            #     "{}\t{}".format(sent, "SKIPANNOTATION"), sid, last_char, pos,
            #     dep, dep_list)
            outfile.write("\n#Text={}".format("SKIPANNOTATION"))
            token_list, last_char, last_spanid = spacy_tokenize_tsv(
                "{}".format("SKIPANNOTATION"), sid, last_char, pos, dep,
                dep_list, last_span_id)

        else:
            outfile.write("\n\n#Sentence.id={}".format(meta))
            outfile.write("\n#Text={}".format(sent))
            token_list, last_char, last_span_id = spacy_tokenize_tsv(
                sent, sid, last_char, pos, dep, dep_list, last_span_id)

        for t in token_list:
            outfile.write("\n")
            outfile.write("\t".join(t) + "\t")


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


samples = glob.glob('batches_cx_spec_w_contexts/textfiles/*.txt')

samples.sort()
samples[:10]

random.seed(10)
random.shuffle(samples)
samples[:10]


def main(
    samples: list,
    data_batch: str,
):
    make_dir('{}/annotation_files_tsv_pos'.format(data_batch))
    make_dir('{}/annotation_files_rawids'.format(data_batch))
    make_dir('{}/annotation_files_tsv_pos_withRealID'.format(data_batch))

    for idx, file in enumerate(samples):
        tail = os.path.split(file)[-1].replace(".txt", '')

        fid = str(idx)
        if len(fid) == 1:
            fid = "000" + fid
        elif len(fid) == 2:
            fid = "00" + fid
        elif len(fid) == 3:
            fid = "0" + fid

        print(tail)
        with open(file, 'r') as f:
            text = f.read()
            sents = text.split("\n")

            with open(
                    '{}/annotation_files_tsv_pos/batch2_{}.tsv'.format(
                        data_batch, fid), 'w') as f:
                sentencesTowebanno_tsv(sents, tail, f, tsv_head_pos, pos=True)

            with open(
                    '{}/annotation_files_tsv_pos_withRealID/{}.tsv'.format(
                        data_batch, tail), 'w') as f:
                sentencesTowebanno_tsv(sents, tail, f, tsv_head_pos, pos=True)

            # id + tokenized text
            with open('{}/annotation_files_rawids/{}_withids.txt'.format(
                    data_batch, tail),
                      'w',
                      encoding='utf-8') as f:
                #tsv_writer = csv.writer(f, delimiter = "\t", escapechar='\\', quoting = csv.QUOTE_NONE)
                f.write(tail + "\t" + text.replace("\n", ' '))


## This is for context specific tags
if __name__ == '__main__':
    samples = glob.glob('batches_cx_spec_w_contexts/textfiles/*.txt')

    samples.sort()
    samples[:10]

    random.seed(10)
    random.shuffle(samples)
    samples[:10]

    main(samples, 'batches_cx_spec_w_contexts3')
    ## write raw file

    # ## This is for
    first_batch = glob.glob('batches_first_round/with_contexts/*.txt')
    first_batch.sort()
    main(first_batch, 'batches_first_round_w_context_cl3')
