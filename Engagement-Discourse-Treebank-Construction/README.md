# Engagement Discourse Treebank — Construction Pipeline

This repository documents and reproduces the complete pipeline used to construct the **Engagement Discourse Treebank (EDT)**: a multi-corpus annotated dataset of academic and learner writing with engagement markers, clause boundaries, and modal sense annotations based on Systemic Functional Linguistics (SFL).

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Source Corpora](#source-corpora)
5. [Pipeline](#pipeline)
   - [Phase 1: Sentence Tokenization](#phase-1-sentence-tokenization)
   - [Phase 2: Sentence ID Extraction & Shuffling](#phase-2-sentence-id-extraction--shuffling)
   - [Phase 3: Per-Corpus Batching](#phase-3-per-corpus-batching)
   - [Phase 4: Combined Annotation Batches](#phase-4-combined-annotation-batches)
   - [Phase 5a: First-Round WebAnno TSV Generation](#phase-5a-first-round-webanno-tsv-generation)
   - [Phase 5b: Context Window Extraction](#phase-5b-context-window-extraction-3-sentence-windows)
   - [Phase 5c: WebAnno TSV with Custom Engagement Model](#phase-5c-webanno-tsv-with-custom-engagement-model)
   - [Phase 6: Manual Annotation in WebAnno](#phase-6-manual-annotation-in-webanno)
   - [Phase 7: Post-Annotation Processing](#phase-7-post-annotation-processing)
6. [Key Parameters](#key-parameters)
7. [Annotation Layers](#annotation-layers)
8. [Output Files](#output-files)

---

## Overview

The Engagement Discourse Treebank is built from five academic writing corpora:

| Corpus | Type | Per-batch size | Total approx. |
|--------|------|---------------|---------------|
| BAWE | Expert academic (UK undergraduate) | 750 | ~50M tokens |
| MICUSP | Expert academic (US upper-level) | 750 | ~18M tokens |
| FCE | Learner (Cambridge CLC, answer2 task) | 150 | ~4.6M tokens |
| TOEFL11 | Learner (ESL, 11 L1 backgrounds) | 200 | ~25M tokens |
| ICNALE | Learner (Asian L2 English) | 150 | ~1.3M tokens |

**Combined batch size:** ~2,000 sentences per annotation batch.  
**Global random seed:** `1993`  
**Context window seed:** `10`

The pipeline proceeds in two major stages:
1. **Pre-annotation** (`1_corpus_sampling/`): sentence tokenization → ID extraction → batching → WebAnno file generation
2. **Post-annotation** (`engagement-corpus/`): inter-annotator reliability → adjudication → IOB conversion → final dataset

---

## Directory Structure

```
Engagement-Discourse-Treebank-Construction/
├── README.md                           ← this file
├── requirements.txt                    ← Python dependencies
│
├── 01_sentence_tokenization/           ← Phase 1
│   ├── sent_tokenization_MICUSP.py
│   ├── make_tokenized_xml.py
│   └── BAWE_extractor.py
│
├── 02_sentence_id_extraction/          ← Phase 2
│   ├── extract_sent_ids.py
│   └── utils.py
│
├── 03_batching/                        ← Phases 3 & 4
│   ├── make_corpus_batches.py
│   ├── combine_into_annotation_batch.py
│   └── utils.py
│
├── 04_annotation_file_generation/      ← Phase 5a
│   ├── making_annotation_file.py
│   └── utils.py
│
├── 05_context_window_extraction/       ← Phases 5b & 5c
│   ├── add_contextTosentences.py
│   └── mutlisentToConll2.py
│
├── 06_post_annotation/                 ← Phase 7
│   ├── tsv_related/
│   │   ├── integrate_annotated.py
│   │   ├── combine_into_one.py
│   │   ├── adding_Sentid.py
│   │   ├── extracting_batch_id.py
│   │   └── tsv2dict.py
│   ├── reliability/
│   │   ├── matching_webannotsv.py
│   │   ├── run_reliability.py
│   │   ├── sorting_for_error_analysis.py
│   │   ├── json2accuracy.py
│   │   └── accuracy_calculator.py
│   ├── adjudication_helper/
│   │   ├── tsv2LinearTags.py
│   │   └── EngageAntConcDirectoryV2.py
│   ├── post_process/
│   │   ├── tsv2iob.py
│   │   ├── tsv2iob2.py
│   │   ├── split_tsv.py
│   │   └── make_kfold.py
│   └── utils/
│       ├── __init__.py
│       └── aligner.py
│
├── output/
│   ├── sentence_ids/                   ← shuffled sentence ID lists (JSON)
│   │   └── batches/                   ← per-corpus batched ID lists (JSON)
│   ├── annotation_batches/             ← combined annotation batches (JSON)
│   └── target_list/                   ← verb semantic class lists (TXT)
│
└── corpus_data/
    ├── README_corpora.md              ← how to obtain & place each corpus
    ├── BAWE/                          ← place corpus here (not included)
    ├── MICUSP_scraped_1.0_20211018/   ← place corpus here (not included)
    ├── fce-released-dataset/          ← place corpus here (not included)
    ├── TOEFL11 2/                     ← place corpus here (not included)
    └── ICNALE_Edited Essays_2.1/     ← place corpus here (not included)
```

The scripts reference paths relative to this repository root as the working directory. **Run all Phase 1–5 scripts from `Engagement-Discourse-Treebank-Construction/`.**

---

## Prerequisites

```
Python >= 3.8
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

**Custom spaCy engagement model** (required for Phase 5c only):

```
en_engagement_RoBERTa_combined-cl_0.2.5
```

Expected path (hardcoded in `mutlisentToConll2.py`):
```
/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/
  Eng_Clause_span_finder/packages/
  en_engagement_RoBERTa_combined-cl_0.2.5/
  en_engagement_RoBERTa_combined/
  en_engagement_RoBERTa_combined-cl_0.2.5
```

If you are running on a different machine, update the `spacy.load(...)` path in `05_context_window_extraction/mutlisentToConll2.py`.

---

## Source Corpora

Raw corpus files are **not included** due to licensing restrictions. See [corpus_data/README_corpora.md](corpus_data/README_corpora.md) for acquisition instructions and the exact directory layout required under `corpus_data/`.

**Quick reference:**

| Corpus | Expected path | Tokenization needed |
|--------|--------------|---------------------|
| BAWE | `corpus_data/BAWE/CORPUS_UTF-8/*.xml` | No (pre-segmented) |
| MICUSP | `corpus_data/MICUSP_scraped_1.0_20211018/raw_xmls/*.xml` | Yes (Phase 1) |
| FCE | `corpus_data/fce-released-dataset/sentence_segmented_xml/*.xml` | No (pre-segmented) |
| TOEFL11 | `corpus_data/TOEFL11 2/.../tokenized/*.txt` | Yes (Phase 1, stanza=False) |
| ICNALE | `corpus_data/ICNALE_Edited Essays_2.1/EE_Unmerged_Unclassified/*_ORIG.txt` | Yes (Phase 1) |

---

## Pipeline

### Phase 1: Sentence Tokenization

**Working directory:** `1_corpus_sampling/`  
**Scripts:** `01_sentence_tokenization/`

Converts raw corpus files into sentence-indexed XML using the Stanza NLP library.

**BAWE and FCE are skipped** — they are already sentence-segmented.

#### MICUSP

```bash
python 01_sentence_tokenization/sent_tokenization_MICUSP.py
```

- Input: `corpus_data/MICUSP_scraped_1.0_20211018/raw_xmls/*.xml`
- Reads text from `<div class='dynacloud'>` elements
- Removes HTML tags: `<p>`, `</p>`, `<i>`, `</i>`, `<u>`, `</u>`, `<b>`, `</b>`, `<br/>`, `<q>`, `</q>`, `<a>`
- Applies `stanza.Pipeline(lang='en', processors='tokenize')`
- Output: `corpus_data/MICUSP_scraped_1.0_20211018/sent_tokenized_xml2/*.xml`

#### TOEFL11

```bash
python 01_sentence_tokenization/make_tokenized_xml.py  # select TOEFL11 section
```

- Input: `corpus_data/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized/*.txt`
- Metadata CSV: `corpus_data/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/index-training-dev.csv` (columns: `file, prompt, l1, score`)
- **No Stanza tokenization** (`stanza=False`): text is pre-tokenized, read as-is
- Output: `corpus_data/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized_xml/*.xml`

#### ICNALE

```bash
python 01_sentence_tokenization/make_tokenized_xml.py  # select ICNALE section
```

- Input: `corpus_data/ICNALE_Edited Essays_2.1/EE_Unmerged_Unclassified/*_ORIG.txt`
- Paragraph delimiter: `\n\n` (double newline)
- Applies Stanza tokenization (`stanza=True`)
- Output: `corpus_data/ICNALE_Edited Essays_2.1/sent_tokenized_xml/*.xml`

#### Output XML Format (all corpora)

```xml
<text>
  <meta>
    <filename>DOC_ID.xml</filename>
  </meta>
  <body>
    <p pid="1">
      <s pid="1" sid="1">First sentence text.</s>
      <s pid="1" sid="2">Second sentence text.</s>
    </p>
    <p pid="2">
      <s pid="2" sid="1">New paragraph, first sentence.</s>
    </p>
  </body>
</text>
```

Attributes: `pid` = paragraph index, `sid` = sentence index within paragraph.

---

### Phase 2: Sentence ID Extraction & Shuffling

**Script:** `02_sentence_id_extraction/extract_sent_ids.py`

Reads all sentence-tokenized XML files, extracts a unique identifier for each sentence, shuffles the list with a fixed seed, and saves both the original and shuffled lists as JSON.

```bash
# From 1_corpus_sampling/
python 02_sentence_id_extraction/extract_sent_ids.py
```

#### Sentence ID Formats

| Corpus | Format | Example |
|--------|--------|---------|
| MICUSP | `{docid}.xml_{pid}_{sid}` | `ENG.G0.24.1.xml_1_3` |
| TOEFL11 | `{docid}.xml_{pid}_{sid}` | `110018.xml_2_1` |
| ICNALE | `{docid}.xml_{pid}_{sid}` | `W_CHN_PTJ0_001_B1_0_ORIG.xml_1_2` |
| FCE | `{docid}.xml_answer2_{pid}_{sid}` | `doc123.xml_answer2_1_1` |
| BAWE | `{docid}.xml_s{sid}.{smax};p{pid}.{pmax}` | `0001b.xml_s1.6;p5.13` |

Note: **FCE uses answer2 task only.** The BAWE format encodes both sentence index and page/subsection range from the `n` attribute of `<s>` tags.

#### Random Seed

```python
seed = 1993   # hardcoded default in save_random()
```

#### Outputs (`sentence_ids/`)

| File | Contents |
|------|----------|
| `BAWE_20220427.json` | All BAWE sentence IDs, original order |
| `BAWE_20220427_shuffled.json` | Same, shuffled with seed=1993 |
| `MICUSP_20220427.json` | All MICUSP sentence IDs |
| `MICUSP_20220427_shuffled.json` | Shuffled |
| `FCE_answer2_20220327.json` | All FCE answer2 sentence IDs |
| `FCE_answer2_20220327_shuffled.json` | Shuffled |
| `TOEFL11_20220327.json` | All TOEFL11 sentence IDs |
| `TOEFL11_20220327_shuffled.json` | Shuffled |
| `ICNALE_20220327.json` | All ICNALE sentence IDs |
| `ICNALE_20220327_shuffled.json` | Shuffled |

JSON format: simple array of strings.
```json
["ENG.G0.24.1.xml_1_1", "ENG.G0.24.1.xml_1_2", ...]
```

---

### Phase 3: Per-Corpus Batching

**Script:** `03_batching/make_corpus_batches.py`

Groups the shuffled sentence IDs into fixed-size batches per corpus. Uses `utils.make_batch_list(sent_list, batch_size)` which splits a flat list into nested lists.

```bash
python 03_batching/make_corpus_batches.py
```

#### Batch Sizes

| Corpus | Batch size | Input JSON |
|--------|-----------|------------|
| BAWE | 750 | `sentence_ids/BAWE_20220427_shuffled.json` |
| MICUSP | 750 | `sentence_ids/MICUSP_20220427_shuffled.json` |
| FCE_answer2 | 150 | `sentence_ids/FCE_answer2_20220327_shuffled.json` |
| TOEFL11 | 200 | `sentence_ids/TOEFL11_20220327_shuffled.json` |
| ICNALE | 150 | `sentence_ids/ICNALE_20220327_shuffled.json` |

#### Corpus Data Directories (hardcoded in script)

```python
corpus_dirs = {
    'BAWE':        'corpus_data/BAWE/CORPUS_UTF-8',
    'MICUSP':      'corpus_data/MICUSP_scraped_1.0_20211018/sent_tokenized_xml',
    'FCE_answer2': 'corpus_data/fce-released-dataset/sentence_segmented_xml',
    'TOEFL11':     'corpus_data/TOEFL11 2/NLI_2013_Training_Data-Developement_Data/tokenized_xml',
    'ICNALE':      'corpus_data/ICNALE_Edited Essays_2.1/sent_tokenized_xml'
}
```

#### Output (`sentence_ids/batches/`)

```json
[
  ["ENG.G0.24.1.xml_1_1", "BIO.G1.02.1.xml_3_2", ...],  // batch 0: 750 BAWE sentences
  ["HIS.G2.01.1.xml_2_4", ...],                           // batch 1
  ...
]
```

---

### Phase 4: Combined Annotation Batches

**Script:** `03_batching/combine_into_annotation_batch.py`

Merges one batch from each of the 5 corpora into a single combined annotation batch (~2,000 sentences). Each combined batch is independently shuffled.

```bash
python 03_batching/combine_into_annotation_batch.py
```

#### Logic

1. Load all 5 `{Corpus}_batch.json` files
2. `zip` corresponding batches (batch[0] from BAWE + MICUSP + FCE + ICNALE + TOEFL11 → combined_batch[0])
3. Shuffle each combined batch: `random.seed(1993)`, then `random.shuffle()`
4. Save as `{N}_annotation_data.json` (numbered from 1)

#### Output (`annotation_batches/`)

```json
[
  ["ENG.G0.24.1.xml_1_1", "First sentence text from MICUSP."],
  ["W_CHN_PTJ0_001_B1_0_ORIG.xml_1_2", "Sentence from ICNALE."],
  ...
]
```

Each entry: `[sentence_id, sentence_text]`. ~2,000 entries per file, 64 files total in this run.

---

### Phase 5a: First-Round WebAnno TSV Generation

**Script:** `04_annotation_file_generation/making_annotation_file.py`

Converts annotation batch JSON files into WebAnno TSV 3.3 format using spaCy for tokenization, POS tagging, and dependency parsing.

```bash
python 04_annotation_file_generation/making_annotation_file.py
```

#### spaCy Model

```python
nlp = spacy.load('en_core_web_trf')
nlp.disable_pipe('ner')  # NER disabled for speed
```

#### Sentences per TSV File

**20 sentences** per output `.tsv` file.

#### WebAnno Header Layers

```
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=webanno.custom.ClauseBoundaryDetection|Clausetype
#T_SP=webanno.custom.Engagement|Engagement
#T_SP=webanno.custom.ModalSenseDisambiguation|ModalSense
#T_RL=de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency|DependencyType|flavor|BT_POS
```

#### Output Directories (under `batches_first_round/`)

| Directory | Format | Contents |
|-----------|--------|----------|
| `annotation_files_raw/` | plain text | Tokenised text only (no IDs) |
| `annotation_files_rawids/` | tab-separated | `{sentid}\t{token}` per line |
| `annotation_files_tsv/` | WebAnno TSV | No POS, no dependency |
| `annotation_files_tsv_pos/` | WebAnno TSV | With POS tags |
| `annotation_files_tsv_pos_dep/` | WebAnno TSV | With POS + all dependency relations |
| `annotation_files_tsv_pos_dep2/` | WebAnno TSV | With POS + filtered dependencies only |

#### Filtered Dependency List (for `pos_dep2`)

```python
["ROOT", "advcl", "appos", "relcl", "acl", "ccomp", "nsubj",
 "nsubjpass", "cc", "mark", "parataxis", "prep", "xcomp", "conj", "dep"]
```

#### File Naming

`batch1_A0.tsv`, `batch1_A1.tsv`, …, `batch1_Z9.tsv`  
(letter A–Z cycled, digit 0–9; up to 260 files per annotation batch JSON)

#### WebAnno TSV Token Format

```
#Text=First sentence text.
1-1	0-5	First	DT	DET	_	_	_	_
1-2	6-14	sentence	NN	NOUN	_	_	_	_
...
```

Columns: `sentence-token_id`, `char_offsets`, `token`, `POS_fine`, `POS_coarse`, `ClauseBoundaryDetection`, `Engagement`, `ModalSenseDisambiguation`, `[DependencyType|flavor|governor_id]`

---

### Phase 5b: Context Window Extraction (3-Sentence Windows)

**Script:** `05_context_window_extraction/add_contextTosentences.py`

For each target sentence in the first-round annotation batches, retrieves the surrounding sentences (previous, target, next) from the full corpus to create a 3-sentence context window.

```bash
python 05_context_window_extraction/add_contextTosentences.py
```

#### Random Seed

```python
random.seed(10)
```

#### Window Logic (`fixed_window=True`, default)

| Condition | Window |
|-----------|--------|
| `sid > 1` (not first sentence in paragraph) | [previous sentence, **target**, next sentence] |
| `sid == 1` (first sentence in paragraph) | [**target**, next sentence, next+1 sentence] |

#### Sentence ID Parsing (Regex)

**Learner corpora (ICNALE, TOEFL11, FCE):**
```python
r'(.+\.xml)_(?:answer\d+_)?(\d+)_(\d+)'
# groups: (docid, pid, sid)
```

**BAWE:**
```python
r'(.+\.xml)_s(\d+)\.(\d+);p(\d+)\.(\d+)'
# groups: (docid, sid, smax, pid, pmax)
```

#### Deduplication

The script tracks sentences already present in the first-round batch. Context sentences that are already annotated are excluded. A deduplication log is written to:
```
{output_dir}/__duplicates_ids.tsv
```

#### Input

`batches_first_round/annotation_files_rawids/` — raw ID files from Phase 5a

#### Output

`batches_first_round_w_context/` — one plain `.txt` file per target sentence  
Each file contains 2–3 lines (one sentence per line).

---

### Phase 5c: WebAnno TSV with Custom Engagement Model

**Script:** `05_context_window_extraction/mutlisentToConll2.py`

Converts the 3-sentence context text files into WebAnno TSV format using the custom `en_engagement_RoBERTa_combined-cl_0.2.5` spaCy model, which pre-fills clause boundary and engagement span predictions to assist annotators.

```bash
python 05_context_window_extraction/mutlisentToConll2.py
```

#### Custom spaCy Model

```python
nlp = spacy.load(
    '.../Eng_Clause_span_finder/packages/'
    'en_engagement_RoBERTa_combined-cl_0.2.5/'
    'en_engagement_RoBERTa_combined/'
    'en_engagement_RoBERTa_combined-cl_0.2.5'
)
```

Update this path if running on a different machine.

#### Random Seed

```python
random.seed(10)
```

#### Additional WebAnno Layer (vs Phase 5a)

```
#T_SP=webanno.custom.Supplementary|Supplementaryrhetoricalmoves
```

The `Engagement` layer also gains a `Hierarchy` attribute (Primary/Secondary).

#### Output Directories (under `batches_first_round_w_context/`)

| Directory | Files | Contents |
|-----------|-------|----------|
| `annotation_files_rawids/` | 1,999 `.txt` | `{sentid}\t{text}` per sentence (one line per sentence in window) |
| `annotation_files_tsv_pos/` | 1,999 `.tsv` | WebAnno TSV with pre-filled clause/engagement spans |

#### File Naming

`batch2_0001.tsv`, `batch2_0002.tsv`, …, `batch2_1999.tsv` (4-digit zero-padded)

#### Multi-Sentence TSV Format

Each file contains multiple `#Sentence.id=` headers — one per sentence in the 3-sentence window:

```
#FORMAT=WebAnno TSV 3.3
...headers...

#Sentence.id=POL.G2.02.1.xml_3_4
#Text=Previous sentence text.
1-1	0-8	Previous	...

#Sentence.id=POL.G2.02.1.xml_3_5
#Text=Target sentence text.
2-1	0-6	Target	...

#Sentence.id=POL.G2.02.1.xml_3_6
#Text=Next sentence text.
3-1	0-4	Next	...
```

---

### Phase 6: Manual Annotation in WebAnno

Import the `.tsv` files from `batches_first_round_w_context/annotation_files_tsv_pos/` into **WebAnno 3.3**.

#### Annotation Layers

| Layer | Attribute | Description |
|-------|-----------|-------------|
| `ClauseBoundaryDetection` | `Clausetype` | Span annotation marking clause type |
| `Engagement` | `Engagement`, `Hierarchy` | Engagement category + Primary/Secondary level |
| `ModalSenseDisambiguation` | `ModalSense` | Modal expression sense |
| `Supplementary` | `Supplementaryrhetoricalmoves` | Rhetorical move type |

#### Core Engagement Tags

`ATTRIBUTE`, `ENTERTAIN`, `CONCUR`, `COUNTER`, `DENY`, `ENDORSE`, `PRONOUNCE`, `MONOGLOSS`

#### Supplementary Tags

`JUSTIFYING`, `CITATION`, `QUOTED`, `TEXT-SEQUENCING`, `COMPARATIVE`, `EXEMPLIFYING`, `EXPOSITORY`, `SUMMATIVE`, `GOAL-ANNOUNCING`

#### Metadata Tags (not used in ML training)

`T-UNIT`, `ENDOPHORIC`, `ADDITIVE`, `MONOGLOSS_S`

Multiple annotators annotate each document. The annotated `.tsv` files are exported from WebAnno and passed to Phase 7.

---

### Phase 7: Post-Annotation Processing

All scripts are in `06_post_annotation/`. Run from the `engagement-corpus/` directory. Scripts reference paths relative to that working directory.

#### Step 7a — Annotation Integration

**Script:** `06_post_annotation/tsv_related/integrate_annotated.py`

- Matches annotated sentences by sentence ID across multiple annotator exports
- Replaces engagement and clause annotations with the reviewed/corrected versions
- Input: exported WebAnno TSV files from multiple annotators
- Output: merged TSV files in `engagement-corpus/data/1_annotated_files/`

#### Step 7b — Combine TSV Files

**Script:** `06_post_annotation/tsv_related/combine_into_one.py`

- Combines multiple per-batch TSV files into a single file
- Adds batch identifiers to each sentence record

#### Step 7c — Reconstruct Sentence IDs

**Script:** `06_post_annotation/tsv_related/adding_Sentid.py`

- Reconstructs the original sentence IDs in context-window annotation files
- Required because WebAnno may strip or reorder the `#Sentence.id=` metadata

#### Step 7d — Align Annotators (Inter-Annotator Agreement)

**Script:** `06_post_annotation/reliability/matching_webannotsv.py`

- Aligns token sequences from two annotators using **Needleman-Wunsch** global sequence alignment
- Also supports **Smith-Waterman** local alignment (implemented in `06_post_annotation/utils/aligner.py`)
- Creates matched token pairs with tags from both annotators
- Input: two sets of annotated TSV files
- Output: JSON file of aligned token pairs in `engagement-corpus/data/input_for_reliability/`

#### Step 7e — Reliability Metrics

**Script:** `06_post_annotation/reliability/run_reliability.py`

- Orchestrates the full reliability workflow for all annotation batches (A–E)
- Calls `json2accuracy.py` → computes **Precision, Recall, F1, Cohen's Kappa** per engagement tag class
- Disagreed sentences saved to: `engagement-corpus/data/disagreed_sentences/`

#### Step 7f — Manual Adjudication

*(No script — manual process in WebAnno)*

- Adjudicator reviews all disagreements in `engagement-corpus/data/disagreed_sentences/`
- Final adjudicated annotations exported as WebAnno TSV
- Output saved to: `engagement-corpus/data/adjudicated/`

#### Step 7g — Linear Tag Conversion & Hierarchy Encoding

**Script:** `06_post_annotation/adjudication_helper/tsv2LinearTags.py`

- Converts adjudicated engagement tags to a linear (flat) tag format
- Separates `ENGAGEMENT` tags from `JUSTIFYING`/`CITATION` (supplementary) tags
- Encodes hierarchy: `Primary` or `Secondary` appended as suffix to tag name
  - e.g., `ENTERTAIN_Primary`, `ATTRIBUTE_Secondary`
- Output directories: `engagement-corpus/data/linearTag/` (with RealID and tsvID variants)

#### Step 7h — IOB/BIO Format for ML Training

**Script:** `06_post_annotation/post_process/tsv2iob2.py`

- Converts WebAnno TSV → **IOB (BIO) sequence tagging** format
- Combines engagement (`engmt`) and supplementary (`spl`) layers
- Encodes hierarchy directly in tag names: `ENGAGEMENT_Primary` vs `ENGAGEMENT_Secondary`
- Strips meta-annotations (T-UNIT, QUOTED, ADDITIVE, ENDOPHORIC, MONOGLOSS_S, etc.)
- Handles multiword spans using bracketed indices from WebAnno format
- Output: `engagement-corpus/data/iob_data/`

IOB format example:
```
The     O
study   O
argues  B-ATTRIBUTE_Primary
that    I-ATTRIBUTE_Primary
...
```

#### Step 7i — Cross-Validation Splits

**Script:** `06_post_annotation/post_process/make_kfold.py`

- Creates **5-fold cross-validation** splits from IOB data
- Distributes documents evenly across folds with balanced tag distribution
- Generates `train.txt`, `dev.txt`, `test.txt` for each fold
- Outputs split statistics (tag counts per split)
- Output: `engagement-corpus/data/iob_data/` (fold subdirectories)

#### Step 7j — Final HuggingFace Dataset

**Script:** `academic_written_corpus/create_huggingface_dataset.py`  
*(located in `1_corpus_sampling/academic_written_corpus/`)*

- Parses adjudicated TSV files into structured JSON
- Splits into train/validation/test sets
- Output: `academic_written_corpus/20230104/json/{train,validation,test}.json`

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Global random seed | `1993` | Phases 2, 3, 4 |
| Context window seed | `10` | Phases 5b, 5c |
| Sentences per TSV file (Phase 5a) | `20` | `making_annotation_file.py` |
| BAWE batch size | `750` | `make_corpus_batches.py` |
| MICUSP batch size | `750` | `make_corpus_batches.py` |
| FCE_answer2 batch size | `150` | `make_corpus_batches.py` |
| TOEFL11 batch size | `200` | `make_corpus_batches.py` |
| ICNALE batch size | `150` | `make_corpus_batches.py` |
| Combined batch size | ~2,000 | `combine_into_annotation_batch.py` |
| Context window size | 3 sentences | `add_contextTosentences.py` |
| FCE task used | answer2 only | `extract_sent_ids.py` |
| Cross-validation folds | 5 | `make_kfold.py` |
| Annotation platform | WebAnno 3.3 | Phase 6 |

---

## Annotation Layers

### WebAnno Custom Layers

```
webanno.custom.ClauseBoundaryDetection  → Clausetype
webanno.custom.Engagement               → Engagement, Hierarchy
webanno.custom.ModalSenseDisambiguation → ModalSense
webanno.custom.Supplementary            → Supplementaryrhetoricalmoves
```

### Standard Layers

```
de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS  → PosValue, coarseValue
de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency → DependencyType, flavor
```

---

## Output Files

| Artifact | Location | Format |
|----------|----------|--------|
| Shuffled sentence ID lists | `sentence_ids/*_shuffled.json` | JSON string array |
| Per-corpus batched IDs | `sentence_ids/batches/*.json` | JSON nested list |
| Combined annotation batches | `annotation_batches/*.json` | JSON array of [id, text] |
| First-round annotation TSVs | `batches_first_round/annotation_files_tsv_pos/` | WebAnno TSV 3.3 |
| Context-window raw IDs | `batches_first_round_w_context/annotation_files_rawids/` | `id\ttext` per line |
| Context-window TSVs | `batches_first_round_w_context/annotation_files_tsv_pos/` | WebAnno TSV 3.3 |
| Adjudicated annotations | `engagement-corpus/data/adjudicated/` | WebAnno TSV 3.3 |
| IOB training data | `engagement-corpus/data/iob_data/` | BIO token format |
| Final dataset | `academic_written_corpus/20230104/json/` | JSON train/val/test |
