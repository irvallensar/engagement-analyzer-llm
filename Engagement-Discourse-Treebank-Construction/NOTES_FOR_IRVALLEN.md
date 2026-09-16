# Notes for Irvallen: Data Structure of the EDT Construction Pipeline

These notes explain what data lives in this repository, which sentences ended up in the Engagement Discourse Treebank (EDT), which were not used, and where to find more information.

---

## 1. The Big Picture

The EDT is built from five source corpora of academic writing. Think of the pipeline as a funnel:

```
5 source corpora (~592K sentences total)
        ↓  [Phase 2–4: shuffle & sample]
64 annotation batches (~128K sentences)
        ↓  [Phase 5–6: generate TSVs, annotate in WebAnno]
EDT (the annotated subset — manual annotation, Phase 6)
```

Your research compares LLM-generated synthetic data against **natural corpus sampling**. This pipeline gives you the entire chain: the raw corpora, the sampled pools, and the annotated gold data.

---

## 2. The Five Source Corpora

All raw corpus files are in `corpus_data/` (not included in git due to licensing).

| Corpus | Type | Total sentences | Files location |
|--------|------|----------------|----------------|
| BAWE | Expert academic writing (UK undergrad) | 269,413 | `corpus_data/BAWE/CORPUS_UTF-8/` |
| MICUSP | Expert academic writing (US upper-level) | 97,546 | `corpus_data/MICUSP_scraped_1.0_20211018/sent_tokenized_xml2/` |
| FCE (answer2 only) | Learner writing (Cambridge CLC) | 34,645 | `corpus_data/fce-released-dataset/sentence_segmented_xml/` |
| TOEFL11 | Learner writing (11 L1 backgrounds) | 181,314 | `corpus_data/TOEFL11 2/.../tokenized_xml/` |
| ICNALE | Learner writing (Asian L2 English) | 9,565 | `corpus_data/ICNALE_Edited Essays_2.1/sent_tokenized_xml/` |
| **Total** | | **592,483** | |

Sentence IDs and sentence texts for all 592K sentences are stored in `output/sentence_ids/` (JSON files, one per corpus). These were extracted in Phase 2.

---

## 3. Which Sentences Are in the 64 Annotation Batches

The pipeline shuffled all sentence IDs (random seed = 1993), then drew fixed-size samples per corpus per batch. A combined batch contains ~2,000 sentences drawn proportionally from all five corpora.

**How many batches were created:** 64 combined batches (limited by ICNALE, which only has enough sentences for 64 batches of 150).

**Coverage in the 64 batches:**

| Corpus | Sentences selected | % of corpus | Sentences left out |
|--------|-------------------|-------------|-------------------|
| BAWE | 48,000 | 17.8% | **221,413** |
| MICUSP | 48,000 | 49.2% | **49,546** |
| FCE (answer2) | 9,600 | 27.7% | **25,045** |
| TOEFL11 | 12,800 | 7.1% | **168,514** |
| ICNALE | 9,565 | 100.0% | 0 |
| **Total selected** | **127,965** | | **464,518 left out** |

**Where to find the batch data:**
- Per-corpus batches: `output/sentence_ids/batches/{Corpus}_batch.json`
- Combined annotation batches: `output/annotation_batches/{N}_annotation_data.json` (N = 1–64)

Each entry in a combined batch file is `[sentence_id, sentence_text]`.

**Important:** Because the IDs were shuffled randomly before batching, the 127K selected sentences are a representative random sample — not the "beginning" of any corpus. The ~465K left-out sentences are also a random remainder.

---

## 4. Which Sentences Are in the EDT (Actually Annotated)

The pipeline automated everything up to generating WebAnno annotation files (Phase 5). **Manual human annotation (Phase 6) was then done in WebAnno** and is not automated. The post-annotation data (Phase 7 outputs) lives in a separate directory (`engagement-corpus/`) outside this repo.

The EDT was built in **two separate annotation rounds**:

---

### Round 1 — Random stratified sample (Batch1 in WebAnno)

- **Source:** `output/annotation_batches/1_annotation_data.json` only (2,000 sentences)
- **Of the 64 annotation batch files, only batch 1 was used for this round.** Batches 2–64 were never annotated as whole batches.
- The 2,000 sentences were split into 100 WebAnno TSV files (20 sentences each), named `batch1_A0` through `batch1_J9`.
- **Only 35 of those 100 TSV files were adjudicated**, covering **700 sentences**:

| TSV group | Sentence range in batch JSON | Sentences |
| --------- | ---------------------------- | --------- |
| A0–A9     | 1–200                        | 200       |
| B0–B2     | 201–260                      | 60        |
| D0, D7    | 601–620, 741–760             | 40        |
| H0–H9     | 1401–1600                    | 200       |
| J0–J9     | 1801–2000                    | 200       |
| **Total** |                              | **700**   |

The remaining 1,300 sentences (TSV groups B3–B9, C0–C9, D1–D6, D8–D9, E, F, G, I) were generated but never annotated.

---

### Round 2 — Construction-specific targeted sample (Batch2 in WebAnno)

- **Source:** `1_corpus_sampling/batches_cx_specific/sentence_ids.json` — 2,002 sentences hand-selected because they contain specific engagement constructions (NOT a random stratified sample).
- These target sentences were given 3-sentence context windows and processed through Phase 5c to pre-fill engagement predictions → output in `1_corpus_sampling/batches_cx_spec_w_contexts3/` (1,996 context window TSV files named `batch2_0000.tsv`…`batch2_1995.tsv`).
- **592 of those 1,996 context windows were adjudicated.**
- The 592 target sentences span the full corpus and multiple annotation batch numbers — they were not constrained to `1_annotation_data.json`. About 503 of them happen to appear in `annotation_batches/` files 3–64, while the rest (~1,493 target sentences) were drawn directly from the raw corpora.

---

### Summary: annotation_batches file status

| File(s)                                              | Status                                                                                                                                         |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `1_annotation_data.json`                             | **Partially annotated** — 700 of 2,000 sentences in EDT Round 1                                                                                |
| `2_annotation_data.json` – `64_annotation_data.json` | **Not annotated as batches** — ~503 individual sentences from these files appear incidentally in the Round 2 construction-specific sample |

The post-annotation data is in `engagement-corpus/data/` (outside this repo). Ask Prof. Eguchi for:

- `engagement-corpus/data/adjudicated/` — adjudicated WebAnno TSV files (Batch1 and Batch2 subdirectories)
- `engagement-corpus/data/iob_data/` — IOB/BIO format training data
- `engagement-corpus/data/iob_data/5_fold_test/` — 5-fold cross-validation splits

The README in this repo describes the full Phase 7 pipeline (see "Phase 7: Post-Annotation Processing") so you know what processing was applied.

---

## 5. Three Tiers of Data for Your Research

For a study comparing LLM synthetic data against natural corpus sampling, the pipeline gives you three natural tiers:

| Tier | Sentences | Description | Location |
|------|-----------|-------------|----------|
| **Gold (EDT)** | ~2,000+ | Human-annotated engagement labels | Outside this repo — ask Prof. Eguchi |
| **Sampled pool** | 127,965 | Drawn from corpora, prepared for annotation, most without labels | `output/annotation_batches/` (64 JSON files) |
| **Unsampled remainder** | 464,518 | Never drawn into any batch | `output/sentence_ids/` + raw corpus files |

These tiers let you ask: does LLM synthetic data best approximate the gold tier, the broad sampled pool, or the full unsampled corpus?

---

## 6. Sentence ID Format (How to Identify Which Corpus a Sentence Comes From)

| Corpus | ID pattern | Example |
|--------|-----------|---------|
| MICUSP | `{doc}.xml_{pid}_{sid}` | `ENG.G0.24.1.xml_1_3` |
| TOEFL11 | `{doc}.xml_{pid}_{sid}` | `110018.xml_2_1` |
| ICNALE | `W_{country}_...xml_{pid}_{sid}` | `W_CHN_PTJ0_001_B1_0_ORIG.xml_1_2` |
| FCE | `doc{N}.xml_answer2_{pid}_{sid}` | `doc123.xml_answer2_1_1` |
| BAWE | `{doc}.xml_s{sid}.{smax};p{pid}.{pmax}` | `0001b.xml_s1.6;p5.13` |

A quick way to determine corpus from an ID:
- Starts with `W_` + contains `_ORIG` → **ICNALE**
- Starts with `doc` + contains `answer` → **FCE**
- Contains `.xml_s` + `;p` → **BAWE**
- All-numeric before `.xml_` → **TOEFL11** (e.g., `110018.xml_`)
- Letters before `.xml_`, no semicolon → **MICUSP**

---

## 7. Key Files to Start With

| Task | File(s) |
|------|---------|
| See all sentences from Batch 1 (the annotated batch) | `output/annotation_batches/1_annotation_data.json` |
| See the full sentence pool for one corpus | `output/sentence_ids/BAWE_20220427.json` (or MICUSP/FCE/TOEFL11/ICNALE) |
| See the sampled batches for one corpus | `output/sentence_ids/batches/BAWE_batch.json` |
| Understand the pipeline | `README.md` (in this directory) |
| Understand corpus acquisition | `corpus_data/README_corpora.md` |

---

## 8. Running the Pipeline Yourself

A verified copy of the pipeline is in `../Engagement-Discourse-Treebank-Construction-copy/`. It has a `.venv/` with all dependencies installed. To re-run from Phase 3 onward:

```bash
cd ../Engagement-Discourse-Treebank-Construction-copy
PYTHONPATH=. .venv/bin/python 03_batching/make_corpus_batches.py
PYTHONPATH=. .venv/bin/python 03_batching/combine_into_annotation_batch.py
PYTHONPATH=. .venv/bin/python 04_annotation_file_generation/making_annotation_file.py
PYTHONPATH=. .venv/bin/python 05_context_window_extraction/add_contextTosentences.py
```

Outputs of Phases 3–5b have been verified to be byte-for-byte reproducible (random seed 1993 gives identical shuffles every time).

Note: Phase 5a POS tags may differ slightly if your `en_core_web_trf` version differs from the original. Tokenization (which is what matters for sentence identity) is identical.




