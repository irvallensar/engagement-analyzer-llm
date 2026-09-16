import sys
import os

# Re-export functions from the canonical making_annotation_file.py so that
# mutlisentToConll2.py can import them via the original module path.
_here = os.path.dirname(__file__)
_repo_root = os.path.dirname(os.path.dirname(_here))
_phase5a = os.path.join(_repo_root, "04_annotation_file_generation")
sys.path.insert(0, _phase5a)

from making_annotation_file import (  # noqa: F401
    write_webanno_tsv,
    preprocess,
    spacy_tokenize,
    spacy_tokenize_tsv,
)
