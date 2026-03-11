You are an expert linguistic annotator for ENGAGEMENT analysis based on the Appraisal framework.

TASK:
Analyze the sentence and output a JSON array of objects strictly matching this schema. If there are no markers, output: []

[
  {
    "text": "exact marker string",
    "label": "MONOGLOSS | DENY | COUNTER | PROCLAIM | ENTERTAIN | ATTRIBUTION | CITATION | SOURCES | ENDOPHORIC | JUSTIFYING",
    "context_before": "2-3 words immediately preceding the marker (or empty string)"
  }
]

DEFINITIONS:
1. MONOGLOSS: Main verb of a definitive factual statement (e.g., 'is', 'proves').
2. DENY: Literal syntactic negators (e.g., 'no', 'not', 'never', 'fail to'). 
3. COUNTER: Transition words of contrast (e.g., 'however', 'although', 'but').
4. PROCLAIM: Words showing authorial backing (e.g., 'undoubtedly', 'in fact', 'conclude').
5. ENTERTAIN: Modal verbs, conditionals, or hedges (e.g., 'might', 'could', 'appear', 'often').
6. ATTRIBUTION: The reporting verb phrase (e.g., 'asserts', 'According to').
7. CITATION: Explicit in-text citations. Extract the ENTIRE parenthetical (e.g., '(Smith, 2019)').
8. SOURCES: The exact noun/pronoun making a claim (e.g., 'researchers', 'They'). 
9. ENDOPHORIC: Document cross-references (e.g., 'in Figure 1', 'Table 8').
10. JUSTIFYING: Causal transition words (e.g., 'Thus', 'Therefore').

Example 1
Sentence: "Although the results are promising , they might not apply to all cases ."
[
  {
    "text": "Although", 
    "label": "COUNTER",
    "context_before": ""
  },
  {
    "text": "might", 
    "label": "ENTERTAIN",
    "context_before": "they"
  },
  {
    "text": "not apply", 
    "label": "DENY",
    "context_before": "might"
  }
]

Example 2
Sentence: "Christen ( 1999 ) strongly asserts that the study area is affected , and Davies ( 2002 , p107 ) warns against this ."
[
  {
    "text": "Christen ( 1999 )", 
    "label": "CITATION",
    "context_before": ""
  },
  {
    "text": "strongly asserts", 
    "label": "ATTRIBUTION",
    "context_before": ") "
  },
  {
    "text": "Davies ( 2002 , p107 )", 
    "label": "CITATION",
    "context_before": "and"
  },
  {
    "text": "warns", 
    "label": "ATTRIBUTION",
    "context_before": ") "
  }
]

Example 3
Sentence: "Researchers emphasize that the data proves the outcome ."
[
  {
    "text": "Researchers", 
    "label": "SOURCES",
    "context_before": ""
  },
  {
    "text": "emphasize", 
    "label": "ATTRIBUTION",
    "context_before": "Researchers"
  },
  {
    "text": "proves", 
    "label": "MONOGLOSS",
    "context_before": "data"
  }
]

Now annotate the following.

Sentence:
{sentence}
