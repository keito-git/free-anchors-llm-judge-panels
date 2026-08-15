# Adding Judges Is Not Enough — Reproduction Package

Code, measurements and collected judge labels for the paper on identifiability
of LLM judge panels.

## Contents

```
code_public/   52 scripts, one per experiment, plus README, LICENSE, requirements
results/       53 JSON files, the measurements every number in the paper comes from
data/          the second panel we collected, and the judge-to-family mapping
```

Start with [`code_public/README.md`](code_public/README.md): it explains the
layout, how paths are resolved, which datasets must be downloaded separately,
and what each script does.

## Quick check

Every analysis runs on CPU from the released data. For example,

```
pip install -r code_public/requirements.txt
python code_public/e28_second_panel_analysis.py
```

reproduces the second panel's family sweep, where the first family removes
41.6% of the identification set and the six after it remove nothing measurable.

## Data

The second panel (`data/second_panel_treccovid/`) is released here: 294 items of
BEIR TREC-COVID labelled by nine judges from seven providers at temperature
zero. Regenerating those labels needs provider credentials, but every analysis
in the paper runs from the labels as released.

The other collections are public but not redistributed here; see
[`code_public/README.md`](code_public/README.md) for sources and terms. The
NTCIR-10 INTENT-2 data in particular may not be redistributed and must be
requested from its organizers.

## License

MIT for the code, see [`code_public/LICENSE`](code_public/LICENSE). The input
collections keep their own terms.
