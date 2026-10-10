#!/bin/bash
# v6 final step: rebuild the 6 broken score caches, refuse to train if any folder is still misaligned, then train and export.
cd /c/Users/Timur/Documents/wetlandear-models
PY=./venv/Scripts/python.exe
export NPROC=3 NJOBS=5
echo "[chain9] logits for the repaired folders"; $PY -I build_logits.py > chain9_logits.log 2>&1
echo "[chain9] alignment check"; ./venv_paper/Scripts/python.exe check_align.py > chain9_align.log 2>&1
if ! grep -q "skipped folders: 0" chain9_align.log; then echo "ALIGNMENT PROBLEM, stopping"; cat chain9_align.log; exit 1; fi
echo "[chain9] train + export"; $PY -I train_v5.py --export > chain9_train.log 2>&1
$PY -I make_species_js.py > chain9_js.log 2>&1
echo CHAIN9_DONE
