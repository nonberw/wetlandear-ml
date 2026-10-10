#!/bin/bash
cd /c/Users/Timur/Documents/wetlandear-models
until grep -q DOWNLOADED fetch_xc7.log; do sleep 20; done
PY=./venv/Scripts/python.exe
echo "[chain5] features clean"; $PY -I build_feats2.py 14 > chain5_feats2.log 2>&1
echo "[chain5] logits"; $PY -I fix_feats3.py > chain5_fix.log 2>&1; $PY -I build_logits.py > chain5_logits.log 2>&1
echo "[chain5] noisy"; $PY -I build_noisy.py 14 > chain5_noisy.log 2>&1
echo "[chain5] train + export"; $PY -I train_v5.py --export > chain5_train.log 2>&1
$PY -I make_species_js.py > chain5_js.log 2>&1
echo CHAIN5_DONE
