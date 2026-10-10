#!/bin/bash
# v6 chain, memory-friendly: few worker processes (the machine had about 1.4 GB free), caches are kept, nothing is deleted.
cd /c/Users/Timur/Documents/wetlandear-models
PY=./venv/Scripts/python.exe
export NPROC=3
echo "[chain8] features (changed species only)"; $PY -I build_feats2.py 3 > chain8_feats2.log 2>&1
echo "[chain8] logits"; $PY -I build_logits.py > chain8_logits.log 2>&1
echo "[chain8] noisy"; $PY -I build_noisy.py 3 > chain8_noisy.log 2>&1
echo "[chain8] train + export"; $PY -I train_v5.py --export > chain8_train.log 2>&1
$PY -I make_species_js.py > chain8_js.log 2>&1
echo CHAIN8_DONE
