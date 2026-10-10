#!/bin/bash
cd /c/Users/Timur/Documents/wetlandear-models
PY=./venv/Scripts/python.exe
rm -rf feats3 feats_noisy
echo "[chain6] logits"; $PY -I build_logits.py > chain6_logits.log 2>&1
echo "[chain6] noisy"; $PY -I build_noisy.py 14 > chain6_noisy.log 2>&1
echo "[chain6] train + export"; $PY -I train_v5.py --export > chain6_train.log 2>&1
$PY -I make_species_js.py > chain6_js.log 2>&1
echo CHAIN6_DONE
