#!/bin/bash
cd /c/Users/Timur/Documents/wetlandear-models
until grep -q DOWNLOADED fetch_xc6.log; do sleep 20; done
PY=./venv/Scripts/python.exe
echo "[chain] features clean" ; $PY -I build_feats2.py 14 > chain_feats2.log 2>&1
echo "[chain] logits"; rm -rf feats3; $PY -I build_logits.py > chain_logits.log 2>&1
echo "[chain] noisy"; $PY -I build_noisy.py 14 > chain_noisy.log 2>&1
echo "[chain] train + export"; $PY -I train_v4.py --export > chain_train.log 2>&1
echo CHAIN_DONE
