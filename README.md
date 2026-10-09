# WetlandEar ML

How the WetlandEar species classifier was built, evaluated and exported. This repository holds the **code and the measurements**, not the audio (Xeno-canto recordings are fetched by script and stay on the training machine) and **no keys**.

The app that uses the model: https://app.wetlandear.tech (repo `wetlandear-app`, files `api/identify.py` and `models/head.npz`).

## Idea in one paragraph
BirdNET V2.4 (Cornell Lab / TU Chemnitz, CC BY-NC-SA) turns every 3-second window of sound into a 1024-number "fingerprint" and a score for 6,522 labels. It does not know many species of Kazakhstan (for example Pelecanus crispus is absent). We train a small logistic-regression head on the fingerprints of Xeno-canto recordings of Kazakh species, then fuse it with BirdNET's own scores and calibrate the confidence. Everything runs on CPU in a few milliseconds per window.

## Pipeline
1. `fetch_xc4.py`, `fetch_xc5.py`: download recordings from Xeno-canto API v3 (needs a free key in the `XC_KEY` environment variable). Recordings licensed *No-Derivatives* are excluded; a quality filter and a per-country cap keep the data varied.
2. `build_feats2.py`: for every recording take the 14 strongest 3-second windows and 3 background windows (hop 1.5 s), compute the BirdNET fingerprint (second-to-last layer) and BirdNET's top species. Parallel, cached per species folder.
3. `build_logits.py`: BirdNET scores for our own species on the same windows (for fusion).
4. `train_eval2.py`, `stack_eval.py`: compare feature sets and classifiers, always with **cross-validation split by recording** (no recording is ever in both train and test).
5. `export_head2.py`: choose the final configuration, fit on all data, fit the temperature on out-of-fold probabilities, write `head.npz` and the sources list.
6. `calib_eval.py`, `fusion_eval*.py`, `eval2.py`, `evaluate.py`: calibration, open-set behaviour, window aggregation.

## Results (cross-validation by recording)
| Version | Species | Recordings | BirdNET alone | WetlandEar head | Per recording |
|---|---|---|---|---|---|
| v1 | 24 | 267 | 68.8% | 78.6% | n/a |
| v2 | 123 | 1,706 | 59.5% | 74.4% | 82.6% |
| v3 | 213 | 2,960 | 56.9% | 71.5% | 81.8% |
| v4 (current) | 222 | 6,214 | 58.2% | **74.7%** | **85.2%** |

(windows = 3-second fragments; "per recording" = mean probability over the fragments of one recording)

What we tried and rejected (v2): a small neural network (MLP) was clearly worse (56.5%); adding YAMNet fingerprints did not help (72.4% vs 72.5%); stacking BirdNET scores as extra features helped per-recording (83.3%) but not per-window; late fusion with weight 0.15 gave the best window accuracy.

Calibration of the final confidence (out of fold): at confidence >= 0.5 about 93% of suggestions are right and they cover 68% of windows; at >= 0.9 about 98%. The app fills a species in automatically only from 0.5.

v3 covers 213 species. WetlandEar beats BirdNET on 183 of them, ties on 11 and is worse on 19 (for example Imperial Eagle, Saker Falcon, Egyptian Vulture). The absolute accuracy is lower than v2 because the task is harder (213 species instead of 123); the lead over BirdNET is about the same (14.6 points vs 14.9 in v2) on a much larger species list, and the full per-species table is published at https://wetlandear.tech/metrics.

## Honest limits
- Test data are Xeno-canto recordings (often clean). Field recordings with wind, water and distance will score lower.
- Species with few recordings are less reliable; the weakest in v3 are mostly rarely vocal raptors and pelicans (Saker Falcon, Egyptian Vulture, Imperial Eagle, Great White Pelican) and a few look-alike passerines.
- Pelecanus crispus (Dalmatian pelican) has no usable training recording yet.
- Models are for non-commercial research and education (BirdNET and Xeno-canto licenses).

## Reproduce
```
python -m venv venv && venv\Scripts\pip install -r requirements.txt
set XC_KEY=<your Xeno-canto key>
python fetch_xc4.py        # recordings
python build_feats2.py 8   # features, 8 processes
python build_logits.py
python train_eval2.py      # honest comparison
python export_head2.py     # writes head.npz
```
Paths at the top of each script point to the author's machine (`C:\Users\Timur\Documents\wetlandear-models`); change `BASE` first. BirdNET V2.4 files come from the official BirdNET-Analyzer download (see the app repo's `models/NOTICE.md`).

## Fine-tuning on confirmed recordings from the app
`retrain_with_app.py` merges the exports from the app (Data > Export for training) with the Xeno-canto features, re-evaluates and re-exports the head. Confirmed field recordings are weighted higher than Xeno-canto ones.

## v4: more data and robustness to noise
- 6,214 recordings (`fetch_xc6.py` tops every species up to 32 recordings, weak species up to 60).
- `build_noisy.py` makes two noisy copies of the six strongest 3-second fragments of every recording: reed rustle, wind, water, rain (synthetic), real field background and other birds singing (from other recordings), at -3 to +12 dB signal-to-noise. `train_v4.py` compares a head trained on clean data only with one trained on clean plus noisy data, both tested on clean and noisy fragments of held-out recordings.
- Result on noisy fragments: BirdNET 52.5% (restricted to the same species list), clean-only head 64.4%, clean plus noisy head **68.1%**; on clean fragments 74.7% vs 75.1% (a small price). Weakest: other birds singing at the same time (59%) and strong background (61%). Per-type and per-SNR tables: `reports/v4_train_and_robustness.log`, https://wetlandear.tech/metrics.
- 208 species are better than BirdNET, 6 equal, 8 worse (Imperial Eagle by 51 points, Greater Scaup by 5, the rest by 1-2).
- Limits: the noise is simulated, real field audio may behave differently. Weak species (few or poor recordings): Greater Scaup, Saker Falcon, Imperial Eagle, Egyptian Vulture, Steppe Eagle, Smew, Great White Pelican.
- `final_chain.sh` runs the whole rebuild; `make_species_js.py` writes the species name table for the app.
