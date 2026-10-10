# Gunshot / engine alerts: check on open-licence recordings (YAMNet groups)
Windows are 3 s, the loudest ones of each clip. Sources: Wikimedia Commons (open licences) and Xeno-canto bird recordings.

| sound | windows | gun alert (>=0.35) | engine alert (>=0.3 and animal<0.5) |
|---|---|---|---|
| gunshots (3 usable clips) | 24 | 37.5% (2 of 3 clips detected) | 0% |
| chainsaws / engines | 56 | 0% | 48% (34% at old threshold 0.5) |
| thunder | 158 | 1.9% | 0% |
| rain | 80 | 0% | 0% |
| wind | 69 | 0% | 0% |
| door slam | 1 | 0% | 0% |
| birds (Xeno-canto) | 1169 | 0.2% | 0% |

Small sample: recall numbers are indicative only. The gun threshold does not change recall between 0.1 and 0.5; false alarms on thunder grow below 0.35.
