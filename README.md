# Asteroid Threat Classifier

A beginner Python machine-learning project that predicts whether an asteroid has the **potentially hazardous asteroid (PHA)** flag in a NASA-derived catalog. The included historical dataset lets you train, evaluate, and try a prediction without an API key.

The model has three inputs: absolute magnitude (H), Earth-relative speed in km/s, and the miss distance of one Earth encounter in astronomical units (au). A smaller H means a brighter object. NASA’s formal PHA definition uses H and the minimum distance between the two orbital paths (MOID). A single encounter’s miss distance measures something different. We leave MOID out of the training data and learn how far the three available measurements can take us. The output is a catalog classification. Computing collision probabilities is a separate orbital-analysis task. [NASA/JPL definition](https://cneos.jpl.nasa.gov/about/neo_groups.html)

## Run the demo

Tested with Python 3.12. From the extracted project folder on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python predict.py --magnitude 21 --speed 18 --distance 0.06
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it in Command Prompt with `.venv\Scripts\activate.bat`.

Training prints the test results and creates `model.joblib`. The last command uses hypothetical measurements and prints a predicted label and a classifier score. Compare that score with the learned cutoff. It is a class-weighted decision-tree score.

## What did the model learn?

The historical CSV contains **3,692 distinct asteroids**, including **576 flagged objects**. The table shows results for 739 held-out asteroids. Of these, 115 had a positive catalog flag.

| Method | Flags caught | Flags missed | False alarms | Correct negatives | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision tree | 110 | 5 | 235 | 389 | 95.7% | 31.9% |
| Brightness rule, H ≤ 22 | 112 | 3 | 263 | 361 | 97.4% | 29.9% |
| Flag every asteroid | 115 | 0 | 624 | 0 | 100.0% | 15.6% |

**Recall** asks “Of the 115 flagged objects, how many did we catch?” The tree caught 110. **Precision** asks “Of the 345 objects the tree flagged, how many had a positive catalog label?” Again the answer is 110. Compared with the brightness rule, the tree made 28 fewer false alarms and missed two additional flagged asteroids.

## How training works

1. Keep the earliest available Earth encounter for each asteroid ID. Repeated visits by one object can otherwise appear in both training and testing.
2. Split by unique asteroid into training (2,214), validation (739), and test (739). Stratification keeps approximately the same fraction of flagged objects in each set. The random seed is 42.
3. Fit a decision tree on the training set. Its maximum depth is four. Balanced class weights give the rarer positive class more influence.
4. Try cutoffs on the validation set. Among those reaching at least 90% recall, choose the one with the highest precision. Break ties by recall, then by cutoff.
5. Freeze the tree and cutoff, then evaluate on the test set. The 90% goal applies to validation. The test result measures generalization to unseen IDs from the historical sample.

## Fetch a different dataset from NASA

```bash
python fetch_data.py --start 2024-01-01 --end 2024-03-31 --out data/feed.csv
python train.py --data data/feed.csv --model feed-model.joblib
python predict.py --model feed-model.joblib --magnitude 21 --speed 18 --distance 0.06
```

The downloader requests seven dates at a time, uses Earth approaches matching each feed date, and writes one row per asteroid. It uses the public `DEMO_KEY` by default. For larger downloads, get a key from NASA and set the `NASA_API_KEY` environment variable locally. Keep the key out of source control. If a request fails, the previous output CSV stays intact. Training needs at least ten distinct objects in each class for its stratified split. [NASA Open APIs](https://api.nasa.gov/) · [NeoWs catalog entry](https://data.nasa.gov/dataset/asteroids-neows-api)

## Data provenance

The packaged demo uses a **historical archive**. Its source is the `shrutimehta/nasa-asteroids-classification` dataset, retrieved from the pinned mirror below. Its 4,687 Earth encounter rows span 1995-01-01 to 2016-09-08. Keeping each ID’s earliest tabulated encounter removed 995 repeat rows. The historical flag is the target; individual labels and orbital estimates can change in later catalogs. This random-split benchmark evaluates this sample rather than a future observing period.

| Packaged column | Historical source column | Use |
| --- | --- | --- |
| asteroid_id | Neo Reference ID | Identify repeat encounters |
| approach_date | Close Approach Date | Select the earliest sampled encounter |
| absolute_magnitude | Absolute Magnitude | Input |
| speed_km_s | Relative Velocity km per sec | Input |
| miss_distance_au | Miss Dist.(Astronomical) | Input |
| pha | Hazardous (True → 1, False → 0) | Target |

[Original dataset](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification) · [Pinned historical CSV](https://github.com/jmattick/asteroids/blob/607d584e6e8ba917bd4cb4b4d43416e4e87443d1/nasa.csv)

## Project files

| File | Purpose |
| --- | --- |
| `data/historical.csv` | Ready-to-train sample |
| `dataset.py` | Feature names, data validation, and removal of repeated asteroid IDs |
| `fetch_data.py` | Optional NASA NeoWs downloader |
| `train.py` | Training, threshold selection, baseline comparison, and model saving |
| `predict.py` | Predict one catalog flag using measurements you enter |
| `requirements.txt` | Tested package versions |

## Viva questions

**Why can accuracy be misleading here?** Only 115 of the 739 test objects had a positive flag. Calling every object unflagged would give about 84% accuracy while missing all 115.

**Why keep a separate validation set?** Choosing the cutoff is part of building the model. Validation lets us choose it while reserving the test asteroids for the final evaluation.

**Why can a bright asteroid produce a false alarm?** Brightness supplies only part of the information used by the catalog. The model sees one encounter and has no direct measurement of MOID.
