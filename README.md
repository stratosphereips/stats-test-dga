# Performance Experiments for DGA Detectors
## What this program does?
Execute a series of repeated experiments to determine, statistically, the performance of your DGA detector. The `scripts/run_test_dga.py` take a random sample of DGA domains and mix it with legit domains extracted from the 500 000 most popular domains in the Tranco list for the date 2023-07-01. Then ask the `inference()` function for predictions of the labels **dga** and **nondga**. The DGA domains can be sampled from one or all the families contained in the UMUDGA dataset.

## How to run the experiments
Clone the repo, enter the scripts directory and run using python. You will need Pandas and Numpy libraries you can obtain with `pip install pandas numpy`.

```
  git clone https://github.com/jwackito/stats-test-dga.git testdga
  cd testdga/scripts
  python run_test_dga.py
```
The script creates a folder scripts/results/ with the results per set of experiments.
