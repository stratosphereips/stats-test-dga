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

## Why run the experiments this way?
Do I need to run each experiment 30 times? Let's say you want to know the accuracy of your DGA detection model. You can't test your model against all the DGAs and known domains. So you use some few DGA and non-DGA domains and check if your model discriminate each domain correctly. You calculate the accuracy ((TP + TN)/(TP + TN + FP + FN)) and that's it, right? The problem with that is you didn't use the total population but a sample of it. Thus, the accuracy of your model will have a probability distribution you don't know, with a mean (μ) and standard deviation (σ) that you also don't know. Thus the accuracy you calculated can be near the real μ or very far off if the σ is big.

But according to the Central Limit Theorem, if you repeatedly measure the value of an independent variable (the accuracy, in our case), the average will converge to the real μ and the standard deviation will converge to the real σ, the more times you measure. As a rule of thumb, if you repeat the experiments 30 times, the values should be close enough.

