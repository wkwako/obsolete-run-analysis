# Description

The goal of this project is to use data from speedrun.com's API to predict speedrunner retention rates across games. The retention rate is defined as how likely a runner is to continue submitting runs, given a particular game and category. We perform the analysis with several models: sklearn's linear regression and gradient boosting models, a neural net with one hidden layer, and sklearn's crossfold validation. Each model is a binary classifier, returning a 1 if we predict a runner will continue submitting runs, or a 0 if we predict a runner will not continue submitting runs.

# Method

The goal: given some date B, and features from before B, can we predict if the runner will submit a run in the 6 months following B? To do this, we generate a list of examples in the following way:

1. For each game, get the dates of the first and last runs of the game, then generate a list of cutoff dates with a spacing of 6 months between those start and end dates.
2. For each runner, for each cutoff date: generate a list of runs that occurred before that date, and a list of runs that occurred after that date.
3. Create features for the 'before' runs
4. Create labels for the 'after' runs
5. Select a date, B, which splits our training and test sets.
6. Move all runners that appear in both the training and test sets to the training set
7. Train our model, then calculate AUC

There are two kinds of leakage around which we construct our data pipeline:
* Temporal leakage: step #5 above keeps the training set before B, and the test set after B. This ensures we are not "peeking ahead" into the future, and given the model data it wouldn't have otherwise known about. If we had not performed this step, the model could glimpse into the future and perform artificially better on both sets.
* Identity leakage: Step #6 above prevents the model from memorizing runner-specific patterns, when prevents overfitting. By moving all runners that appear in both the training and test sets to only the training set, we only train the agent on runner-agnostic information.

Our examined features include:

* Recency: this feature is split into two variables: last_run_days, which represents the number of days since the runner's most recent run; and first_run_days, the number of days from the runner's first recorded run (within a specified window)
* Frequency: this feature is split into two variables: lifetime_frequency, which is the total number of runs submitted by the runner before the cutoff date; and windowed_frequency, the number of runs submitted by the runner in the last several months before the cutoff date.
* Run density: the rate at which the runner submits runs, calculated as lifetime_frequency/(first_run_days - last_run_days)
* Consistency: How often the runner submits runs, calculated by using the standard deviation and mean
* Unique categories per runner: the number of unique categories the runner submitted runs to within the window

# Results

We calculate the AUC across three games and present their results:

| Model | Hollow Knight | Super Metroid | Destiny 2 |
|---|---|---|---|
| Logistic Regression (recency-only) | 0.7870 | .07974 | 0.6989 |
| Logistic Regression (full) | 0.7930 | 0.7979 | 0.7297 |
| Gradient Boosting (recency-only) | 0.7879 | 0.7954 | 0.7220 |
| Gradient Boosting (full) | 0.7919 | 0.8072 | 0.7115 |
| MLP (recency-only) | 0.7921 | 0.8009 | 0.6975 |
| MLP (full) | 0.7902 | 0.8113 | 0.6989 |
| Grouped CV (leaky, full) | 0.8814 +/- 0.0135 | 0.8783 +/- 0.0139 | 0.8564 +/- 0.0501 |



# Discussion

# Conclusion