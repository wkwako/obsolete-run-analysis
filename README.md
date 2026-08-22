# Description

The goal of this project is to use data from speedrun.com's API to predict speedrunner retention rates across games, using known ML techniques with a novel domain. The retention rate is defined as how likely a runner is to continue submitting runs, given a particular game and category. We perform the analysis with several models: sklearn's logistic regression and gradient boosting models, a neural net with one hidden layer, and sklearn's cross-validation. Each model is a binary classifier, returning a 1 if we predict a runner will continue submitting runs, or a 0 if we predict a runner will not continue submitting runs.

# Architecture

We use the speedrun.com API to ingest data, and perform cleaning, caching, and storage in a local database before model training. Our architecture includes the following structures and steps:
1. Use the speedrun.com API to retrieve data on games, runners, and categories.
2. Clean the data by removing unusable values, storing it in dictionaries for structured access, and caching game data to text files for fast future retrieval.
3. Compute features and labels from this cleaned data, and save to a sqlite3 database.
4. Retrieve sqlite data and train/test the models.

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
* Identity leakage: Step #6 above prevents the same runner appearing in both sets, which would let the model recognize runners it will be tested on rather than learning generalizable patterns.

Splitting in this way produces varying class balances by game. Hollow Knight, for example, produces a class balance of 25.3% retained (versus churned) in train, and 41.9% retained in test. Due to this inbalance, we report AUC rather than accuracy because it returns the probability we rank a retained runner above a churned runner, avoiding accuracy-based pitfalls.

Our examined features include:

* Recency: this feature is split into two variables: last_run_days, which represents the number of days since the runner's most recent run; and first_run_days, the number of days from the runner's first recorded run (within a specified window)
* Frequency: this feature is split into two variables: lifetime_frequency, which is the total number of runs submitted by the runner before the cutoff date; and windowed_frequency, the number of runs submitted by the runner in the last several months before the cutoff date.
* Run density: the rate at which the runner submits runs, calculated as lifetime_frequency/(first_run_days - last_run_days)
* Consistency: the regularity of submission timing, calculated as the coefficient of variation (standard deviation / mean) of the gaps between consecutive runs, which distinguishes steady runners from bursty ones.
* Unique categories per runner: the number of unique categories the runner submitted runs to within the window

# Results

We calculate the AUC for three popular speedrunning games: Hollow Knight, Super Metroid, and Destiny 2. We present results for each of the four models, displaying recency-only features as separate calculations as a benchmark. The results are displayed in the following table:

| Model | Hollow Knight | Super Metroid | Destiny 2 |
|---|---|---|---|
| Logistic Regression (recency-only) | 0.7870 | 0.7974 | 0.6989 |
| Logistic Regression (full) | 0.7930 | 0.7979 | 0.7297 |
| Gradient Boosting (recency-only) | 0.7879 | 0.7954 | 0.7220 |
| Gradient Boosting (full) | 0.7919 | 0.8072 | 0.7115 |
| MLP (recency-only) | 0.7921 | 0.8009 | 0.6975 |
| MLP (full) | 0.7902 | 0.8113 | 0.6989 |
| Grouped CV (full, leaky) | 0.8814 +/- 0.0135 | 0.8783 +/- 0.0139 | 0.8564 +/- 0.0501 |

# Discussion


For every game and model except for gradient boosting in Destiny 2, the full-featured model achieves a higher AUC than the recency-only featured model. However, the difference is minor, indicating that recency dominates prediction capabilities. Gradient boosting usually outperforms logistic regression, but again, the difference is minor. The MLPs perform similarly within games, which is expected for a relatively small, non-complex dataset.

Recency is a stronger indicator on some games than others, however. Super Metroid's MLP scores 0.8113, while Destiny 2 only reaches 0.6989. This cross-game variation appears tied to the structure of each game's runner population. Super Metroid and Hollow Knight are dedicated single-player speedrunning games with committed communities of roughly 550–600 qualifying runners each, yielding thousands of training examples. Destiny 2, by contrast, is a live-service game with no full-game category. Instead, runs are individual missions, and its population is dominated by one-and-done participants: only 61 runners cleared our minimum-history threshold, producing 480 examples versus over 5,000 for the other two games, which likely explains Destiny 2's lower AUC. These estimates are also less stable, reflected in Destiny 2's higher standard deviation (±0.050 versus ±0.014 for the others). More broadly, this suggests game-level properties such as community maturity and player commitment modulate how well activity patterns predict retention.

Last, the grouped-CV AUC (~0.88) exceeds our temporal+grouped result (~0.79) by roughly 0.09. Grouped-CV catches identity but not temporal leakage, so the gap between these scores directly measures how much temporal leakage inflates AUC. This both quantifies the risk temporal leakage poses and confirms that our manual temporal split is working as intended.

# Conclusion

Recency dominates speedrunner retention prediction, and this result holds across three model families, three games, and two feature sets (recency-only and full). The ceiling on prediction is an AUC of 0.79, and the grouped-CV leakage demonstration confirms our methodology to remove identity and temporal leakage is sound.