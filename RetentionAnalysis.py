import numpy as np
import pandas as pd
import requests
import os
import json
import datetime
import dateutil
import sqlite3

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

from sklearn.base import clone

import diagnostics
import retention_mlp

class Repository():
    def save_to_db(self, df, game_id):
        """Given a df with feature and label data, and a game_id, saves
            it to the database."""

        #connect to the db
        conn = sqlite3.connect("retention_examples.db")
        cur = conn.cursor()

        # does the table exist yet? (first-ever run, it won't)
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='retention_examples'"
        )
        table_exists = cur.fetchone() is not None

        # if it exists, clear out any prior rows for THIS game
        if table_exists:
            cur.execute("DELETE FROM retention_examples WHERE game_id = ?", (game_id,))
            conn.commit()

        # append the fresh rows for this game
        df.to_sql("retention_examples", conn, if_exists="append", index=False)

        #close the connection
        conn.close()

    def load_from_db(self, game_id=None):
        """Given a game_id, returns the df from the db for that game.
            If no game is specified, loads the df for all games."""
        conn = sqlite3.connect("retention_examples.db")
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='retention_examples'")
        if cur.fetchone() is None:
            conn.close()
            return pd.DataFrame()          # no table yet → empty frame
        if game_id is None:
            df = pd.read_sql("SELECT * FROM retention_examples", conn)
        else:
            df = pd.read_sql("SELECT * FROM retention_examples WHERE game_id = ?", conn, params=(game_id,))
        conn.close()
        return df

class RetentionAnalysis():
    def __init__(self):
        self.repo = Repository()

        #defining a header as requested in https://github.com/speedruncomorg/api
        self.headers = {"User-Agent": """Project title: SR Retention Analysis. Description: I am writing an agent to predict retention rates across players and games (given metadata and a player's history, if they'll still be submitting runs after some period of time has passed). This analysis uses tools like sklearn and PyTorch, and requires pulling all runs for several games."""}
        self.game_ids_path = "game_ids.txt"

        #retrieve game id info if it exists
        if os.path.isfile(self.game_ids_path):
            with open (self.game_ids_path, "r") as file:
                self.game_to_id = json.load(file)
                self.id_to_game = {value:key for key,value in self.game_to_id.items()}

        #otherwise, create new dicts for this info
        else:
            self.game_to_id = {}
            self.id_to_game = {}

        
        self.user_data = {}

    def get_user(self, user):
        """Takes either a user or user_id, and returns (user_id, user_name)"""
        r = requests.get(f"https://www.speedrun.com/api/v1/users/{user}")
        data = r.json()["data"]
        name_international = data["names"]["international"]
        user_id = data["id"]
        return user_id, name_international

    def get_game_id(self,  game):
        """Given a game name (string), returns the game_id."""
        #check the dictionary first
        if game in self.game_to_id:
            return self.game_to_id[game]

        r = requests.get("https://www.speedrun.com/api/v1/games", params={"name": game}, headers=self.headers)
        game_id = r.json()["data"][0]["id"]

        #store game in dict, then write to file
        self.game_to_id[game] = game_id
        self.id_to_game[game_id] = game
        with open(self.game_ids_path, "w") as file:
            json.dump(self.game_to_id, file)

        return game_id

    def get_categories(self, game_id):
        """Given a game_id, returns a list of categories"""
        r = requests.get(f"https://www.speedrun.com/api/v1/games/{game_id}/categories")

        categories = r.json()["data"]

        return categories

    def get_runs(self, game):
        game_id = self.get_game_id(game)

        os.makedirs("run_data", exist_ok=True)

        # build a filepath for this game's cached runs
        runs_path = os.path.join("run_data", f"runs_{game_id}.json")

        # if we've already fetched this game, load from disk and skip the API
        if os.path.isfile(runs_path):
            with open(runs_path, "r") as file:
                self.user_data = json.load(file)
            print(f"Loaded cached runs for \"{game}\" from disk")
            return self.user_data

        # otherwise, fetch fresh from the API
        runs = []
        m = 200

        categories = self.get_categories(game_id)

        for category in categories:
            category_id = category["id"]
            offset = 0
            while True:
                params = {
                    "game": game_id,
                    "category": category_id,
                    "orderby": "date",
                    "direction": "asc",
                    "max": m,
                    "offset": offset
                }

                r = requests.get("https://www.speedrun.com/api/v1/runs", params=params)

                print("offset:", offset, "status:", r.status_code)

                data = r.json()["data"]
                if not data:
                    break

                runs.extend(data)

                offset += m

        # clean the runs into the nested structure
        runs = self.clean_runs(runs)

        # write to disk so future runs load from cache
        with open(runs_path, "w") as file:
            json.dump(runs, file)
        print(f"Fetched and cached runs for \"{game}\"")

        self.user_data = runs

        return self.user_data

    def clean_runs(self, runs):
        """Given raw runs info from the API, returns a dictionary in the form {game_id: {user_id1: {category1, ...}, ...}}"""
        cleaned_runs = {}
        for i,run in enumerate(runs):
            #run is rejected, don't log it
            if run["status"]["status"] == "rejected":
                continue

            #if game has multiple players, don't log it
            if len(run["players"]) != 1:
                continue

            #run is by a guest, don't log it
            if run["players"][0]["rel"] == "guest":
                continue

            #if the date doesn't exist, skip it
            if not run["date"]:
                continue

            user_id = run["players"][0]["id"]
            game_id = run["game"]
            category_id = run["category"]

            #game doesn't exist, add it
            if game_id not in cleaned_runs:
                cleaned_runs[game_id] = {user_id: {category_id: [run]}}

            #game exists but user doesn't
            elif user_id not in cleaned_runs[game_id]:
                cleaned_runs[game_id][user_id] = {category_id: [run]}

            #game and user exist, category doesn't
            elif category_id not in cleaned_runs[game_id][user_id]:
                cleaned_runs[game_id][user_id][category_id] = [run]

            #run doesn't exist
            else:
                cleaned_runs[game_id][user_id][category_id].append(run)

        return cleaned_runs

    def build_db_entry(self, game, game_id, lookahead_window, min_runs, freq_months):
        """Builds the df for a game and saves it to the db."""

        #makes sure the run_data exists before we continue
        data = self.get_runs(game)

        df = self.repo.load_from_db(game_id)

        if not df.empty:
            print ("Game data already exists.")
            return df

        self.get_runs(game)
        cutoffs = self.generate_cutoffs(game_id, lookahead_window)
        table = self.generate_table(game_id, cutoffs, lookahead_window, min_runs, freq_months)
        self.repo.save_to_db(table, game_id)
        df = self.repo.load_from_db(game_id)
        return df

    def recency_runs(self, runs):
        """Given a sorted list of runs, returns the first and last runs in the list."""

        if not runs:
            return None, None

        first_run = runs[0]
        last_run = runs[-1]

        return first_run, last_run

    def load_run_data(self, game_id):
        """Given a game_id, loads the loads and outputs the run dictionary."""
        runs_path = os.path.join("run_data", f"runs_{game_id}.json")
        if not os.path.isfile(runs_path):
            print("Game data does not exist. Exiting...")
            return None
        with open(runs_path) as file:
            data = json.load(file)

        return data

    def load_runs(self, game_id, user_id):
        """Given a game_id and a user_id, loads the run dictionary from file and outputs an unsorted list of runs."""

        data = self.load_run_data(game_id)

        #get all runs for this game and user, put them into a single list
        runs = []
        for category in data[game_id][user_id].keys():
            runs.extend(data[game_id][user_id][category])

        return runs
 
    def sort_and_filter_runs(self, runs, cutoff_datetime):
        """Pulls runs with game_id and user_id before or equal to cutoff_datetime. Sorts by date, ascending.
           Use after load_runs()."""

        runs = [run for run in runs if datetime.datetime.fromisoformat(run["date"]) < cutoff_datetime]

        runs.sort(key=lambda run: run["date"])

        return runs

    def get_label(self, runs):
        """Given a list of sorted runs, returns 1 if data exists, and 0 otherwise."""
        if runs:
            return 1

        return 0

    def features(self, cutoff_datetime, months, runs):
        """Given a list of sorted and filtered runs and parameters for feature calculation,
           calculations all features and returns a tuple with their results."""

        first_run_days, last_run_days = self.feature_recency(runs, cutoff_datetime)

        lifetime_freq, windowed_freq = self.feature_frequency(runs, cutoff_datetime, months)

        density = self.feature_density(lifetime_freq, first_run_days, last_run_days)

        num_cats = self.feature_engagement_depth(runs)

        #cv = self.feature_consistency(runs)

        return (first_run_days, last_run_days, lifetime_freq, windowed_freq, density, num_cats)

    def feature_recency(self, runs, cutoff_datetime):
        """Given a list of runs and a datetime cutoff, returns the number of days
           from the user's most recent run (last_run_days) and their first run
           (first_run_days)."""

        first_run, last_run = self.recency_runs(runs)
        first_run_days, last_run_days = None, None

        #if first_run exists, get day difference
        if first_run:
            first_run_dt = datetime.datetime.fromisoformat(first_run["date"])
            first_run_days = (cutoff_datetime - first_run_dt).days            

        #if last_run exists, get day difference
        if last_run:
            last_run_dt = datetime.datetime.fromisoformat(last_run["date"])
            last_run_days = (cutoff_datetime - last_run_dt).days

        return first_run_days, last_run_days

    def feature_frequency(self, runs, cutoff_datetime, months):
        """Given a list of runs, a datetime cutoff, and months to look back, 
           calculates the number of lifetime runs and number of runs within
           the lookback period."""

        #calculate total runs before T (lifetime frequency)
        lifetime_freq = len(runs)

        #filter run by start date
        start_window = cutoff_datetime - dateutil.relativedelta.relativedelta(months=months)
        runs = [run for run in runs if datetime.datetime.fromisoformat(run["date"]) > start_window]

        #calculate runs in the last x months (windowed frequency)
        windowed_freq = len(runs)

        return lifetime_freq, windowed_freq

    def feature_density(self, lifetime_freq, first_run_days, last_run_days):
        """Given lifetime runs, and days since the user's first and last runs,
           calculates num_runs/active_span, where num_rums is lifetime freq and
           active_span is (last_run_days - first_run_days)."""
        
        active_span = first_run_days - last_run_days
        if active_span == 0:
            active_span = 1
        density = lifetime_freq/active_span
        return density

    def feature_engagement_depth(self, runs):
        """Given a list of sorted runs, returns the number of unique
           categories present in the list. This is how many categories
           a user submitted runs to within the date range."""

        cats = set([run["category"] for run in runs])

        num_cats = len(cats)

        return num_cats

    def feature_consistency(self, runs):
        """Given a list of sorted runs, returns a value that represents
           how consistently the user submitted runs."""
        days = [datetime.datetime.fromisoformat(run["date"]).toordinal() for run in runs]
        gaps = np.diff(days)

        if len(gaps) == 0 or gaps.mean() == 0:
            return None

        cv = gaps.std() / gaps.mean()

        return cv

    def generate_cutoffs(self, game_id, lookahead_window, diff=6):
        """Given a game_id, a lookahead window, and a distance between dates (diff),
           returns a list of datetimes we can use as cutoff dates for modeling."""

        data = self.load_run_data(game_id)
        game = data[game_id]

        #get all dates for all runs
        all_dates = [run["date"] for user_dict in game.values() for cats in user_dict.values() for run in cats]

        if not all_dates:
            return []

        #get the date of the earliest run. dates are in YYYY-MM-DD format without conversion
        start_year = int(min(all_dates)[:4])

        #get the date of the latest run
        last_date = datetime.datetime.fromisoformat(max(all_dates))

        #get the final window, which must be lookahead_window months before last_date
        ceiling = last_date - dateutil.relativedelta.relativedelta(months=lookahead_window)

        #generate a list of dates using diff as the spacing between dates
        delta = dateutil.relativedelta.relativedelta(months=diff)
        cur_date = datetime.datetime(start_year, 1, 1)
        valid_datetimes = []
        while cur_date <= ceiling:
            valid_datetimes.append(cur_date)
            cur_date += delta

        return valid_datetimes

    def generate_table(self, game_id, cutoffs, lookahead_window, min_runs, freq_months):
        """Creates a dataframe of features of labels for the given cutoff date
           and lookahead window."""

        data = self.load_run_data(game_id)

        rows = []

        for user_id in data[game_id].keys():

            #get a list of all runs for a user
            runs = [run for cats in data[game_id][user_id].values() for run in cats]

            #if fewer runs than min_runs, go to next user
            if len(runs) < min_runs:
                continue

            for cutoff in cutoffs:

                #get runs before each cutoff and sort them
                before = [run for run in runs if datetime.datetime.fromisoformat(run["date"]) < cutoff]
                before.sort(key=lambda run: run["date"])

                #if fewer runs than min_runs, go to next cutoff
                if len(before) < min_runs:
                    continue

                #get the date at the end of the lookahead window and generate runs between cutoff and the end of the window
                delta = dateutil.relativedelta.relativedelta(months=lookahead_window)
                after = [run for run in runs if cutoff <= datetime.datetime.fromisoformat(run["date"]) <= (cutoff+delta)]
                after.sort(key=lambda run: run["date"])

                #generate features for the "before" runs
                features = self.features(cutoff, freq_months, before)

                #generate a label for the "after" runs
                label = self.get_label(after)

                rows.append((game_id, user_id, cutoff, *features, label))

        #define columns
        columns = ["game_id", "user_id", "cutoff", "first_run_days", "last_run_days", "lifetime_freq", "windowed_freq", "density", "num_cats", "label"]
        df = pd.DataFrame(rows, columns=columns)

        #save to the db
        self.repo.save_to_db(df, game_id)

        return df

    # def split(self, game, game_id, p_break, lookahead_window, min_runs, freq_months):
        
    #     #load only for a single game
    #     #df = self.load_from_db(game_id)

    #     df = self.get_or_build(game, game_id, lookahead_window, min_runs, freq_months)

    #     counts_sorted, total = self.cutoff_diagnostic(df)

    #     B = self.build_B(counts_sorted, total, p_break)

    #     #get rows where cutoff < B (train_set)
    #     train = df[df["cutoff"] < B]

    #     #get rows where cutoff >= B (test_set)
    #     test = df[df["cutoff"] >= B]

    #     #if user_id appears in both, move rows with that user_id from test_set to train_set
    #     both = set(train["user_id"]) & set(test["user_id"])
    #     mask = test["user_id"].isin(both)
    #     to_move = test[mask]
    #     train = pd.concat([train, to_move])
    #     test = test[~mask]

    #     #print (train["label"].mean())
    #     #print (test["label"].mean())

    #     return (train, test)

    def split_one_game(self, df, p_break):
        """Temporal + grouped split for a single game's dataframe.
        Returns (train, test) with straddlers moved to train."""

        #get sorted run counts and total runs
        counts_sorted, total = self.cutoff_counts(df)

        #determine at which date to split train/test data
        B = self.build_B(counts_sorted, total, p_break)

        #split the data into train/test
        train = df[df["cutoff"] < B]
        test  = df[df["cutoff"] >= B]

        # move within-game straddlers to train

        #get users that are in both train and test sets
        both = set(train["user_id"]) & set(test["user_id"])

        #get the rows in test corresponding to these users
        to_move = test[test["user_id"].isin(both)]

        #create a new train df with itself and what we're moving from test
        train = pd.concat([train, to_move])

        #create a new test df that doesn't have rows with users appearing in both sets
        test  = test[~test["user_id"].isin(both)]

        return train, test

    def split(self, game_id, p_break, all_games=False):

        #load just one game's data
        if not all_games:
            df = self.repo.load_from_db(game_id)
            train, test = self.split_one_game(df, p_break)

        #load all game data
        else:
            df_all = self.repo.load_from_db()
            trains, tests = [], []
            for gid in df_all["game_id"].unique():
                game_df = df_all[df_all["game_id"] == gid]
                g_train, g_test = self.split_one_game(game_df, p_break)
                trains.append(g_train)
                tests.append(g_test)

            train = pd.concat(trains)
            test  = pd.concat(tests)

            # --- cross-game straddler fix ---
            # a runner may be train in one game but test in another; force them all to train
            both = set(train["user_id"]) & set(test["user_id"])
            to_move = test[test["user_id"].isin(both)]
            train = pd.concat([train, to_move])
            test  = test[~test["user_id"].isin(both)]

        # reset indices after all the concatenation
        train = train.reset_index(drop=True)
        test  = test.reset_index(drop=True)

        # guardrail: no runner in both sets, ever
        assert set(train["user_id"]).isdisjoint(set(test["user_id"])), "runner in both train and test!"

        # sanity check
        print(f"train: {len(train)} rows, {train['label'].mean():.3f} positive")
        print(f"test:  {len(test)} rows, {test['label'].mean():.3f} positive")

        return train, test

    def prep_train_test(self, train, test):
        """Given train and test data, splits the data into X train, y train,
           X test, and y test sets."""
        feature_columns = ["first_run_days", "last_run_days", "lifetime_freq", "windowed_freq", "density", "num_cats"]
        X_train, y_train = train[feature_columns], train["label"]
        X_test, y_test = test[feature_columns], test["label"]

        return (X_train, y_train, X_test, y_test)

    # def get_or_build(self, game, game_id, lookahead_window, min_runs, freq_months):
    #     df = self.load_from_db(game_id)

    #     #not in db, fetch the information
    #     if df.empty:
    #         self.get_runs(game)
    #         cutoffs = self.generate_cutoffs(game_id, lookahead_window)
    #         table = self.generate_table(game_id, cutoffs, lookahead_window, min_runs, freq_months)
    #         self.save_to_db(table, game_id)
    #         df = self.load_from_db(game_id)
    #     return df

    def get_class_weights(self, df):
        "Used for debugging. Given a df, returns the percentage of positive examples."
        count = 0
        for item in df:
            val = item[-1]
            if val == 1:
                count += 1

        print (f"{count/len(df)} percent positive examples")

    def cutoff_counts(self, df):
        """Given the df of examples, returns the number of examples in each cutoff,
           and the number of total examples. Used to determine how to split the data
           for the train/test sets."""
        counts = {}
        total = 0
        for date in df["cutoff"]:
            counts[date] = counts.get(date, 0) + 1
            total += 1

        counts_sorted = dict(sorted(counts.items()))
        return (counts_sorted, total)

    def build_B(self, counts_sorted, total, p_break):
        """Determines how to split train and test sets. Splits so that
           p_break percent of the data falls into train and (1-p_break)
           into test. Returns the datetime to split at such that
           cutoff < B is train and cutoff >= B is test."""
        n_so_far = 0
        B = None

        for key, val in counts_sorted.items():
            n_so_far += val
            if (n_so_far/total) >= p_break:
                B = key
                break

        return B

    def cross_validate(self, game_id):
        df = self.repo.load_from_db(game_id)
        feature_cols = ["first_run_days", "last_run_days", "lifetime_freq", "windowed_freq", "density", "num_cats"]
        X = df[feature_cols]
        y = df["label"]
        groups = df["user_id"]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ])

        gfk = GroupKFold(n_splits=5)

        scores = cross_val_score(pipe, X, y, groups=groups, cv=gfk, scoring="roc_auc")

        print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    def AUC(self, X_train, y_train, X_test, y_test, model=None, scale=True):
        """Given train and test sets, trains a model and returns the AUC.
           Returns the recency AUC and the full AUC."""
        if model is None:
            model = LogisticRegression()

        #we compare the full feature set to just recency features. this is our baseline
        recency_cols = ["first_run_days", "last_run_days"]

        def run(train_cols):
            Xtr = X_train[train_cols]
            Xte = X_test[train_cols]

            if scale:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

            m = clone(model)          # fresh, unfitted copy each call
            m.fit(Xtr, y_train)
            probs = m.predict_proba(Xte)[:, 1]
            return roc_auc_score(y_test, probs)

        #calculate recency and full auc
        auc_recency = run(recency_cols)
        auc_full = run(list(X_train.columns))

        print(f"recency-only AUC: {auc_recency:.4f}")
        print(f"full AUC:         {auc_full:.4f}")
        return auc_recency, auc_full

#ret = RetentionAnalysis()
#game = "Hollow Knight"
#ret.get_runs(game)

#game_id = ret.get_game_id(game)
#print (game_id)
#user = "Lep"
#user_id, user = ret.get_user(user)
#cutoff_date = "2023-04-05"
#lookahead_window = 12

ret = RetentionAnalysis()
repo = Repository()
game = "Destiny 2"
game_id = ret.get_game_id(game)
ret.build_db_entry(game, game_id, lookahead_window=12, min_runs=5, freq_months=3)
train, test = ret.split(game_id, p_break=0.50, all_games=False)
X_train, y_train, X_test, y_test = ret.prep_train_test(train, test)

print ("----SKLEARN MANUAL LINEAR-REGRESSION----")
ret.AUC(X_train, y_train, X_test, y_test, LogisticRegression(), scale=True)

print ("----SKLEARN MANUAL GRADIENT BOOSTING----")
ret.AUC(X_train, y_train, X_test, y_test, GradientBoostingClassifier(random_state=1), scale=False)

print ("----CROSS VALIDATION----")
print (ret.cross_validate(game_id))

print ("----NEURAL NET----")
print ("NN Recency: ")
retention_mlp.train_mlp(X_train, y_train, X_test, y_test, feature_cols=["first_run_days", "last_run_days"])
print ("NN All Features")
retention_mlp.train_mlp(X_train, y_train, X_test, y_test)