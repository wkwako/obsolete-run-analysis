import numpy as np
import pandas as pd
import requests
import os
import json
import datetime

#from datetime import date, timedelta

class RetentionAnalysis():
    def __init__(self):
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
        r = requests.get(f"https://www.speedrun.com/api/v1/users/{user}")
        data = r.json()["data"]
        name_international = data["names"]["international"]
        user_id = data["id"]
        return user_id, name_international

    def get_game_id(self,  game):
        #check the dictionary first
        if game in self.game_to_id:
            return self.game_to_id[game]

        r = requests.get("https://www.speedrun.com/api/v1/games", params={"name": game}, headers=self.headers)
        game_id = r.json()["data"][0]["id"]

        self.game_to_id[game] = game_id
        self.id_to_game[game_id] = game
        with open(self.game_ids_path, "w") as file:
            json.dump(self.game_to_id, file)

        return game_id

    def get_categories(self, game_id):
        r = requests.get(f"https://www.speedrun.com/api/v1/games/{game_id}/categories")

        categories = r.json()["data"]

        return categories

    def get_runs(self, game):
        game_id = self.get_game_id(game)

        os.makedirs("run_data", exist_ok=True)

        # build a filepath for this game's cached runs
        runs_path = os.path.join("run_data", f"runs_{game_id}.json")

        # if we've already fetched this game, load from disk and skip the API entirely
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
        #put into form: {game: {user1: {category1: [run1, ...], ...}, ...}, ...}
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

            #only run doesn't exist
            else:
                cleaned_runs[game_id][user_id][category_id].append(run)

        return cleaned_runs

    def recency_runs(self, game_id, user_id, cutoff_datetime):
        """Expects a cutoff_datetime converted using datetime.datetime()"""

        runs = self.load_and_format_runs(game_id, user_id, cutoff_datetime)

        first_run = runs[0]
        last_run = runs[1]

        return first_run, last_run

    def load_and_format_runs(self, game_id, user_id, cutoff_datetime):
        """Pulls runs with game_id and user_id before or equal to cutoff_datetime."""
        #identify correct file
        path = os.path.join("run_data", f"runs_{game_id}.json")

        if not os.path.isfile(path):
            print ("Game data does not exist")
            return

        #get all runs associated with this user from the file
        with open(path) as file:
            data = json.load(file)

        for category in data[game_id][user_id].keys():
            runs.extend(data[game_id][user_id][category])

        runs = [run for run in runs if run["date"] <= cutoff_datetime]

        runs.sort(key=lambda run: run["date"])

        return runs

    def feature_recency(self, game_id, user_id, cutoff_date):
        """Expects a string cutoff_date in the form YYYY-MM-DD"""
        cutoff_datetime = datetime.datetime.fromisoformat(cutoff_date)

        first_run, last_run = self.recency_runs(game_id, user_id, cutoff_datetime)

        if first_run:
            first_run_dt = datetime.datetime.fromisoformat(first_run["date"])
            first_run = (cutoff_datetime - first_run_dt).days

        if last_run:
            last_run_dt = datetime.datetime.fromisoformat(last_run["date"])
            last_run = (cutoff_datetime - last_run_dt).days

        return first_run, last_run

    def count_runs(self, game_id):
        count_runs = {}
        
        #get game path
        runs_path = os.path.join("run_data", f"runs_{game_id}.json")

        #if the file doesn't exist, exit
        if not os.path.isfile(runs_path):
            print ("Game data does not exist")
            return

        with open(runs_path) as file:
            data = json.load(file)

        for game, user_dict in data.items():
            for user, category_dict in user_dict.items():
                total = 0
                for category, runs_list in category_dict.items():
                    total += len(runs_list)
                count_runs[user] = total


        buckets = {"1": 0, "2-5": 0, "6-10": 0, "11+": 0}
        for user, total in count_runs.items():
            if total == 1:
                buckets["1"] += 1
            elif total <= 5:
                buckets["2-5"] += 1
            elif total <= 10:
                buckets["6-10"] += 1
            else:
                buckets["11+"] += 1

        print(buckets)

    def retention_diagnostic(self, game_id, cutoff_str, min_prior_runs=3, window_months=12):
        """
        Measures class balance for the retention label among qualifying runners.

        game_id:        the game to analyze (must already be cached)
        cutoff_str:     cutoff date T as "YYYY-MM-DD"
        min_prior_runs: minimum runs BEFORE T for a runner to qualify (need history for features)
        window_months:  how many months after T defines the retention window
        """
        # --- load cached data ---
        runs_path = os.path.join("run_data", f"runs_{game_id}.json")
        if not os.path.isfile(runs_path):
            print("Game data does not exist. Exiting...")
            return
        with open(runs_path) as file:
            data = json.load(file)

        # --- parse the cutoff and window end ---
        cutoff = datetime.fromisoformat(cutoff_str)
        window_end = cutoff + datetime.timedelta(days=window_months * 30)  # approx months

        # --- counters ---
        total_runners = 0
        qualifying = 0          # runners with >= min_prior_runs before cutoff
        retained = 0            # qualifying runners active in [cutoff, window_end]
        churned = 0             # qualifying runners NOT active in that window
        skipped_no_dates = 0    # runners we couldn't place in time at all

        # data is {game: {user: {category: [runs]}}}
        for game_key, user_dict in data.items():
            for user, category_dict in user_dict.items():
                total_runners += 1

                # flatten this user's run dates across ALL categories
                run_dates = []
                for category, runs_list in category_dict.items():
                    for run in runs_list:
                        d = run.get("date")
                        if d:  # skip null dates
                            try:
                                run_dates.append(datetime.fromisoformat(d))
                            except ValueError:
                                pass  # skip malformed dates

                if not run_dates:
                    skipped_no_dates += 1
                    continue

                # count runs strictly before the cutoff
                prior_runs = [d for d in run_dates if d < cutoff]
                if len(prior_runs) < min_prior_runs:
                    continue  # not enough history to qualify

                qualifying += 1

                # label: did they submit anything in [cutoff, window_end]?
                active_in_window = any(cutoff <= d <= window_end for d in run_dates)
                if active_in_window:
                    retained += 1
                else:
                    churned += 1

        # --- report ---
        print(f"\n{'='*55}")
        print(f"RETENTION DIAGNOSTIC — game {game_id}")
        print(f"Cutoff T = {cutoff_str}, window = {window_months} months, min prior runs = {min_prior_runs}")
        print(f"{'='*55}")
        print(f"Total runners in game:        {total_runners}")
        print(f"Skipped (no usable dates):    {skipped_no_dates}")
        print(f"Qualifying runners (>= {min_prior_runs} prior): {qualifying}")
        if qualifying == 0:
            print("No qualifying runners — try an earlier cutoff or lower min_prior_runs.")
            return
        ret_pct = 100 * retained / qualifying
        chu_pct = 100 * churned / qualifying
        print(f"  Retained: {retained} ({ret_pct:.1f}%)")
        print(f"  Churned:  {churned} ({chu_pct:.1f}%)")
        print(f"{'='*55}\n")

        return {"qualifying": qualifying, "retained": retained, "churned": churned}

ret = RetentionAnalysis()
game = "Hollow Knight"
#ret.get_runs(game)

game_id = ret.get_game_id(game)
user_id = "o86w5pwx"
cutoff_date = "2023-04-05"

first_date, last_date = ret.feature_recency(game_id, user_id, cutoff_date)
print (first_date, last_date)

#print (ret.get_user("Lep"))

#ret.retention_diagnostic(game_id, "2022-01-01", min_prior_runs=5, window_months=12)