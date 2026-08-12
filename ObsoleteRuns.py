import numpy as np
import pandas as pd
import srcomapi as sr
import srcomapi.datatypes as dt
import requests
import os
import json

class ObsoleteRuns():
    def __init__(self):
        #defining a header as requested in https://github.com/speedruncomorg/api
        self.headers = {"User-Agent": """Project title: Obsolete Run Analysis. Description: I am writing an agent to predict when and by how much obsolete runs will be broken, using tools like sklearn and PyTorch. This requires pulling all runs (including obsolete runs) for several games."""}
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

    def get_runs(self, game, user_id=None):
        game_id = self.get_game_id(game)

        runs = []
        offset = 0
        m = 200
        while True:
            params = {
                "user": user_id,
                "game": game_id,
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

        #clean the runs
        runs = self.clean_runs(runs)
        self.user_data = runs

        return self.user_data

    def clean_runs(self, runs):
        #put into form: {user1: {game1: {category1: [run1, ...], ...}, ...}, ...}
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

            user_id = run["players"][0]["id"]
            game_id = run["game"]
            category_id = run["category"]

            #user doesn't exist, add them
            if user_id not in cleaned_runs:
                cleaned_runs[user_id] = {game_id: {category_id: [run]}}

            #user exists but game doesn't
            elif game_id not in cleaned_runs[user_id]:
                cleaned_runs[user_id][game_id] = {category_id: [run]}

            #user and game exist, category doesn't
            elif category_id not in cleaned_runs[user_id][game_id]:
                cleaned_runs[user_id][game_id][category_id] = [run]

            #only run doesn't exist
            else:
                cleaned_runs[user_id][game_id][category_id].append(run)

        return cleaned_runs


obsruns = ObsoleteRuns()
obsruns.get_game_id("Super Mario Bros")
obsruns.get_runs("super mario bros")
