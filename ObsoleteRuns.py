import numpy as np
import pandas as pd
import srcomapi as sr
import srcomapi.datatypes as dt
import requests

class ObsoleteRuns():
    def __init__(self):
        self.game_to_id = {}
        self.id_to_game = {}

    def get_game_id(self,  name):
        #check the dictionary first
        if name in self.game_to_id:
            print (f"Retrieving game \"{name}\", already stored as id \"{game_id}\"")
            return self.game_to_id["name"]

        r = requests.get("https://www.speedrun.com/api/v1/games", params={"name": name})
        game_id = r.json()["data"][0]["id"]

        self.game_to_id[name] = game_id
        self.id_to_game[game_id] = name

        print (f"Stored game \"{name}\" as id \"{game_id}\"")

        return game_id

obsruns = ObsoleteRuns()
obsruns.get_game_id("Super Mario Bros")

r = requests.get("https://www.speedrun.com/api/v1/games?name=super%20mario%20world")
r.json()["data"][0]["id"]

r = requests.get("https://www.speedrun.com/api/v1/leaderboards/smw/category/96_Exit")
print (r.json())

r.json()["data"]["game"]

# api = sr.SpeedrunCom()
# game = api.search(sr.datatypes.Game, {"name": "super mario sunshine"})[0]

# game.categories

# sms_runs = {}
# for category in game.categories:
#   if not category.name in sms_runs:
#     sms_runs[category.name] = {}
#   if category.type == 'per-level':
#     for level in game.levels:
#       sms_runs[category.name][level.name] = dt.Leaderboard(api, data=api.get("leaderboards/{}/level/{}/{}?embed=variables".format(game.id, level.id, category.id)))
#   else:
#     sms_runs[category.name] = dt.Leaderboard(api, data=api.get("leaderboards/{}/category/{}?embed=variables".format(game.id, category.id)))

# sms_runs["Any%"].runs[0]["run"]

# print (game.categories[0].type)