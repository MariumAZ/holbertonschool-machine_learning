#!/usr/bin/env python3
import json
import requests
"""
collect apis
"""
def availableShips(passengerCount):
    """

    Returns the list of ships that can hold a given
    number of passengers using Swapi API

    """
    star_req  = requests.get("https://swapi-api.hbtn.io/api/starships/")
    #print(type(star_req))
    star_json = star_req.json()
    for k, v in star_json.items():
        if k =="results":
            for d in v :
                #print(d)
                #print(d["passengers"])
                try:
                    if int(d["passengers"]) >= passengerCount :
                        print(d["name"])
                except:
                    print([])        






