#!/usr/bin/env python3
import json
import requests
"""
collect apis
"""
def availableShips(passengerCount):
    """
    function to collect apis
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






