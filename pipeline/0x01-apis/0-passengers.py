#!/usr/bin/env python3

"""
collects apis
"""
import requests
from requests.api import request

def availableShips(passengerCount):
    """

    Returns the list of ships that can hold a given
    number of passengers using Swapi API

    """
    names = []
    url = "https://swapi-api.hbtn.io/api/starships/"
    while True:
        star_json = requests.get(url).json()
        for k, v in star_json.items():
            if k =="results":
                for d in v :
                    try:
                        if int(d["passengers"].replace(',', '')) >= passengerCount :
                            names.append(d["name"])
                    except ValueError:
                            continue
        url = requests.get(url).json()['next']
        print(url)
        if url is None:
            break
    return names                
                    






