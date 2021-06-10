#!/usr/bin/env python3
"""
returns the list of names of the home planets of all sentient species.
"""
import requests

def sentientPlanets():
    """
    returns the list of names of the home planets of all sentient species.
    """
    home_names = []
    url = "https://swapi-api.hbtn.io/api/species/"
    while True:
        req_json = requests.get(url).json()
        for result in req_json['results']:
            if result['designation']=='sentient' or result['classification'] =='sentient':
                try:
                    home_url = result['homeworld']
                    home_json = requests.get(home_url).json()
                    name = home_json['name']
                    home_names.append(name)
                except ValueError:
                    continue    
        url = requests.get(url).json()['next']
        if url is None:
            break   

    return home_names            



        




     

