#!/usr/bin/env python3
"""
script that prints the location of a specific user

"""
import requests
import sys
import time

if __name__ == '__main__': 
    #https://www.dataquest.io/blog/python-api-tutorial/
 url = sys.argv[1]
 if url != '':
    try:
        req = requests.get(url)
        code = req.status_code
        if code == 403:
            limit = req.headers['X-Ratelimit-Reset']
            limit = int((int(limit) - int(time.time())) / 60)
            print("Reset in {} min".format(limit))
        elif code == 200:
            try:
                location = req.json()['location']
                print(location)
            except KeyError:
                print("Not found")    
        else:
            print("Not found")               
    except ValueError:
        print("Please provide args")
          
