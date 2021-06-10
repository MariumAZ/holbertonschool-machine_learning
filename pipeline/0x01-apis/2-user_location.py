#!usr/bin/env python3

import requests
import sys
import os



if __name__ == '__main__': 
 url = sys.argv[1]
 if url != '':
    try:
        req = requests.get(url)
        code = req.status_code
        if code == 403:
            print("Reset in X min")
        if code == 200:
                req_json = req.json()
                name = req_json['login']
                if name is None:
                    print("Not Found")   
                else:   
                    location = req_json['location']
                    print(location)
    except ValueError:
        print("Please provide args")
          
