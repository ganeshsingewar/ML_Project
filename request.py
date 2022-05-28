# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:08:31 2022

@author: Ganesh Singewar
"""


import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

print(r.json())