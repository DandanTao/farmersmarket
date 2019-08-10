import requests
from pprint import pprint
import API_KEY
from get_all_business import _check_state, _check_open, _add_business
import json

API_KEY = API_KEY.api_key
DEFAULT_PATH = "https://api.yelp.com/v3/businesses/"
FM_BIZID = "../data/farmers_market_2018_bizid"

def request(id):
    headers = {"Authorization": "Bearer {}".format(API_KEY)}
    response = requests.request("GET", DEFAULT_PATH + id, headers=headers)

    return response.json()

with open(FM_BIZID) as f:
    res = {"businesses":[]}
    name = set()
    for line in f:
        line = line.strip()
        resp = request(line)
        if line not in name:
            name.add(line)
            res['businesses'].append(resp)


with open('../data/farmers_market_2018.json', 'w') as f:
    f.write(json.dumps(res, indent=4))
print(res)
