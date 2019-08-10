#!/usr/bin/python3
import requests
import API_KEY
from pprint import pprint

API_KEY = API_KEY.api_key
DEFAULT_PATH = "https://api.yelp.com/v3/businesses/search"
ZIP_FILE = "../datas/zip_code.txt"
FM_DETAIL="farmers_market_2018.json"
FM_ALIAS="farmers_market_2019_biz"
QUERY_TERM = "farmers market"


def _bizid_request(query):
    """
    Returns HTTP response from yelp fusion API with given query
    """
    headers = {"Authorization": "Bearer {}".format(API_KEY)}
    response = requests.request("GET", DEFAULT_PATH, headers=headers, params=query)
    return response.json()

def _check_state(biz):
    """
    Returns True iff given business is located in Illinois, False o.w.
    """
    return biz['location']['state'] == 'IL'

def _check_open(biz):
    """
    Returns True iff given business is open, False o.w.
    """
    return biz['is_closed'] == False

def _add_business(resp, s, name):
    """
    Parses HTTP response and appends all releavent business and its
    alias to 's' and 'name'
    """
    if 'businesses' in resp:
        for biz in resp['businesses']:
            if _check_state(biz) and _check_open(biz):
                if biz['alias'] not in name:
                    s['businesses'].append(biz)
                    name.add(biz['alias'])

def get_all_bizid(file):
    """
    Iterates though a given file and searches for all open businesses located
    in Illinois
    """
    res = {"businesses":[]}
    name = set()
    with open(file) as f:
        for line in f:
            query={"term":QUERY_TERM, "location":line.rstrip()}
            resp = _bizid_request(query)
            _add_business(resp, res, name)

    return res, name

def main():
    """
    Reads zipcode file; searches and returns all the releavent businesses
    """
    res, name = get_all_bizid(FILE)
    with open(FM_DETAIL, 'w') as f:
        import json
        f.write(json.dumps(res))
    with open(FM_ALIAS, 'w') as f:
        f.write('\n'.join(n for n in name))

if __name__ == '__main__':
    main()
