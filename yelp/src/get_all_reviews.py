#!/usr/bin/python3
import requests
from pprint import pprint
from bs4 import BeautifulSoup
import json

PATH="https://www.yelp.com/biz/"
QUERY="?start="

## Please change the DATA_PATH and FILE_PATH as appropriate
DATA_PATH="../reviews_2018"
FILE_PATH="../datas/farmers_market_2018_bizid"

def _scrap_as_soup(url):
    """
    Returns a bs4 object from the given URL
    """
    req = requests.request("GET", url)
    return BeautifulSoup(req.text, "html.parser")

def _first_20_reviews(name):
    """
    Returns the first 20 reviews from the given business id
    """
    url = PATH + name + QUERY + "0"
    soup = _scrap_as_soup(url)

    reviews = None
    try:
        for rev in soup.find_all("script", type="application/ld+json")[1].children:
            reviews = rev.lstrip().rstrip()
    except:
        for rev in soup.find_all("script", type="application/ld+json")[0].children:
            reviews = rev.lstrip().rstrip()

    reviews = json.loads(reviews)
    num_reviews = 0
    if "aggregateRating" in reviews:
        num_reviews = reviews["aggregateRating"]["reviewCount"]
    return reviews, num_reviews

def _get_20_reviews(name, start, all_reviews):
    """
    Gets and appends additional 20 reviews from the given business id
    """
    url = PATH + name + QUERY + str(start)
    soup = _scrap_as_soup(url)

    reviews = None
    try:
        for rev in soup.find_all("script", type="application/ld+json")[1].children:
            reviews = rev.lstrip().rstrip()
    except:
        for rev in soup.find_all("script", type="application/ld+json")[0].children:
            reviews = rev.lstrip().rstrip()

    reviews = json.loads(reviews)
    all_reviews["review"].append(reviews["review"])

def get_all_reviews(name):
    """
    Returns all the reviews from the given business id
    """
    all_rev, num_reviews = _first_20_reviews(name)

    if num_reviews == 0:
        return all_rev

    start = 20
    while start < num_reviews:
        _get_20_reviews(name, start, all_rev)
        start += 20

    return all_rev

def write_as_json(json_dict, fm):
    """
    Given a dictionary and a file name, it writes the dictionary
    into a json file
    """
    with open(fm+".json", 'w') as f:
        f.write(json.dumps(json_dict, indent=4))

def main():
    import os
    os.chdir(DATA_PATH)
    with open(FILE_PATH, "r") as f:
        for i, line in enumerate(f):
            line=line.rstrip()
            rev = get_all_reviews(line)
            write_as_json(rev, line)

if __name__ == '__main__':
    main()
