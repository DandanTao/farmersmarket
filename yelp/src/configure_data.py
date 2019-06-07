#!/usr/bin/python3
import os
import json

PATH = "../reviews_2018"
DATA_PATH = "../data"
OUTFILE="aggregated_reviews.txt"

def main():
    """
    Combines all reviews into one file to analyze each review
    by sentences.
    """
    os.chdir(PATH)
    all_rev = []
    for filename in os.listdir(os.getcwd()):
        with open(filename, "r") as f:
            data = json.load(f)
            if 'review' in data:
                for rev in data['review']:
                    if 'description' in rev:
                        all_rev.append(rev['description'])

    os.chdir(DATA_PATH)

    with open(OUTFILE, "w") as outfile:
        outfile.write(" ".join(x for x in all_rev))
if __name__ == '__main__':
    main()
