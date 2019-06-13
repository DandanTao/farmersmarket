import os
import json
import subprocess

REVIEW_DIR = '/home/jaelee/farmersmarket/yelp/reviews_2019/'
REVIEWS_SENTENCES_DIR = "/home/jaelee/farmersmarket/yelp/reviews_2019_sentences/"
SEPARATE_SENTENCE_DIR = "/home/jaelee/farmersmarket/yelp/reviews_2019_separate_sentences/"
SENTENCE_BOUNDARY_DIR = "/home/jaelee/farmersmarket/sentenceboundary/"

def merge_all_reviews(dir):
    for file in os.listdir(dir):
        alias = file.split(".json")[0]
        with open(dir + file) as f:
            data = json.load(f)
            sentences = "None"
            if "review" in data:
                sentences = ''.join(x['description'] for x in data['review'])

            with open(REVIEWS_SENTENCES_DIR + alias + "_sentences_in.txt",  'w') as f:
                f.write(sentences)

def separate_reviews_by_sentence(dir):
    os.chdir(SENTENCE_BOUNDARY_DIR)
    cmd = SENTENCE_BOUNDARY_DIR + 'sentence-boundary.pl -d ' + SENTENCE_BOUNDARY_DIR+ 'HONORIFICS -i {} -o {}'

    for file in os.listdir(dir):
        alias = file.split("_sentences_in.txt")[0]
        os.system(cmd.format(dir+file, SEPARATE_SENTENCE_DIR+alias+".txt"))

def main():
    merge_all_reviews(REVIEW_DIR)
    separate_reviews_by_sentence(REVIEWS_SENTENCES_DIR)

if __name__ == '__main__':
    main()
