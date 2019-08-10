import os
from merge_sentences import BASE_PATH

# USAGE: sentence_boundary.pl -d HONORIFICS -i input_file -o output_file

SEN_BOUNDARY_PATH = '/Users/jaewooklee/farmers_market/sentenceboundary'
HONORIFICS = os.path.join(SEN_BOUNDARY_PATH, 'HONORIFICS')
SEN_BOUNDARY_PL = os.path.join(SEN_BOUNDARY_PATH, 'sentence-boundary.pl')
OUT_BASE = '/Users/jaewooklee/farmers_market/twitter/separate_sentences_out'

def separate_sentences():
    for file in os.listdir(BASE_PATH):
        if not file.startswith('label'):
            continue
        infile = os.path.join(BASE_PATH, file)
        outfile = os.path.join(OUT_BASE, file)
        cmd = f'{SEN_BOUNDARY_PL} -d {HONORIFICS} -i {infile} -o {outfile}'
        os.system(cmd)
