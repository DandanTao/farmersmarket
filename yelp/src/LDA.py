# Modified code from https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

import gensim
from proprocess import preprocess
import numpy as np
from gensim import corpora, models
