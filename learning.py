# This Python file uses the following encoding: utf-8
import sys
import gensim, logging
import ast
import os
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

curdir = os.getcwd()

def create_corpus(corpus):
    for filename in os.listdir(curdir + '/data/processed/'):
        #open our text
        f = open(curdir + '/data/processed/' + filename, 'r', encoding='utf-8')
        t = ast.literal_eval(f.read())
        for s in t:
            corpus.append(s)
        f.close()

corpus = []
create_corpus(corpus)

model = Word2Vec(corpus, min_count=20, size=100, window=15, workers=4, sg=0, hs=1, iter=10)
model.save('shakespeare.model')
#print(model.most_similar('любовь_NOUN'))