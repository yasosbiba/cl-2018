# This Python file uses the following encoding: utf-8
import sys
import gensim, logging
import ast

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import word2vec
from gensim.test.utils import get_tmpfile

f = open('сонеты_обработанные.txt', 'r', encoding='utf-8')
data = ast.literal_eval(f.read())
f.close()

model = word2vec.Word2Vec(data, min_count=5, size=50, window=5, workers=4, sg=1, hs=1)
model.save('shakespeare.model')
print(model.most_similar('глаз_NOUN'))