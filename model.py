import sys
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

m = 'ruscorpora_upos_skipgram_300_5_2018.vec.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)
print(model.most_similar('человек_NOUN'))
a = model.most_similar('человек_NOUN')
print(type(a))