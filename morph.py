import ast
from functools import reduce
import gensim
from nltk import sent_tokenize
from nltk.corpus import stopwords
import os
import re
from string import punctuation
from ufal.udpipe import Model, Pipeline
import wget

modelfile = 'udpipe_syntagrus.model'
model = Model.load(modelfile)
russian_stopwords = stopwords.words("russian")
curdir = os.getcwd()

# funcion for lemmatization from
# https://github.com/akutuzov/webvectors/blob/master/preprocessing/rusvectores_tutorial.ipynb
def tag_ud(text='Текст нужно передать функции в виде строки!', modelfile='udpipe_syntagrus.model'):
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    processed = pipeline.process(text) # обрабатываем текст, получаем результат в формате conllu
    output = [l for l in processed.split('\n') if not l.startswith('#')] # пропускаем строки со служебной информацией
    tagged = [w.split('\t')[2].lower() + '_' + w.split('\t')[3] for w in output if w] # извлекаем из обработанного текста лемму и тэг
    tagged_propn = []
    propn  = []
    for t in tagged:
        if t.endswith('PROPN'):
            if propn:
                propn.append(t)
            else:
                propn = [t]
        else:
            if len(propn) > 1:
                name = '::'.join([x.split('_')[0] for x in propn]) + '_PROPN'
                tagged_propn.append(name)
            elif len(propn) == 1:
                tagged_propn.append(propn[0])
            tagged_propn.append(t)
            propn = []
    return tagged_propn

#open text
def open_untouched_text(textfile):
    text = open(curdir + '/data/untouched/' + textfile, 'r', encoding='utf-8').read()
    return text

#split to sentences
def split_to_sentences(text):
    sentences = sent_tokenize(text.lower())
    return sentences

#split to words
def split_to_words(sentences):
    splitted_text = []
    for s in sentences:
        splitted_text.append(gensim.utils.simple_preprocess(s))
    return splitted_text

#delete stopwords
def delete_stopwords(splitted_text):
    text_without_stopwords = []
    for s in splitted_text:
        t = []
        for w in s:
            if w not in russian_stopwords:
                t.append(w)
        text_without_stopwords.append(t)
    return text_without_stopwords

#add pos tags (lemmatizing it inside)
def add_pos_tags(text_without_stopwords):
    pos_tagged_text = []
    for s in text_without_stopwords:
        pos_tagged_text.append(tag_ud(' '.join(s), modelfile=modelfile))
    return pos_tagged_text

#helper function
def pipeline(* steps):
    return reduce(lambda x, y: y(x), list(steps))

def process_text():
    for filename in os.listdir(curdir + '/data/untouched/'):
        #open our text
        t = open_untouched_text(filename)

        #open file for processed text
        f = open(curdir + '/data/processed/' + filename[:filename.find('_исходник')] + '_обработанный.txt', 'w+', encoding='utf-8')
    
        #text processing
        pos_tagged_text = pipeline(t, split_to_sentences, split_to_words, delete_stopwords, add_pos_tags)
    
        #write out result
        f.write("[\n")
        for l in pos_tagged_text:
            f.write("%s,\n" % l)
        f.write("\n]")
        f.close()

process_text()
