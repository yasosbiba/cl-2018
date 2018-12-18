import wget
import os
from ufal.udpipe import Model, Pipeline
import gensim
from nltk import sent_tokenize
import ast


#udpipe_url = 'http://rusvectores.org/static/models/udpipe_syntagrus.model'
#modelfile = wget.download(udpipe_url)

modelfile = 'udpipe_syntagrus.model'
textfile = 'сонеты_исходник.txt'

def tag_ud(text='Текст нужно передать функции в виде строки!', modelfile='udpipe_syntagrus.model'):
    model = Model.load(modelfile)
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
    
text = open(textfile, 'r', encoding='utf-8').read()

sentences = sent_tokenize(text)
list_of_lists = []
for sentence in sentences:
    list_of_lists.append(tag_ud(' '.join((gensim.utils.simple_preprocess(sentence))), modelfile=modelfile))

with open('сонеты_обработанные.txt', 'w', encoding='utf-8') as f:
    f.write("[\n")
    for l in list_of_lists:
        f.write("%s,\n" % l)
    f.write("\n]")
f.close()

f = open('сонеты_обработанные.txt', 'r', encoding='utf-8')
mylist = ast.literal_eval(f.read())
print("\n\n")
print(mylist)

