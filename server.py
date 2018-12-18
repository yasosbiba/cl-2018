import sys
import gensim, logging
from gensim.models import Word2Vec
import requests, re


from flask import Flask
from flask import render_template
from flask import request


app = Flask(__name__)

#m = 'ruscorpora_upos_skipgram_300_5_2018.vec.gz'
m = 'shakespeare.model'
model = Word2Vec.load(m)
#similar = model.most_similar('человек_NOUN')

MODEL = 'ruscorpora_upos_skipgram_300_5_2018'
FORMAT = 'csv'

def api_neighbor(m, w, f):
    neighbors = {}
    url = '/'.join(['http://rusvectores.org', m, w, 'api', f]) + '/'
    r = requests.get(url=url, stream=True)
    for line in r.text.split('\n'):
        try: # первые две строки в файле -- служебные, их мы пропустим
            word, sim = re.split('\s+', line) # разбиваем строку по одному или более пробелам
            neighbors[word] = sim
        except:
            continue
    return neighbors

@app.route('/')
def hello_world():
    return render_template('hello.html')

@app.route('/search')
def show_similar():
    word = request.args.get('word', '')
    res = model.most_similar(word)
    #res = request.form['word']
    return render_template('show_similar.html', res=res)

@app.route('/rusvectors')
def show_similar_rv():
    WORD = request.args.get('word-wv', '')
    res = api_neighbor(MODEL, WORD, FORMAT)
    return render_template('show_similar_rv.html', res=res)
    