import sys
import gensim, logging
from gensim.models import Word2Vec
import requests, re


from flask import Flask
from flask import render_template
from flask import request
from flask import abort

app = Flask(__name__)

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


def api_similarity(m, w1, w2):
    url = '/'.join(['http://rusvectores.org', m, w1 + '__' + w2, 'api', 'similarity/'])
    r = requests.get(url, stream=True)
    return r.text.split('\t')[0]

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/')
def hello_world():
    return render_template('hello.html')

@app.route('/description')
def description():
    return render_template('description.html')

@app.route('/search/synonyms')
def show_synonym():
    word = request.args.get('word', '')
    try:
        res = model.most_similar(word)
    except:
        abort(404)
    res_rv = api_neighbor(MODEL, word, FORMAT)
    return render_template('show_synonyms.html', res=res, res_rv=res_rv)

@app.route('/search/similarity')
def show_similar():
    words = request.args.get('word', '').split()
    res = model.most_similar(words[0], words[1])
    res_rv = api_similarity(MODEL, words[0], words[1])
    return render_template('show_similar.html', res=res, res_rv=res_rv)
