import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

inp = 'data_open_yuliaofenci.txt'
outp1 = 'data_model'
outp2 = 'wiki_vector'

model = Word2Vec(LineSentence(inp),sg= 0,size = 100,iter = 10,window = 2, min_count = 1, workers = 5)
model.save(outp1)
model.wv.save_word2vec_format(outp2 , binary = False)
