from gensim.models import Word2Vec


model = Word2Vec.load('data_model')

res = model.similarity('英国','贵州')
print(res)