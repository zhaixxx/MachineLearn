from gensim.corpora import WikiCorpus
space = " "

with open('wiki-zh-article.txt', 'w',encoding="utf-8") as f:
    wiki = WikiCorpus('zhwiki-latest-pages-articles.xml.bz2',lemmatize=False,dictionary={})
    for text in wiki.get_texts():
        f.write(space.join(text)+"\n")
    print("Finished Saved")
