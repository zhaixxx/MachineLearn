import codecs
import jieba
from pyhanlp import *
import re

infile = 'data_open_yuliao.txt'
outfile = 'data_open_yuliaofenci.txt'
jieba.load_userdict('data_open_dict.txt')
descsFile = codecs.open(infile, 'rb',encoding='utf-8')
i = 0
with open(outfile, 'w',encoding='utf-8') as f:
    for line in descsFile:
        i +=1
        if i % 10000 == 0:
            print(i)
        line = line.strip()
        stopwords = [line.strip() for line in open('stopWord.txt',encoding='gbk').readlines()]
        words = jieba.cut(line,cut_all=False )
        # print(Hanlp.segment())
        for word in words  :
            if word not in stopwords:
                f.write(word + ' ')
        f.write('\n')

