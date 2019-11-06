from pyhanlp import *
import codecs

infile = 'data_open_yuliao.txt'
outfile = 'data_open_yuliaofenci.txt'


descsFile = codecs.open(infile, 'rb',encoding='utf-8')
i = 0
with open(outfile, 'w',encoding='utf-8') as f:
    for line in descsFile:
        i +=1
        if i % 10000 == 0:
            print(i)
        line = line.strip()
        dic='data_open_dict.txt'
        # stopwords = [line.strip() for line in open('stopWord.txt',encoding='gbk').readlines()]
        words = HanLP.segment(line)
        print(words)
        # print(Hanlp.segment())
        # for word in words  :
            # if word not in stopwords:
            # f.write(word + ' ')
        # f.write('\n')