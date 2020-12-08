import numpy as np
import math
from pprint import pprint

def tfIdf(dataList):

    tfMapList = []
    wordMap = {}
    wordCount = 0

    for data in dataList:
        tfMap = {}
        for word in data.split():
            if word in tfMap.keys():
                tfMap[word] += 1
            else:
                tfMap[word] = 1

            if word not in wordMap.keys():
                wordMap[word] = wordCount
                wordCount += 1

        tfMapList.append(tfMap)

    print(tfMapList)
    print(wordMap)

    table = [[0] * len(wordMap) for _ in range(len(tfMapList))]
    row = 0

    for tfMap in tfMapList:
        for word, tf in tfMap.items():
            word_count = 0
            for map in tfMapList:
                if word in map.keys():
                    word_count += 1

            idf = math.log10(len(tfMapList) / word_count)
            tf_idf = tf * idf
            column = wordMap[word]
            table[row][column] = tf_idf

        row += 1

    return wordMap, table


dataList = ['차트 가수 증권사 연예 작곡 오늘 신곡 해외 코스피 그리고',
            '경제 오늘 경제 주식 투자 미국 그리고 경제 코스피 주식 해외',
            '미국 차트 신곡 그리고 연예 앨범 작사 해외 작곡 연예',
            '오늘 코스피 투자 그리고 급등 증권사 주식 대표 경제',
            '신곡 솔로 차트 아이돌 그리고 가수 오늘 연예 미국 작곡 신곡']


pprint(tfIdf(dataList))
