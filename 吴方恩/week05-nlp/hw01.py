
import jieba
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from resources.bm25_code import bm25
import numpy as np
def fix_comments():
    fixed = open('吴方恩/week05-nlp/resources/doubanbook_fixed_comments.txt','w')

    lines = [line for line in open('吴方恩/week05-nlp/resources/doubanbook_top250_comments.txt')]

    for i, line in enumerate(lines):
        if(i==0):
            fixed.write(line)
            prev_line = ''
            continue

        terms = line.split('\t')
        if(terms[0]==prev_line.split('\t')[0]): # 是同一本书
            if (len(prev_line.split('\t'))==6): # 这一行是六列
                fixed.write(prev_line+'\n')
                prev_line = line.strip()
            else:
                prev_line = ''
        else:
            if(len(terms)==6): # 这一行是六列
                prev_line = line.strip()
            else:
                prev_line += line.strip()
    fixed.close()

def preprocess_file():
    # 读取修复后的评论文件
    lines = [line for line in open('吴方恩/week05-nlp/resources/doubanbook_fixed_comments.txt')]

    book_dict = {}
    for line in lines:
        # 按制表符分割字段
        terms = line.split('\t')
        book_name = terms[0]      # 提取书名（首字段）
        book_comment = terms[5]   # 提取评论（第6个字段）
        
        # 过滤无效数据
        if(book_name=='book' or book_name==''):
            continue
            
        # 初始化书籍条目（如果不存在则创建空列表）
        book_dict[book_name] = book_dict.get(book_name,[])
        # 使用结巴分词处理评论并存入字典
        book_dict[book_name].extend(jieba.lcut(book_comment))
    
    # 将处理结果保存为JSON文件
    with open('吴方恩/week05-nlp/resources/data.json','w') as f:
        json.dump(book_dict,f)


if __name__ == '__main__':
    with open('吴方恩/week05-nlp/resources/data.json','r') as f:
        book_dict = json.load(f)
    book_names = []
    book_comments = []
    for book_name in book_dict:    
        book_names.append(book_name)    
        book_comments.append(book_dict[book_name])
    # print(book_names[0],book_comments[0])

    stopwords = [line.strip() for line in open('吴方恩/week05-nlp/resources/stopwords.txt','r')]

    # vectorizer = TfidfVectorizer(stop_words=stopwords)
    # result_matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comments])
    result_matrix = bm25(book_comments)
    similarity_matrix = cosine_similarity(result_matrix)

    print(book_names)
    while True:
        input_book = input('请输入书名：')
        input_book.strip()
        input_book_index = book_names.index(input_book)
        recommed_book_idx = np.argsort(-similarity_matrix[input_book_index])[1:11]
        for idx in recommed_book_idx:
            print(f"《{book_names[idx]}》 \t 相似度：{similarity_matrix[input_book_index][idx]*100:.2f}%")