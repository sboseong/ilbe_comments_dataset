# written by sboseong

from konlpy.tag import Okt
from tqdm.auto import tqdm
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import random
import pandas as pd
import numpy as np


# okt 형태소 분석기를 이용한 형태소 분석
okt = Okt()

ilbe_file = pd.read_csv('ilbe_comments.csv')
ilbe_comments = ilbe_file['댓글'].tolist()

ilbe_tmp1 = []

for comment in tqdm(ilbe_comments):
    tokens = okt.pos(comment)
    tmp = []
    
    for token in tokens:
        tmp.append(token)
    
    ilbe_tmp1.append(tmp)

# 형태소 분석이 완료된 ilbe 댓글 데이터 저장
with open('ilbe_okt.pkl', 'wb') as f:
    pickle.dump(ilbe_tmp, f)

# 모델 학습
model = Word2Vec(sentences = ilbe_tmp, vector_size=300, window = 5, min_count = 3, sg = 0, epochs=100)

# 모델 저장
model.save('word2vec_ilbecomments.model')

# word2vec에서 시드 어휘로 단어확장

# 1차 확장
hate_dict = {}
hate_seed = ['혐오']
hate_seed1 = ['경멸', '인종차별', '여성혐오', '지역감정', '남혐']
hate_tmp = []

for seed in tqdm(hate_seed1):
    words = model.wv.most_similar(seed, topn=1000)
    
    for word in tqdm(words):
        if word[1] >= 0.35:
            hate_tmp.append(word[0])
            
            print(seed, word)
            
hate_seed1.extend(hate_tmp)
hate_seed1 = list(set(hate_seed1))

# 2차 확장
hate_seed2 = ['차별', '흑인', '인종', '여혐', '전라도', '경상도', '갈등', '페미']
hate_tmp = []

for seed in tqdm(hate_seed2):
    words = model.wv.most_similar(seed, topn=1000)
    
    for word in tqdm(words):
        if word[1] >= 0.5 and word[0] not in hate_seed2:
            hate_tmp.append(word[0])
            
            print(seed, word)
            
hate_seed2.extend(hate_tmp)
hate_seed2 = list(set(hate_seed2))

# 3차 확장
hate_seed3 = ['홍어', '전라디언', '호남', '전라', '백인']
hate_tmp = []

for seed in tqdm(hate_seed3):
    words = model.wv.most_similar(seed, topn=1000)
    
    for word in tqdm(words):
        if word[1] >= 0.525 and word[0] not in hate_seed3:
            hate_tmp.append(word[0])
            
            print(seed, word)
            
hate_seed3.extend(hate_tmp)
hate_seed3 = list(set(hate_seed3))

# 4차 확장
hate_seed4 = ['틀', '빨갱이', '좌좀', '틀니', '동양인']
hate_tmp = []

for seed in tqdm(hate_seed4):
    words = model.wv.most_similar(seed, topn=1000)
    
    for word in tqdm(words):
        if word[1] >= 0.55 and word[0] not in hate_seed4:
            hate_tmp.append(word[0])
            
            print(seed, word)
            
hate_seed4.extend(hate_tmp)
hate_seed4 = list(set(hate_seed4))

hate_seed.extend(hate_seed1)
hate_seed.extend(hate_seed2)
hate_seed.extend(hate_seed3)
hate_seed.extend(hate_seed4)
hate_seed = list(set(hate_seed))

# 벡터 구성
hate_dict_vecsum = np.zeros(300)
s = 0

for hate in hate_seed:
    hate_dict_vecsum += model.wv.get_vector(hate)

hate_dict_vecmean = [(hate_dict_vecsum / len(hate_seed)).tolist()]

output = []

for sent in tqdm(ilbe_tmp):
    values = np.zeros(300)

    for token in sent:
        if token in model.wv.key_to_index:
            values += model.wv.get_vector(token)
    
    if len(sent) == 0:
        values_mean = values
    else:
        values_mean = values / len(sent)

    output.append(values_mean.tolist())

# 코사인 유사도계산
score = cosine_similarity(hate_dict_vecmean, output).tolist()

# 결과 값 출력
t = []

for comment, value in zip(ilbe_comments, score[0]):
    if 0.35 <= value:
        tmp = []
        tmp.append(comment)
        tmp.append(value)
        t.append(tmp)

t.sort(key=lambda x:x[1])

with open('hate.txt', 'w', encoding='utf-8') as f:
    for i in t:
        if i[1] >= 0.35:
            tmp = i[0] + '\n'
            f.write(tmp)

# SVM 분류기

# SVM 모델
svm_model = SVC()

# 테스트 데이터 셋(Kaggle korean hate speech 데이터 셋)
kaggle1 = pd.read_csv('train.hate.csv')
kaggle2 = pd.read_csv('dev.hate.csv')
kaggle = pd.concat([kaggle1, kaggle2])

cond1 = (kaggle.label == 'hate')
cond2 = (kaggle.label == 'none')

kaggle_end = pd.concat([kaggle[cond1], kaggle[cond2]])
kaggle_end = kaggle_end.sample(frac=1)

train_tmp1 = random.sample(t, 50000)
train_tmp2 = random.sample(y, 50000)
train_tmp = train_tmp1 + train_tmp2
random.shuffle(train_tmp)
cond = list(model.wv.key_to_index.keys())

# 훈련 데이터 셋(ilbe 댓글 데이터)
ilbe_file = pd.read_csv('ilbe_comments.csv')
ilbe_comments = ilbe_file['댓글'].tolist()

train_X = []
train_Y = []

for i in tqdm(train_tmp):
    nouns = okt.nouns(i[0])
    nouns_vec = np.zeros(300)
    
    for j in nouns:
        if j not in cond:
            continue
        nouns_vec += model.wv.get_vector(j)
    
    train_X.append(nouns_vec)
    if i[1] > 0:
        train_Y.append(1)
    else:
        train_Y.append(0)

y = []

for comment, value in zip(ilbe_comments, score[0]):
    if value < 0:
        tmp = []
        tmp.append(comment)
        tmp.append(value)
        y.append(tmp)

y.sort(key=lambda x:x[1])

# 모델 훈련
svm_model.fit(train_X, train_Y)

kaggle_end['label'] = kaggle_end['label'].replace({'none':0, 'hate':1})

test_tmp = []
test_X = []
er = 0

for i in kaggle_end['comments']:
    test_nouns = okt.nouns(i)
    test_nouns_vec = np.zeros(300)
    
    for j in test_nouns:
        if j not in cond:
            continue
            
        test_nouns_vec += model.wv.get_vector(j)
        
    test_X.append(test_nouns_vec)

test_X_tmp = []
test_Y_tmp = []
con = np.zeros(300)

for i, j in zip(test_X, kaggle_end['label']):
    if i == con:
        continue
    else:
        test_X_tmp.append(i)
        test_Y_tmp.append(j)

pred = svm_model.predict(test_X)

O = 0
X = 0

for i, j in zip(pred, kaggle_end['label']):
    if i == j:
        O += 1
    else:
        X += 1

accuracy = O / (O + X)

print("Dataset Accuracy : ", accuracy)