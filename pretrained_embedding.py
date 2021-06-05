import csv
from gensim.models import FastText, Word2Vec
from konlpy.tag import Okt
import argparse
from glove import Corpus, Glove


parser = argparse.ArgumentParser(description='fastText parameters')

parser.add_argument('--trainData', type=str, default=None, help='trainData for fastText [default:None]')
parser.add_argument('--model', type=str, default='fastText', help='select embeddign model [default:fastText]')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs for train [default: 5]')
parser.add_argument('--size', type=int, default=300, help='vector size [default: 300]')
parser.add_argument('--window', type=int, default=3, help='window size [default: 3]')
parser.add_argument('--sg', type=int, default=1, help='Training algorithm: skip-gram if sg=1, otherwise CBOW. [default: 1]')
parser.add_argument('--min_count', type=int, default=1, help='Ignore words with number of occurrences below this [default: 1]')
parser.add_argument('--workers', type=int, default=4, help='workers [default: 4]')
parser.add_argument('--save', type=str, default='./saved_pretrained_Embedding/', help='trainData for fastText [default:None]')
args = parser.parse_args()


stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
#okt = Okt()
tokenized_data=[]


print('학습 파일:', args.trainData)
f = open(args.trainData, 'r')
fw = open('trainset.txt', 'w')
lines = f.readlines()

corpus_count = 0
corpus_total = 0
for sentence in lines:
    corpus_count += 1 
    #temp_X = okt.morphs(sentence, stem = True)
    temp_X = sentence.split(' ')
    corpus_total += len(temp_X)
    temp_X = [word for word in temp_X if not word in stopwords]
    tokenized_data.append(temp_X)
    temp_X =' '.join(temp_X)
    fw.write(temp_X+'\n')

f.close()
fw.close()

print("총 학습 문장 수:", corpus_count)
print("총 학습 단어 수:", corpus_total)
'''
corpus_file - 말뭉치 경로
min_count - 총 빈도가 이보다 낮은 모든 단어 무시
size - 단어 벡터의 차원
epoch - number of iterations
total_examples - 전체 문장 수
total_words - 전체 단어 수
window - 문장 내에서 현재 단어와 주변 단어 사이의 최대거리
worker - 작업자 스레드를 사용하여 모델 학습 (=멀티코어를 사용하여 더 빠른 학습)
alpha - 초기 학습률
min_alpha - 학습이 진행됨에 따라 학습률이 min_alpha로 선형적으로 떨어짐
sg: sg=1이면 skip-gram, 그렇지 않으면 CBOW

'''

if(args.model == 'fastText'):
    model = FastText(corpus_file = 'trainset.txt', iter  = args.epochs, size=args.size,  sg = args.sg, window=args.window, min_count=args.min_count, workers = args.workers)
    model.save(args.save+'fastText.model')
    model = FastText.load(args.save+'fastText.model')
    print("완성된fastText 임베딩 크기 확인:", model.wv.vectors.shape)
elif(args.model == 'word2vec'):
    model = Word2Vec(sentences  = tokenized_data, size = args.size, window=args.window, min_count=args.min_count, workers = args.workers)
    model.save(args.save+'Word2Vec.model')
    model = Word2Vec.load(args.save+'Word2Vec.model')
    print("완성된word2vec 임베딩 크기 확인:", model.wv.vectors.shape)
elif(args.model == 'glove'):
    corpus = Corpus() 
    corpus.fit(tokenized_data, window=5)
    glove = Glove(no_components=100, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    model.save(args.save+'glove.model')
    model = FastText.load(args.save+'glove.model')
    print("완성된glove 임베딩 크기 확인:", model.wv.vectors.shape)
    

print(model.wv.most_similar("핸드폰"))
print(model.wv.most_similar("도로"))
