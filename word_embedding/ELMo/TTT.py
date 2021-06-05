from elmo import Embedder

e =Embedder("/home/dlwngud3028/TextClassification_Korean/word_embedding/ELMo/output/en/")
sents = ['나는 오늘 밥을 먹었어', '너 오늘 밥 어떗냐', '시벌 밥 맛없더라']
d = e.sents2elmo(sents)
print(d)
print(len(d[1][0]))
print(len(d[2]))



