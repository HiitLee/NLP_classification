from elmo import Embedder
import torch

e = Embedder("/home/dlwngud3028/TextClassification_Korean/word_embedding/ELMo/output/en/")
sents = ['나는 오늘 밥을 먹었어']
elmo_emb = e.sents2elmo(sents)
elmo_emb = torch.Tensor(elmo_emb)

print(elmo_emb)
print(elmo_emb.shape)



