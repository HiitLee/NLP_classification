# TextClassification_Korean



## Objective

요즘 자연어 처리 하면 "언어모델"이라는 키워드가 반드시 언급됩니다. 과거에 CNN, RNN, LSTM, GRU.. 의 모델은 어느샌가 등장하지 않고, 대용량의 corpus로 학습한 언어모델이 Sota의 성능을 보이며  국내 모든 기업에서 이를 개발하기 위한 연구를 하고 있습니다.

최근 대용량 언어모델로는 BERT, RoBERTa, XLNet, ALBERT, RoBERTa, T5, ELECTRA 등이 발표되었습니다.

대부분의 task에서 대용량 corpus로 pretrain한 언어모델들이 sota의 성능을 보이고 있지만, 여전히 다음과 같은 문제가 존재합니다.

1. 대용량의 corpus로 학습하는데에 많은 시간과 비용이 듦
2. 모델이 경량화 되어도 여전히 서비스 하기에 모델이 무거움
3. 한국어의 교착어라는 특성이 존재함
4. 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않음
5. 오타와 신조어가 자주 발생함

본 프로젝트에서는 CNN, RNN, LSTM, C-LSTM, Hierarchical-Attention-LSTM과 같은 가벼운 모델에서 hyperparameter를 찾아 가장 최고의 성능을 내는 모델을 자동으로 선택하게 하는 것을 목표로 함

## Structure

- 





## Model

- LSTM
- Bi-LSTM
- LSTM-Attention
- LSTM-Maxpool
- CNN
- C-LSTM
- Hierarchical-LSTM
- Hierarchical-CLSTM
- Hierarchical-LSTM + LSTM (ensemble)

## How to use

parameter

