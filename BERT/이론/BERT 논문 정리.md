# BERT : Bidirectional Encoder Representation from Transformers
- Transformer의 Bidirectional Encoder (따라서 BERT를 알려면 Transformer부터 이해해야 함.)
***
***
### BERT 역사
- by Google
- 모든 자연어 처리 분야에서 범용적 성능을 내는 Language Model
- 기본적으로 unlabeled data로 모델을 미리 학습 시킨 후, labeled data에 대해 transfer learning을 하는 방식
- 기존에도 GPT, ELMo와 같은 방식이 이러한 전이학습 방식을 취했는데 BERT는 언어 처리 방식이 bidirectional 하다는 것이 특징
***
### BERT를 한 마디로 하자면
- 사전 훈련 언어 모델 : 특정 과제를 하기 전 사전 훈련 embedding (단어->vector)을 통해 성능을 높인다.
***
### 기존 모델과 BERT와의 다른 점 1
- 기존 모델은 데이터 전처리의 embedding을 Word2Vec, glove, fasttext 등으로 처리하였음.
>> 기존 모델의 구조 : [data] -> [model (LSTM, CNN)] -> 분류
- BERT의 구조 : [Corpus 뭉치] -> [BERT] -> [data] -> [model] -> [분류]
>> 추가된 부분은 Corpus 뭉치로 data를 사전 학습 한다는 점.
>> model 부분은 간단한 DNN으로도 좋은 성능을 낼 수 있음.
***
### 기존 모델과 BERT와의 다른 점 2
- 
### 내부 동작
