# BERT : Bidirectional Encoder Representation from Transformers
- Transformer의 Bidirectional Encoder (따라서 BERT를 알려면 Transformer부터 이해해야 함.)
***
***
## Transformer (2017, by Google)
- encoder + decoder 딥러닝 모델... 기계 번역에 우수
>> encoder는 입력값을 양방향에서 처리, decoder는 왼쪽에서 오른쪽으로 입력값을 단방향으로 처리
#### Transformer의 작동방식
1. 입력값을 encoder에 입력
2. token들이 positional encoding과 더해지고
3. encoder는 이 값들에 대한 행렬 계산을 통해서 attention vector를 만든다.
* attention vector는 token의 의미를 구하기 위함이다.
* 각각의 token은 문장 속 모든 token을 살펴봄으로써 각 token의 의미를 모델에 전달하는 방식으로 학습됨.
* ex) text와 message라는 token이 함께 있을 경우, text는 '글자'보다는 '핸드폰 문자'의 의미가 더 강할 것.
* attention vector는 ex)와 같은 관계를 담는 vector로 보면 됨.
4. attention vector는 fully connected layer를 통해 한 방에 token들이 계산된다.
***
***
***
### BERT 역사
- by Google
- 모든 자연어 처리 분야에서 범용적 성능을 내는 Language Model
- unlabeled data로 모델을 미리 학습 시킨 후, labeled data에 대해 transfer learning을 하는 방식
- 기존에도 GPT, ELMo와 같은 방식이 이러한 전이학습 방식을 취했는데 BERT는 기존과 달리 언어 처리 방식이 bidirectional 하다는 것이 특징
- ![2 ](https://user-images.githubusercontent.com/87637394/147519560-7c1f851e-a5a5-494e-a869-1e672377bfe8.PNG)
***
### BERT를 한 마디로 하자면
- 사전 훈련 언어 모델 : 특정 과제를 하기 전 사전 훈련 embedding (단어->vector)을 통해 성능을 높인다.
- Traditional LM : How are you + doing에서 doing이 나올 것을 예측
- Bidirectional LM : How are [] + doing에서 []부분에 you가 나올 것을 예측 (masked token)
- BERT의 입력값 : 두 문장까지도 가능
* 한 문장 task : 스팸 or not, 긍정 or 부정 // 두 문장 task : 질의 및 응답
- BERT는 token간, 문장 간 상관관계 모두 학습할 수 있다.
***
### 기존 모델과 BERT와의 다른 점 1
- 기존 모델은 데이터 전처리의 embedding을 Word2Vec, glove, fasttext 등으로 처리하였음.
>> 기존 모델의 구조 : [data] -> [model (LSTM, CNN)] -> 분류
- BERT의 구조 : [Corpus 뭉치] -> [BERT] -> [data] -> [model] -> [분류]
>> 추가된 부분은 Corpus 뭉치로 data를 사전 학습 한다는 점.
>> model 부분은 간단한 DNN으로도 좋은 성능을 낼 수 있음.
***
### 기존 모델과 BERT와의 다른 점 2
- MLM (Maksed Language Model)
>> MLM은 input에서 무작위로 몇 개의 token을 mask시킨다. 그리고 이를 transformer 구조에 넣어 주변 단어의 context만 보고 mask된 단어를 예측 (한꺼번에 작업 시행)
- next sentence prediction 
### 내부 동작
![1](https://user-images.githubusercontent.com/87637394/147519984-35a62fb0-27bc-4adb-84b5-e44a127aaa38.PNG)
- 3가지 embedding값의 합이 input
- 모든 sentence의 첫 번째 token은 언제나 CLS (Special Classification Token)이다. 문장을 구분하는 용도로 사용할 수 있음.
- Sentence pair는 합쳐져서 single sequence로 입력하게 된다. 각각의 sentence는 실제로 수 개의 sentence로 이루어져 있을 수 있다.
>> (e.g. QA Task는 Question-Paragraph 구조에서 paragraph가 여러 개의 문장일 수 있기 때문에 뒤의 문장을 구분하기 위해서 SEP Token을 사용한다.
>>  SEGMENT Embedding의 역할 : 앞의 문장에 A, 뒤의 문장에 B
***
***
### MLM
