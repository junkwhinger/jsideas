---
layout:     post
title:      "Self Attention: 이름 분류기"
date:       2018-09-01 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/selfattention/variants_of_junsik.png
---



# Self Attention: 이름 분류기



## Intro

교환학생 시절 영국에서 새로 사귄 친구들에게 'Junsik'은 쉽지 않은 발음이었다. 'sik'이 'sick'처럼 들리기도 했고 기억하기도 쉽지 않은듯해 'Junsik'을 줄인 'June'을 쓰기로 했다. 그 이름이 사실 여자 이름이라는 건 몇달이 지난 후 현지 친구를 통해 알게 되었다. 그 날 이후 e를 빼고 'Jun'을 쓰고 있다.



문화/언어권에 따라 학습이 필요한 경우도 있지만 대체로 우리는 누군가의 이름을 듣고 어렵지 않게 그 사람이 남자인지 여자인지 떠올릴 수 있다. 'Robert', 'Mark', 'Dave'처럼 남자 이름은 센 발음으로 구성되고, 여자 이름은 'Lucy', 'Stella', 'Valerie'처럼 부드러운 발음으로 구성되는 경우가 많다. 또 'Julius(남)', 'Julia(여)'처럼 끝마무리에 따라 성별을 구분짓기도 한다.



우리가 이름을 듣고 성별을 떠올리는 사고과정을 뉴럴 네트워트가 흉내낼 수 있을까? 또 우리가 발음이나 끝 부분을 보고 판단을 내리듯, 뉴럴 네트워크가 이름의 어느 부분을 보고 성별을 결정하는지 알 수 있을까? 



## Dataset

구글에 "Baby Names"로 검색하면 성별로 인기있는 영문 이름을 제공하는 사이트를 여럿 찾을 수 있다. 이 중 하나를 `selenium` 과 `BeautifulSoup`을 사용해 크롤링하여 남아 이름 15,546건, 여아 이름 21,798건을 얻었다. 내가 크롤링한 사이트에서는 이름이 속한 언어권(Origin) 정보도 제공했다. 언어권에 따라 패턴이 달라질 수 있으므로, `성별`과 `언어권`을 기준으로 training / validation set을 만들었다.

### example

| babyname | sex  | origin |
| -------- | ---- | ------ |
| Aakesh   | boy  | Indian |
| Aaren    | boy  | Hebrew |
| Abalina  | girl | Hebrew |
| ...      | ...  |        |



### Exploration

성별에 따라 이름은 어떤 특징을 가지고 있을까?

#### 가장 많이 사용된 첫번째 글자은?

| Rank | Total               | Girl               | Boy                 |
| ---- | ------------------- | ------------------ | ------------------- |
| 1    | A - 3,792건 (10.2%) | A - 2,118건 (9.7%) | A - 1,674건 (10.8%) |
| 2    | C - 2,887건 (7.7%)  | C - 1,880건 (8.6%) | B - 1,051건 (6.8%)  |
| 3    | S - 2,862건 (7.7%)  | S - 1,825건 (8.4%) | M - 1,051건 (6.8%)  |
| 4    | M - 2,615건 (7.0%)  | M - 1,564건 (7.2%) | S - 1,037건 (6.7%)  |
| 5    | K - 2,262건 (6.1%)  | K - 1,531건 (7.0%) | C - 1,007건 (6.5%)  |

성별에 관계없이 A가 1등이라는게 의외였다. 남아 이름에서는 'B'가 눈에 띈다.



#### 가장 많이 사용된 첫 2개 글자 조합은?

| Rank | Total               | Girl              | Boy               |
| ---- | ------------------- | ----------------- | ----------------- |
| 1    | Ma - 1,454건 (3.9%) | Ma - 907건 (4.2%) | Ma - 547건 (3.5%) |
| 2    | Ka - 914건 (2.4%)   | Ka - 726건 (3.3%) | Ha - 368건 (2.4%) |
| 3    | Sh - 864건 (2.3%)   | Sh - 715건 (3.3%) | Al - 336건 (2.2%) |
| 4    | Ca - 856건 (2.3%)   | Ca - 624건 (2.9%) | Da - 305건 (2.0%) |
| 5    | Al - 842건 (2.3%)   | Ch - 600건 (2.8%) | De - 300건 (1.9%) |

첫 2개 글자 집계 결과는 조금 다르다. 'Ma' 조합이 맨 위로 올라온다. 'Ma'로는 'Mason', 'Maggie' 정도 밖에 생각나지 않았는데 데이터셋을 살펴보니 'Maddock', 'Marcas', 'Marjori' 등 어디서 들어본 이름들이 튀어나온다.

첫 글자에 비해 첫 2개 글자에서는 성별에 따른 차이가 조금 보이는 듯 하다. 



#### 성별에 따라 가장 차이가 많이 나는 첫 2개 이니셜 조합은?

| Rank | first_two_letters | Girl_ratio | Boy_ratio | Absolute Difference |
| ---- | ----------------- | ---------- | --------- | ------------------- |
| 1    | Sh                | 3.3%       | 1.0%      | 2.3%                |
| 2    | Ka                | 3.3%       | 1.2%      | 2.1%                |
| 3    | Ch                | 2.8%       | 1.3%      | 1.5%                |
| 4    | Ha                | 0.9%       | 2.4%      | 1.5%                |
| 5    | Ca                | 2.9%       | 1.5%      | 1.4%                |
| 6    | Ba                | 0.5%       | 1.7%      | 1.2%                |
| 7    | Ga                | 0.4%       | 1.4%      | 1.0%                |
| 8    | Jo                | 1.8%       | 1.0%      | 0.8%                |
| 9    | La                | 2.0%       | 1.3%      | 0.7%                |
| 10   | Co                | 0.9%       | 1.6%      | 0.7%                |

여아 이름에는 '샤', '카', 'ㅊ', '라', '조' 발음이 더 많이 사용되고, 남아 이름에는 '하', '바', '가', '코'가 더 자주 사용되는 경향이 있었다. 그러나 성별에 따른 차이가 크지는 않아 보인다.



####  가장 많이 사용되는 마지막 글자는?

| Rank | Total                | Girl                 | Boy                 |
| ---- | -------------------- | -------------------- | ------------------- |
| 1    | a - 10,693건 (28.6%) | a - 10,198건 (46.8%) | n - 3,123건 (20.1%) |
| 2    | e - 6,471건 (17.3%)  | e - 4,989건 (22.9%)  | o - 1,519건 (9.8%)  |
| 3    | n - 4,813건 (12.9%)  | n - 1,690건 (7.8%)   | e - 1,482건 (9.5%)  |
| 4    | y - 1,992건 (5.3%)   | y - 1,036건 (4.8%)   | s - 1,448건 (9.3%)  |
| 5    | s - 1,825건 (4.9%)   | i - 927건 (4.3%)     | r - 1,123건 (7.2%)  |

마지막 글자는 첫 이니셜에 비해 특정 글자에 편중된 경향을 보인다. 여아 이름은 'a'로 끝나는 경우가 전체의 절반 가량이나 된다. 'e'도 전체의 약 22%에 달한다. 반대로 남아 이름은 마지막 'n'이 가장 인기있는 선택이다. 'a'로 끝나는 남자 이름은 잘 떠오르지 않는다. 찾아보니 'Nemanja', 'Kapila' 등 동유럽 느낌이 나는 이름들이 다수 있다.



#### 성별에 따라 가장 차이가 많이 나는 마지막 글자 조합은?

| Rank | last_letter | Girl_ratio | Boy_ratio | Absolute Difference |
| ---- | ----------- | ---------- | --------- | ------------------- |
| 1    | a           | 46.8%      | 3.2%      | 43.6%               |
| 2    | e           | 22.9%      | 9.5%      | 13.4%               |
| 3    | n           | 7.8%       | 20.1%     | 12.3%               |
| 4    | o           | 0.8%       | 9.8%      | 9.0%                |
| 5    | s           | 1.7%       | 9.3%      | 7.6%                |
| 6    | r           | 0.9%       | 7.2%      | 6.3%                |
| 7    | d           | 0.7%       | 5.9%      | 5.2%                |
| 8    | l           | 2.6%       | 6.6%      | 4.0%                |
| 9    | k           | 0.1%       | 2.7%      | 2.6%                |
| 10   | t           | 1.5%       | 4.0%      | 2.5%                |

첫 글자나 첫 두글자에 비해 마지막 글자는 더 두드러진 특징을 보인다. 여아 이름은 'a'나 'e'로 끝나는 경우가 더 많고, 남아 이름은 'n'으로 마치는 빈도가 더 높다. 같은 'Juli'로 시작하더라도 끝이 'us'로 끝나면 남아 이름이 되고 'a'로 끝나면 여아 이름이 된다. 마지막 글자가 주요 변수로 떠오른다.



#### 그럼 마지막 두 글자는?

| Rank | Total               | Girl                 | Boy               |
| ---- | ------------------- | -------------------- | ----------------- |
| 1    | na - 2,419건 (6.5%) | na - 2,375건 (10.9%) | on - 890건 (5.7%) |
| 2    | ne - 1,713건 (4.6%) | ne - 1,476건 (6.8%)  | an - 763건 (4.9%) |
| 3    | ia - 1,510건 (4.0%) | ia - 1,472건 (6.8%)  | er - 558건 (3.6%) |
| 4    | ie - 1,022건 (2.7%) | la - 909건 (4.2%)    | in - 482건 (3.1%) |
| 5    | on - 1,000건 (2.7%) | ta - 837건 (3.8%)    | us - 476건 (3.1%) |

빈도가 높았던 마지막 글자의 두글자 버전이 다수 발견되었다. 최빈 5개 글자조합을 보면 이름을 쉽게 떠올릴 수 있다. 여아는 'Anna', 'Anne', 'Aria', 'Carla', 'Violetta', 남아는 'Ron', 'Brian', 'Roger', 'Augstin', 'Albus'.



#### 성별에 따른 차이는?

| Rank | last_letter | Girl_ratio | Boy_ratio | Absolute Difference |
| ---- | ----------- | ---------- | --------- | ------------------- |
| 1    | na          | 10.9%      | 0.3%      | 10.6%               |
| 2    | ia          | 6.8%       | 0.2%      | 6.6%                |
| 3    | ne          | 6.8%       | 1.5%      | 5.3%                |
| 4    | on          | 0.5%       | 5.7%      | 5.2%                |
| 5    | la          | 4.2%       | 0.2%      | 4.0%                |
| 6    | an          | 0.9%       | 4.9%      | 4.0%                |
| 7    | ta          | 3.8%       | 0.2%      | 3.6%                |
| 8    | ra          | 3.6%       | 0.2%      | 3.4%                |
| 9    | er          | 0.4%       | 3.6%      | 3.2%                |
| 10   | da          | 3.3%       | 0.1%      | 3.2%                |

마지막 글자보다는 조금 덜하지만 마지막 2글자 역시 성별에 따른 차이를 대략 확인할 수 있다. 여아이름의 마무리로 'na'가 상당히 많다. 우리 이름에서도 'na'는 여성스러운 느낌이 난다. 안나, 지나, 미나, 이나. 남아 이름으로는 'on'이나 'an'로 끝나는 경우가 많았다. 우리 이름에서는 '시온', '주안' 정도가 있겠다.



간단한 집계를 통해 성별에 따른 이름 구성상의 차이를 살펴보았다. 첫 시작보다는 마지막 첫 혹은 두글자에서 그 성별에 대한 단서를 조금 엿볼 수 있었다. 텍스트를 분류하는 딥러닝 네트워크도 이러한 특징을 잡아낼 수 있을까?



## Model

일반적인 텍스트 분류 모델에서는 LSTM이나 CNN 레이어를 통해 데이터에서 패턴을 추출한다. 그리고 이를 단층 혹은 다층으로 구성된 Fully Connected Layer에 통과시켜 각 레이블에 해당할 확률값을 얻는다.

![basic_lstm_classifier](../assets/selfattention/basic_lstm_classifier.png) 

BiLSTM 모델은 이름을 분류하는 목적은 달성했지만 이름의 어떤 부분에 근거했는지는 말해주지 않는다. Local Interpretable Model-Agnostic Explanations (LIME) 방법을 사용해 글자를 하나씩 빼가면서 어떤 글자가 결과에 가장 큰 영향을 주었는지 판단하는 방법을 쓸 수도 있긴 하지만, 모델 자체는 답을 가지고 있지 않다.



### A Structure Self-Attentive Sentence Embedding

2017년 3월 발표된 A Structured Self-Attentive Sentence Embedding은 Self-Attention을 통해 이 문제를 해결한다. 이 논문에서 제안한 방식은 BiLSTM이나 CNN 레이어를 통과한 feature를 다음과 같은 방식으로 처리한다.

![self_attention](../assets/selfattention/self_attention.png)

BiLSTM 레이어를 통해 얻은 feature를 2개의 FC 레이어에 통과시켜 n_token x hops 크기를 가진 매트릭스 형태의 Attention을 얻는다. 이때 W_s1의 `da`와 W_s2의 `hops`는 하이퍼파라미터로 적절한 값을 선택해 입력한다. 입력 텍스트를 하나의 벡터로 임베딩했던 기존의 Attention 메커니즘과 달리 본 논문에서 제안한 Attention은 매트릭스 형태를 띈다. 이러한 임베딩 추출 방식을 통해 다음과 같은 효과를 기대할 수 있다.

- 문장이 가진 여러 측면의 특성을 다수의 벡터로 표현할 수 있다. 매트릭스의 개별 row가 개별 특성에 대응된다. hop의 수를 늘리면 그만큼 row의 수 역시 늘어난다.
- 분류 대상 문장 외에 별도의 인풋이 없더라도 Attention을 얻을 수 있다.
- Attention이 LSTM의 모든 타임 스텝에 접근할 수 있으므로, LSTM이 지는 장기기억의 부담을 덜어줄 수 있다.
- 마지막으로 Attention을 히트맵의 형태로 얻을 수 있어 쉽게 시각화할 수 있다. `hops` x `n_token`의 크기를 가진 매트릭스를 얻기 때문에 각 토큰에 쉽게 매칭할 수 있다.



BiLSTM 이후에 처리 방식은 다음과 같다.

- LSTM 레이어의 출력값을 2개의 FC 레이어에 통과시키며 `da`x`n_token` => `hops` x`n_token`으로 변형한다. 이후 `hops`디멘션을 기준으로 softmax를 씌워 각 row의 합이 1이 되도록 만들고 이를 Attention이라 한다. 합이 1이므로 히트맵으로 시각화하기 좋다. 

- LSTM 레이어의 출력값을 transpose한 후 이를 Attention과 dot product 연산을 거쳐 Sentence Embedding을 추출한다. (여기서는 이름이므로 Name에 대한 Embedding이 된다.)
- Embedding Matrix를 flatten한 후 FC 레이어 2개를 거쳐 최종적으로 남아/여아 이름일 확률을 구한다.



### loss function

분류 문제에서는 보통 `nn.CrossEntropyLoss`를 사용해 예측값과 타켓 레이블간의 오차를 구한다. 논문에서는 여기에 `Penalization Term`이라는 부가적인 loss를 제안한다. 위에서 도출한 Attention은 matrix의 형태로, 각 row vector는 모델이 집중하는 부분을 의미한다. 만약 이들 벡터가 모두 문장이 특정 부분에만 집중적으로 높은 값을 가지게 되면 (그곳만 바라보게 되면) 굳이 vector가 아닌 matrix로 표현하는 의미가 없어진다. 따라서 row들이 서로 다른 곳을 바라보도록 가이딩을 해주는 역할을 `Penalization Term`이 수행한다.

$ P = ||(AA^T - I )||_F^2$

$A$는 앞에서 구한 Attention으로 그 자신의 역행렬과 곱을 한 후, 그 크기만큼의 Identity Matrix를 빼준다. 그 결과로서 얻는 매트릭스의 Frobenius Norm을 구한 것이 Penalization Term이 된다. 논문에서는 여기에 하이퍼파라미터인 `coef`(1.0)를 곱한 후 원래 loss에 더한다.



## Experiment

### Training Setting

논문에서는 문장의 감성을 분류하거나 작성자의 연령을 맞추는 사례에 Self Attention을 적용했다. 문장을 단어 단위로 자른 후, 단어를 Word Embedding을 통해 숫자로 치환했다. 그리고 이를 네트워크에 통과시켜 분류확률과 어텐션 등을 구했다.

내가 가진 데이터셋은 이름 텍스트로, 그 구성단위가 단어가 아닌 개별 알파벳이다. 글자의 다양성은 단어의 그것에 비해 훨씬 폭이 좁고, 이름 텍스트의 길이도 긴 편이 아니므로, 논문에서 사용한 하이퍼파라미터를 그대로 적용하기에는 무리가 있다고 생각했다.

그래서 일부 파라미터는 논문의 1/10이나 1/5 수준으로 낮춰서 모델 복잡도를 낮추어서 적용해보았다.

```
{
    "num_epochs": 10,
    "batch_size": 16,
    "save_summary_steps": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "embedding_dim": 100,
    "hidden_dim": 300,
    "nb_layers": 1,
    "nb_hops": 5,
    "da": 30,
    "fc_ch": 300,
    "nb_classes": 2,
    "device": "cpu",
    "train_size": 33609,
    "val_size": 3735,
    "vocab_size": 35,
    "coef": 0.5,
    "isPenalize": 1,
    "dropout": 0.0,
    "model": "selfattention"
}
```



### Model Performance

분류 성능 비교를 위해 논문의 베이스라인 모델 중 하나인 BiLSTM + MaxPooling 모델을 만들었다. Self Attention 모델과 같은 하이퍼파라미터를 사용했다 (사용하지 않는 레이어 제외).

| Models                        | Validation Accuracy |
| ----------------------------- | ------------------- |
| BiLSTM + MaxPooling (epoch 8) | 0.884               |
| Self Attention (epoch 10)     | 0.892               |

논문에서는 Self Attention 모델이 다른 두 베이스라인 모델에 비해 성능이 더 좋게 나왔다. 내 실험에서도 Self Attention 모델이 베이스라인 LSTM에 비해 조금 더 좋은 Validation Accuracy를 기록했다. Training Accuracy도 비슷한 수치를 기록해 모델이 적절히 학습되었음을 확인했다.

클래스별로 쪼개어보면..

| Models              | Precision(girl) | Precision(boy) | Recall(girl) | Recall(boy) |
| ------------------- | --------------- | -------------- | ------------ | ----------- |
| BiLSTM + MaxPooling | 0.901           | 0.860          | 0.900        | 0.862       |
| Self Attention      | 0.903           | 0.876          | 0.913        | 0.862       |

두 모델 모두 특정 클래스에 편중된 결과를 보이지는 않았다.



## Visualizing Self Attention

### Attention Heatmap

모델은 이름의 어떤 부분을 보고 판단을 내렸을까? Self Attention은 `hops`갯수의 row vector들로 구성된 matrix이며, 각 vector의 합은 1이 되도록 softmax를 취했다. 이름 텍스트를 전처리하여 모델에 집어넣고 뽑은 Attention을 시각화해보았다.

![attention_heatmap](/Users/junkwhinger/jsideas/assets/selfattention/attention_heatmap.png)

히트맵을 뿌려보면 위와 같이 각 어텐션 벡터(y축)가 이름의 특정 글자들에만 반응하고 있는 것을 볼 수 있다. loss에 더해준 Penalization Term이 가이딩해준 효과로 보인다. 하이퍼파라미터를 튜닝할 때 `hops`를 5개가 아닌 30개로 늘려서도 해보았는데 이때는 거의 모든 글자에 어텐션이 할당되는 결과를 낳았다. 분류 성능에는 차이가 없었으나 해석이 더 어려워졌다. 데이터셋의 형태에 따라 hops의 크기를 적절하게 선택하는 것이 중요해보인다.



### Attention on Names

매트릭스를 바로 시각화하는 것보다 깔끔하게 정리해서 이름위에 뿌리는 것이 여러모로 더 보기가 좋다. 논문에서 제안한 방식대로 각 row를 글자 축을 기준으로 모두 합한 다음 softmax를 취해 하나의 벡터로 만든다음 뿌려보자.

먼저 유명한 이름들부터 해보자. 해리포터의 주요 인물들이다.

![harrypotter](/Users/junkwhinger/jsideas/assets/selfattention/harrypotter.png)

대부분 어텐션이 끝에 몰려있는 가운데, Harry, Hermione와 Albus, Draco는 조금 다른 양상을 보인다. 언어학자는 아니지만 대략 공감이 가는 해석이라고 생각한다.



다음은 마블 시네마틱 유니버스의 캐릭터들이다.

![mcu](/Users/junkwhinger/jsideas/assets/selfattention/mcu.png)

해리포터는 다 맞췄으나 Tony, Loki를 여자로, Pepper를 남자로 분류했다. 흠터레스팅.. 페퍼는 끝의 'er'이 여자 이름에 거의 없기 때문이 아닐까 싶다.



그럼 이름의 끝부분을 조금씩 변경해보면 어떤 결과가 나올까?

![variants_of_cat](/Users/junkwhinger/jsideas/assets/selfattention/variants_of_cat.png)

`Cat`도 여성일 확률이 높았지만, 이후에 `ne`, `na`를 붙임에 따라 분류 확률이 거의 100%에 근접하게 올라가는 것을 확인할 수 있었다.

![variants_of_chris](/Users/junkwhinger/jsideas/assets/selfattention/variants_of_chris.png)

 

모델은 `Chris`와 `Christina`를 여자로 분류했다. 그러나 끝에 `o`를 붙여 우리 형을 만들자 매우 높은 확률로 남자로 분류했다.  `Christian`을 여자로 분류한 점이 조금 아쉽다.  



일부 잘못 판단한 경우가 있긴 했지만 어느정도 괜찮은 성능을 보인다. 데이터셋에 없는 전혀 다른 언어권에도 이 모델이 잘 동작할까? 해외 아기 이름을 크롤링해서인지 데이터셋에는 아시안 언어권의 이름이 서구권에 비해 상대적으로 적었고 한국어권 이름은 아예 없었다. 한글 이름을 넣어보자. 회사 동료들의 이름을 랜덤으로 소환했다.

 

![koreanboys](/Users/junkwhinger/jsideas/assets/selfattention/koreanboys.png)

남자는 올패스.



![koreangirls](/Users/junkwhinger/jsideas/assets/selfattention/koreangirls.png)

여자는 모두 틀렸다. 모델이 한국어 이름의 엔딩을 남성적으로 판단한 것으로 보인다.



마지막으로 내 이름은 어떨까?

![variants_of_junsik](/Users/junkwhinger/jsideas/assets/selfattention/variants_of_junsik.png)

아쉽게도 Jun을 여자로 분류하긴 했으나 June보다는 확률이 좀 떨어졌다. 뉴럴 네트워크에게도 June은 여자 이름처럼 들렸나보다.



### On Embedding Space

모델의 forward propagation step을 살펴보면, 추출한 Attention을 LSTM의 출력에 곱해 Sentence Embedding(내 실험에서는 Name Embedding)을 얻는다. 이 Embedding을 텍스트 인풋의 수치적 표현이라고 생각한다면, 이 역시 연속적 평면에 시각화할 수 있을 것이다. MNIST 데이터를 2차원 평면에 늘어놓았을때 같은 레이블끼리 뭉친것처럼, 이름들도 그 특성에 따라 뭉쳐있을까?

모델에서 도출한 Embedding은 Matrix 형태로 되어있으므로, 이를 flatten하여 하나의 긴 벡터로 변환한 후 TNSE를 사용해 2차원으로 축소시켜보았다. 전체 3만건의 임베딩을 뽑고 TSNE를 돌리는게 조금 버거워 남녀 인기순위 100개 씩을 뽑아 시각화해본다.

먼저 성별에 따라 살펴보자.

![bySex](/Users/junkwhinger/jsideas/assets/selfattention/bySex.png)

재밌는 결과가 나왔다. 남아 이름들은 우상단으로, 여아 이름들은 좌하단에 몰려있는 경향을 보인다. 자세히 들여다보면 이름의 끝부분끼리 몰려있다. 먼 좌하단은 Mia, Sophia, Olivia, Victoria들이 있고, 먼 우상단에는 Nicholas, Thomas, Lucas가 뭉쳐있다. 성별이 다르지만 끝부분이 유사한 Scarlett, Wyatt, Robert, Margaret도 서로 근접해 위치한다.



혹 이름이 속한 문화권이 그 이름의 끝부분을 결정짓는걸까? 이름의 문화권을 기준으로 살펴보자.

![byOrigin](/Users/junkwhinger/jsideas/assets/selfattention/byOrigin.png)



성별만큼 명확하지는 않지만 이름들은 같은 문화권끼리 뭉치는 경향을 보여준다. 중앙 상단의 Grayson, Jackson, Brandon, Jameson 등 영국에서 온 슨자 돌림 남자들이 보인다. 좌하단에는 Valentina, Emilia, Victoria, Aurora, Olivia 등 아자 돌림 라틴 여자분들도 있다.



남자들만-

![boysByOrigin](/Users/junkwhinger/jsideas/assets/selfattention/boysByOrigin.png)



여자들만-

![girlsByOrigin](/Users/junkwhinger/jsideas/assets/selfattention/girlsByOrigin.png)





### Matrix Computation: Emilia - Emily + Lucy?

King - Man + Woman = Queen은 Word Embedding의 멋진 사례 중 하나로 빠짐없이 등장한다. 왕에서 남자라는 성을 제거하면 권력이 남고, 그 권력을 여자라는 성에 더하면 여왕이 된다. 관념적으로도 말이 되고 간단한 사칙연산으로도 말이 된다. 뉴럴 네트워크를 통해 얻은 숫자 뭉치가 우리가 가진 관념적인 정보를 담고 있음을 보여주는 멋진 결과다.

![king_to_queen](/Users/junkwhinger/jsideas/assets/selfattention/king_to_queen.jpeg)

source: https://medium.com/@thoszymkowiak/how-to-implement-sentiment-analysis-using-word-embedding-and-convolutional-neural-networks-on-keras-163197aef623  



단어와 마찬가지로 이름을 임베딩 스페이스상에 표현할 수 있다면 사칙연산을 통해 재밌는 결과를 만들어볼 수 있을까? 이름은 단어만큼 풍부한 의미를 가지지는 않지만, 그냥 해보자.



먼저 Training과 Validation 데이터셋을 모두 모델에 넣어 이름의 임베딩을 얻었다. 각 임베딩은 `hops`x`hidden_dim * 2`크기의 매트릭스 형태로 내 실험에서는 5x600 크기였다. 이름 3개를 고른 후 element-wise로 - + 연산을 수행하여 query matrix를 만든다. 그리고 이 query matrix와 모든 embedding matrix간의 matrix euclidean distance를 구하고, distance가 가장 낮은 5개를 출력해보았다.



1) Emilia - Emily  + Lucy = Lucia!

![emily](/Users/junkwhinger/jsideas/assets/selfattention/emily.png)

Emilia에서 Emily를 빼면 ~ia가 남고 거기에 Lucy를 더하면 Lucia가 될거라고 생각했는데 실제로 됐다!



2) Susie - Susanne + Roxie = Roxie!

![susie](/Users/junkwhinger/jsideas/assets/selfattention/susie.png)

3) Christina - Christine + Austine = Austina!

![christina](/Users/junkwhinger/jsideas/assets/selfattention/christina.png)

예상과 같은 결과가 나오긴 하지만 King - Man + Woman = Queen 만큼 어떤 추상적인 의미를 조작했다고 보기는 어렵다. 예를 들어 "Paul"에서 "John"을 뺀 다음 "Hank"를 더하면 어떨까?

4) Paul - John + Hank = ?

![paul](/Users/junkwhinger/jsideas/assets/selfattention/paul.png)

 "Bank"라는 결과를 얻을 수 있지만 어떤 로직이나 의미가 느껴지지 않는다.



## Outro

이번 포스팅에서는 LSTM + Self Attention 모델을 사용해 남아 / 여아 이름을 분류하는 모델을 만들고, Attention을 통해 뉴럴 네트워크가 이름의 어떤 부분에 집중했는지 시각화해보았다. 뉴럴 네트워크가 집중했던 흔적은 보통 이름의 끝 부분에 남겨져 있었다. 같은 Jun이라도 e가 붙으면 여자가 되고 sik이 붙으면 남자가 되었다. 분석 전에 내가 어렴풋이 생각했던 가설과 비슷해 놀라웠다. 또 2차원 공간에 뿌려진 이름들이 성별이나 문화권에 따라 끼리끼리 모이는 점도 흥미로웠다. 해석가능한 딥러닝은 내가 생각하는 방식을 되돌아보는 재미가 있다.