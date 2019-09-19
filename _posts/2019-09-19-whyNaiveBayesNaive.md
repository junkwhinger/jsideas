---
layout:     post
title:      나이브 베이즈는 왜 나이브한가
date:       2019-09-19 00:00:00
author:     "Jun"
<!-- img: 20190919.png -->
tags: [python, Machine Learning]

---

# Naive Bayes

나이브 베이즈는 무엇인가? 위키피디아 첫 줄을 보자.

> In machine learning, naïve Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features. ([https://en.wikipedia.org/wiki/Naive_Bayes_classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier))

3가지 주요 키워드가 보인다. 하나씩 알아보자.

1. probabilistic classifier
2. Bayes' theorem
3. naïve independence assumption

<br>

## 1. probabilistic classifiers

> In machine learning, a probabilistic classifier is a classifier that is able to predict, given an observation of an input, a probability distribution over a set of classes, rather than only outputting the most likely class that the observation should belong to. Probabilistic classifiers provide classification that can be useful in its own right[1] or when combining classifiers into ensembles. ([https://en.wikipedia.org/wiki/Probabilistic_classification](https://en.wikipedia.org/wiki/Probabilistic_classification))

probabilistic classifier는 인풋이 주어졌을 때 가장 확률이 높은 타깃 클래스 하나를 리턴하는 것이 아니라, 타깃 클래스 세트에 대한 **확률 분포를 출력**한다. 예를 들어, 오늘 날씨를 보고 내일 날씨가 "비"라고 하는 것이 아니라, "맑음 20%, 비 50%, 흐림 30%"라는 결과를 준다.

즉, 나이브 베이즈는 클래스 자체가 아닌 클래스들이 갖는 확률을 리턴한다.

<br>

## 2. Bayes' theorem

> In probability theory and statistics, Bayes’ theorem (alternatively Bayes’ law or Bayes’ rule) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if cancer is related to age, then, using Bayes’ theorem, a person's age can be used to more accurately assess the probability that they have cancer this can be done without knowledge of the person’s age. ([https://en.wikipedia.org/wiki/Bayes'_theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem))


베이즈 정리는 어떤 이벤트와 관련된 조건에 대한 사전 믿음으로 그 이벤트가 발생할 확률을 표현한다.

<br>

### 2.1 양성 판정을 받았을 때 진짜 병 걸렸을 확률을 베이즈 정리로 알아보자.

아래 영상에서 설명한 내용을 간단히 소개하면 아래와 같다. 

[https://www.youtube.com/watch?v=R13BD8qKeTg](https://www.youtube.com/watch?v=R13BD8qKeTg)

- 컨디션이 별로라서 병원에 갔더니 천명 중 한명 걸린다는 희귀병 xx병 테스트 결과 양성 뜸.
- 이거 얼마나 정확한 거임? 의사 왈: 병 걸린 사람이 테스트하면 정확하게 분류할 확률이 99%임
- 그러나! 내가 그 병에 걸렸을 확률이 99%를 의미하는 것이 아님.
- 왜? "병 걸린 사람이" 라는 조건이 붙었기 때문. 나는 병에 걸렸는지 모르니까 조건이 안 걸린 상태임.
- 나는 양성이 떴을 때 병에 걸렸을 확률을 알고 싶은 거임

    $$p(병|양성)$$

- 그런데 가진 것은 병에 걸릴 확률(전체 인구 중 환자 수)와 병에 걸렸을 때 양성일 확률임

    $$p(병) = 0.001,\ p(양성|병)=0.99$$

- 베이즈 정리를 이용하면 이 두가지 정보를 이용해 원하는 확률을 얻을 수 있음.

    $$P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{P(B|A)P(A)}{P(A)P(B|A) + P(-A)P(B|-A)}$$

- 이걸 응용하면

    $$P(병|양성) = \frac{p(양성|병)P(병)}{P(병)P(양성|병)+P(멀쩡)P(양성|멀쩡)}$$

- 분모 오른편의 두 확률은 기존 값을 사용해 구할 수 있다. 테스트의 정확도 99%를 이용하면, 다음과 같은 테이블로 양성/음성과 병/멀쩡간의 관계표를 만들 수 있다. 100,000명을 가정해서 관계표를 만들면..

    |      | 병     | 멀쩡     | 소계      |
    |----- |----   |-------- |---------  |
    | 양성  | 99    | 999     | 1,098     |
    | 음성  | 1     | 98,901  | 98,902    |
    | 소계  | 100   | 99,900  | 100,000   |

    이를 이용하면 아래의 값이 맞는지 검증할 수 있다.

    $$P(멀쩡) = 1 - P(병), \ P(양성|멀쩡) = 1 - p(양성|병)$$

- 그래서 값을 다 집어넣으면 양성 결과가 나왔을 때 실제로 병에 걸렸을 확률은 9%가 된다.

    $$P(병|양성) = \frac{0.99 * 0.001}{0.001 * 0.99 + 0.999 * 0.01} \approx 0.09$$

- 여기서 병에 걸릴 확률을 사전 확률이라 함. 병에 걸릴 확률을 어떻게 알겠음. 신도 아니고. 그래서 전체 인구 중에 걸린 사람 수를 나눠서 적절히 구한 것임.
- 전체 인구 수를 기준으로 병에 걸릴 사전 확률은 0.1%에 불과했으나, 한번 양성이 뜨니까 9%로 병에 걸릴 확률이 올라갔음. 이를 사후 확률이라고 함.
- 베이즈 정리는 이와 같이 사전 믿음(병 걸릴 확률)을 새로운 정보(테스트 결과)를 사용해 새로운 사후 확률(테스트 결과를 보고 났더니 병에 걸릴 확률)로 업데이트하는 것임
- 이 의사 못 믿겠어서 다른 병원에서 테스트를 한번 더 봤는데 다시 양성이 떴다. 이럴 때 병에 걸렸을 확률은?
- 여기서 사전 확률은 이전 테스트를 통해 얻은 사후 확률이 된다. 병에 걸렸을 때 테스트가 양성일 확률이 99%로 동일하다고 하면  
    $$P(병|양성) = \frac{0.99 * 0.09}{0.09*0.99 + 0.91*0.01} \approx .91$$

- 새로운 사전 확률로 업데이트한 사후 확률은 91%. 미심쩍으면 두번 테스트해보면 되겠다.

<br>

### 2.2 나이브 베이즈에서의 베이즈 정리

나이브 베이즈에서는 베이즈 정리를 어떻게 사용하는 걸까? 

나이브 베이즈는, 분류 모델로 설명하자면, 인풋이 주어졌을 때 타깃 클래스들의 확률을 출력한다.

자주 등장하는 스팸메일 필터를 떠올리면, "점심", "학교"라는 단어가 나오면 정상 메일일 가능성이 높고, "다이어트", "클릭"이 나오면 스팸일 가능성이 높다. 수식으로 표현하면..

$$P(스팸|단어)$$

양성, 음성 2가지 보다 메일에서 나올 수 있는 단어의 수는 훨씬 많다. 
피쳐의 수가 이렇게 많아지면 이걸 바로 구할 수 없기 때문에, 베이즈 정리를 이용해서 변환한다.

$$P(스팸|단어) = \frac{P(단어|스팸)P(스팸)}{P(단어)}$$

현실 세계에서 베이즈 정리를 이용해 문제를 풀 때, evidence인 P(단어)는 보통 무시하고 분자만 계산한다.  
왜냐하면 분모인 P(단어)는 클래스 정보가 들어있지 않아, 클래스 입장에서는 상수나 마찬가지임. 그래서 분모만 따로 떼면..

$$P(스팸|단어) = P(단어|스팸)P(스팸)$$

여기에서 베이즈 정리를 한번 더 이용해서 변환한다. 이번에는 결합확률로 표현하는 베이즈 정리의 형태를 빌린다.

$$P(A|B) = \frac{P(A \cap B)}{\color{red}{P(B)}}$$

분모를 좌변으로 넘기면 

$$P(A|B){\color{red}{P(B)}} = P(A\cap B)$$

이걸 사용해서 조건부 확률을 결합확률로 변형할 수 있다.

$$P(단어|스팸)P(스팸) = P(단어 \cap 스팸)$$

단어가 총 3개 있다고 대충 가정하면 위 식은

$$P(단어1\cap단어2\cap단어3\cap스팸)$$

이걸 결합 확률의 체인 룰(chain rule)로 풀어쓰면..[https://en.wikipedia.org/wiki/Chain_rule_(probability)](https://en.wikipedia.org/wiki/Chain_rule_(probability))

$$P(단어1|단어2\cap단어3\cap스팸) \cdot P(단어2|단어3\cap스팸) \cdot P(단어3|스팸) \cdot P(스팸)$$

단어들이 서로 조건부 관계를 형성하고 있다. 엄청 복잡해보인다.

<br>

## 3. naïve independence assumption

`쟤는 참 사람이 나이브해`하면 만사를 조금 너무 쉽게 보고 대충대충 한다는 그런 느낌이 있다. 

나이브 베이즈는 뭘 대충 하길래 나이브라는 이름이 붙었을까.

바로 직전에 베이즈 정리를 이용해서 메일에 등장하는 단어들로 스팸을 예측하는 수식을 유도했다. 

$$P(단어1|단어2\cap단어3\cap스팸) \cdot P(단어2|단어3\cap스팸) \cdot P(단어3|스팸) \cdot P(스팸)$$

체인 룰로 결합법칙을 풀어버리면서 단어들간에 의존관계가 생겼다. 

저 수많은 given $\vert$들은 좀 과하게 복잡해 보이는데 정말 필요할까?

"다이어트에 딱 좋은 이 약을 구매하세요"와 "나 어제부터 다이어트하는데 개실패함"이라는 두 문장이 있을 때 둘 다 `다이어트`가 사용되었으나, 두번째는 스팸 냄새가 덜 난다. 근처의 단어들이 `다이어트`가 갖는 속성에 영향을 주기 때문.

따라서 위에 풀어쓴 수식처럼 단어간의 의존 관계를 반영한 모델을 만드는 것이 필요해 보인다.

그리고 나이브 베이즈는 그 의존관계를 깡그리 무시하고, 단어들은 서로 완전히 독립적이라는 다소 순수한, 즉 나이브한 가정을 베이즈 정리에 적용하기 때문에, 나이브 베이즈라는 이름이 붙은 것이다.

예를 들어

$$P(단어1|단어2\cap단어3\cap스팸) = P(단어1|스팸)$$

이렇게 조건 부분에서 나머지 피쳐를 날려버리고 클래스만 남겨둔다. 나머지에도 동일하게 적용하면, 최종적으로는 아래와 같이 수식이 간단해진다.

$$P(단어1|스팸)\cdot P(단어2|스팸) \cdot P(단어3|스팸) \cdot P(스팸)$$

수학 기호를 이용해서 다시 쓰면..

$$p(C_k|x_1, ..., x_n) = \frac{1}{Z}p(C_k)\prod_{i=1}^np(x_i | C_k)$$

Z는 evidence 상수고, 중간에 생긴 prod는 시그마의 곱 버전이다. 풀면 바로 위 수식으로 전개된다.

피쳐간의 관계를 독립적이라고 가정해버리는 나이브 베이즈의 선택은 일견 데이터의 특성을 온전히 반영하지 못하는 듯 하다. 그러나 실제로 스팸분류기 등의 분류 모델에서 나이브 베이즈가 어느정도 성능이 잘 나오는 것을 보면, (1)실제로 피쳐간 관계를 따질 만큼 문제가 복잡하지 않거나 (특정 단어의 등장만으로도 판단할 수 있거나) 혹은 독립성 가정으로 파라미터가 적은 모델이 상대적으로 노이즈에 더 강하기 때문이지 않을까.   

더 자세한 것은 [https://nlp.stanford.edu/IR-book/html/htmledition/properties-of-naive-bayes-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/properties-of-naive-bayes-1.html) 보자.

나이브 베이즈 끝!

# Reference
- wikipedia
- https://nlp.stanford.edu/IR-book/html/htmledition/properties-of-naive-bayes-1.html

