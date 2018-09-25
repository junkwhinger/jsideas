---
layout: post
title:  "coursera_ML: logistic regression"
date:   2016-01-03 23:34:25
tags: [octave, deep learning]
---

(주의 - 공부하면서 까먹지 않기 위한 개인적인 생각 및 정리이므로 수리적, 이론적으로 정확하지 않을 가능성이 있습니다.)

## 선형회귀가 못하는 것
Coursera 1주차에서는 선형회귀를 다루었다. 2주차에서는 로직스틱 회귀를 다루는데, 1주차에서 은근슬쩍 넘어가는 모양새가 재미있다. 선형회귀를 통해서 나는 변수가 주어졌을때 그것을 바탕으로 y값을 예측하는 방식을 배웠다. 로지스틱 회귀는 연속적인 y값을 예측하는 것이 아닌 output이 어떤 category에 속하느냐를 따지는 classification에 쓰인다. 이름이 선형회귀와 비슷해서 헷갈리기가 쉽다.

여튼 Andrew Ng 교수는 다음과 같은 예를 들면서 선형회귀가 긁지 못하는 가려운 곳을 로지스틱 회귀로 긁을 수 있다고 한다. 예를 들어.. 종양의 크기와 양성/악성 여부가 아래와 같이 분포한다고 하면, 선형회귀식을 구성하여 0.5보다 크면 악성, 0.5보다 작으면 양성이라고 판정할 수 있을거다.

![선형회귀식으로 판별하기](/assets/materials/20160104/1.png)

하지만 엄청나게 큰 악성 종양이 데이터셋으로 들어오는 순간 만들어진 선형식은 기울기가 떨어지게 되고 결국 정확한 판별을 할 수 없게 된다. 이때는 선형회귀보다는 sigmoid function을 사용하는 것이 바람직하다.

![선형회귀식으로 판별하기가 잘 안된다](/assets/materials/20160104/2.png)

## Sigmoid Function
시그모이드 함수는 g(z)로 표기하는데, 이는 곧 1 / (1+ e^-z)다. e는 지수를 의미한다. 이를 그래프상에 옮기면 아래와 같다.

![sigmoid function](/assets/materials/20160104/3.png)

z를 x축에, g(z)를 y축에 두면, z가 0일때 g(z)는 0.5다. z가 커지면 g(z)는 1에 가까위지고, 그 반대는 0에 가까워진다. 값이 계속적으로 상승하거나 감소하는 선형회귀와는 달리 아무리 z가 커지고 작아지더라도 g(z)는 0과 1 사이에 위치한다. 즉, 앞선 종양 사례를 다시 생각해본다면, 종양인지 아닌지를 판별하는 기준으로 sigmoid함수가 써먹기 좋다는 생각이 든다. 여튼 g(z)라는 시그모이드 함수가 있다면 z가 0보다 크면 결과값이 1이게 된다. 만약 h_theta(x) = g(5-x)라고 한다면, x가 5보다 작은 영역이 y가 1인 영역이 되고, x=5가 클래시피케이션을 결정하는 decision boundary가 된다. (문제는 g(5-x) 처럼 쉽게 수식이 떨어지지 않는다는 것...)

## cost function
로지스틱 회귀식은 어떻게 구성해야 될까. 선형회귀와 마찬가지로 결국 결과값인 y와 x변수를 기반으로 예측한 h_theta(x)와의 차이인 cost를 계산해야 한다. 마찬가지로 J(theta)가 사용되는데 문제가 하나 있다. J(theta)가 최소값이 되는 지점을 찾기 위해서는 J(theta) 곡선이 convex(볼록한) 형태여야 하는데, 1차 방정식인 선형회귀와는 달리 로지스틱 회귀에서 사용하는 h_theta(x)는 선형식이 아니므로, 다른 cost function을 찾아야 한다.

![cost function](/assets/materials/20160104/4.png)

로지스틱 회귀분석을 위한 cost function은 다음과 같이 구성된다.

연속적인 값을 예측하는 선형회귀와는 달리, 로지스틱회귀는 기냐 아니냐의 문제다. 즉 1이어야 하는데 0이거나, 0이어야 하는데 1이라고 값을 뱉는다면 잘못되었다는 피드백을 주어야 한다. 이를 수학적으로 표현하기 위해 log를 사용하는데 그 발상이 참 재밌다. y축, x축을 기준으로 뒤집어서 표현을 하는데, y가 1일때 h_theta(x)가 1이라면 cost는 0이지만, 0이라면 cost는 무한대로 늘어난다. y가 0일때는 바로 그 반대가 적용된다. (여기서 또 고등학교때 로그함수를 잘 배워놓아야할 동기가 생긴다..)

![penalising wrong classifications](/assets/materials/20160104/5.png)

여튼 이제 이 녀석을 하나의 수식으로 깔끔하게 만들어본다. y가 1이거나 0이라는 점을 적절히 활용하면 되는데 이것도 참 기발하다는 생각이 든다.

![cost, J(theta), grad](/assets/materials/20160104/6.png)

## gradient descent
선형회귀와 마찬가지로 theta값을 찾기 위해 gradient descent를 적용하기로 한다. cost function도 좀 달라졌으니까 gradient descent도 좀 다르지 않을까 싶었는데 사실 선형회귀와 같다. 결국 여기서 바뀐 것은 h_theta(x)를 무엇으로 정의하느냐의 문제이므로 결국 h_theta(x) = theta_0 * x_0 + theta_1 * x_1 이 1 / (1+ e^-theta’*x)로 바뀐 것 뿐이다. octave 상에서 바꿔주면 된다.

하지만 gradient descent 말고도 다른 더 발달된(advanced) 알고리즘을 적용할 수 있다. conjugate descent, BFGS, L-BFGS가 예시로 주어진다. 그냥 gradient descent와는 달리 이러한 알고리즘들은 알파값(러닝 레이트)를 일일히 지정할 필요가 없으며, 종종 더 빠르다고 한다. 그러나 역시 advanced의 단점을 조금 더 복잡하다는 것.

## one vs. all

바이너리 클래시피케이션, 즉 양성이냐 악성이냐하는 문제는 1과 0으로 넣어서 풀었다. 그런데 판별해야하는 값이 여러개인 multi-class classification은 어떻게 풀어야 할까? 로지스틱회귀를 사용해서 하려면 각각의 그룹을 나머지 그룹 모두에 대해서 판별하는 one vs. all 방식을 사용해서 분류해야 한다. 만약 분류해야 할 범주가 3개라면, 1번 그룹 vs. 나머지, 2번 그룹 vs. 나머지, 3번 그룹 vs. 나머지 이런식으로 3번을 돌려야 하는 것. 그래서 각각의 h_theta(x)를 구한 후, 가장 신뢰도가 높은 (즉 값이 높은) 범주를 선택하면 된다.


추가 - 2016.01.10
## overfitting
인공신경망 과제를 하려다가 갑자기 튀어나온 regularization을 보고 그제서야 logistic regression 정리에서 regularization을 빼먹었다는 사실을 알게되었다. 한주에 하나씩 정리글을 쓰려했는데 완전 실패. 여튼 마지막으로 부연하자면 다음과 같다.

먼저 logistic regression은 정리하자면 non-linear classification으로, 단순한 선형회귀선이 구분하지 못하는 구분을 가능하게 한다. 그런데 이녀석의 단점은 overfitting, 즉 트레이닝 데이터셋을 너무 잘 학습한 나머지 실제 예측은 잘 하지 못한다는 것이다. 예를 들어 시험의 성향을 이해하지 않고 족보만 달달 외웠다가 꼬아서 낸 문제는 모조리 틀리는 것. 반대로 underfitting은 주어진 족보를 제대로 학습하지 못한 것이라 하겠다.

여튼간에 logistic regression의 변수도 polynomial (2차 이상)이나 수없이 많은 변수를 집어넣게 되는 경우에 overfitting이 발생할 수 있어 이를 차단해야 한다.

![underfitting & overfitting](/assets/materials/20160104/7.png)

여기서 underfitting은 high bias, overfitting은 high variance라고 표현한다. <a href='https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff'>Wikipedia</a>에 따르면 bias는 알고리즘의 잘못된 가정에서 발생하는 오류를 의미하는데, 결국 bias가 높을수록 예측을 잘못한 것이므로 underfitting에 해당된다. high variance는 언뜻보면 underfitting인듯 한데 사실은 그 반대다. variance는 작은 변화에 대한 민감도에서 발생하는 에러를 의미하는데 이건 바로 직관적으로 이해하기가 어려웠다. 더 내용을 살펴보니 이런식으로 이해할 수 있을 듯 하다. 학습을 시도할 때마다 fitting을 위한 선을 계속 그리는데 이 선들간에 차이가 많으면 많을수록, 즉 variance가 높으면 높을수록 무슨 선을 믿고 따라가야할지 모르는 상황이 온다. 즉 어떤 일반화된 결과물이 나오지 않으므로 향후 새로운 데이터값이 들어왔을때 정확한 예측을 할 수 없게 된다.

그럼 어떻게 이를 처리할 것이냐. 결과적으로 더 간단한 모델을 만들면된다. 모든 트레이닝 데이터셋을 만족하지 않지만 어느 정도 신뢰성있게 맞추면서 일반화된 규칙을 뽑을 수 있는 그런 모델. 그걸 만드는 방법은 2가지가 있다. 하나는 피쳐의 갯수 자체를 줄이는 것. 100개의 피쳐를 다 넣지 않고 일부만 넣는 것인데, 한가지는 분석자가 직접 걸러내는 방식이 있고 다른 하나는 자동적으로 걸러내는 방식이 있다. 두번째는 여기서 다루는 regularization이다.

## regularization
regularization은 뭐냐. 피쳐 갯수를 줄이지 않고도 과적합의 오류를 피하는 방식이다. 피쳐 갯수를 안줄이면 어떻게 하냐? 피쳐를 유명무실하게 만들어버리면 된다. 즉 0에 가깝게 만들어서 계산에 들어가지 않도록 하는 방식. 그럼 어떤 애들을 0으로 만들어주냐? 그건 lambda 라는 친구가 결정한다. (근데 lambda값은 누가 결정하냐? 보니까 일단 적당한 값으로 때려넣는듯..)

![regularization](/assets/materials/20160104/8.png)

위의 수식을 보면 regularization이 들어가면서 뒤에 lambda가 붙은게 추가되었다. 코스트펑션인 J(theta)에는 lambda/2m*sigma(theta_j^2)가 붙고 gradient descent도 뒤에 lambda부분을 미분한 녀석이 붙었다. 여기서 lambda값을 모든 피쳐에 대해 곱해주면서 어떤 피쳐를 살리고 죽일지 결정하게 된다. lambda값을 너무 크게 넣으면 과적합을 너무 해소한 나머지 underfitting이 되어버리고 너무 작게 줄수록 overfitting에 가까워진다. 또 한가지 중요한 점은, regularization은 반드시 theta0에는 적용하지 않는다는 것. theta0에 적용하면 예측값이 달라진다고 하니 쓰지 말도록 하자. 예를 들어 아래 octave에서 돌린 결과물을 보자.

![lambda=0](/assets/materials/20160104/9.png)

lambda가 0일때, 즉 regularization이 들어가지 않은 결과다. decision boundary가 거의 정확하게 들어갔다. 하지만 위에 꼬랑지를 보면 알수 있듯이 너무 정확하게 맞췄다. 이를 방지하기 위해 lambda를 1로 올려보자.

![lambda=1](/assets/materials/20160104/10.png)

lambda가 1일때는 우리가 원했던(?) 원형으로 boundary가 잡혔다. 바로 전보다는 덜 정확하지만 일반화할 수 있는 규칙이 나왔다. 혹시 lambda를 더 올려보면 어떻게 될까.

![lambda=10](/assets/materials/20160104/11.png)

lambda가 10일때. 원형은 그대로이지만 뭔가 좀 덜 똑똑해졌다. 이번에는 100으로 올려보자.

![lambda=100](/assets/materials/20160104/12.png)

중앙타겟을 점점 더 빗나가고 있다. 너무 많은 피쳐를 없애버린나머지 정확도가 떨어지고 있다.


다음은 무엇일까? 선형회귀에서 은근슬쩍 넘어왔듯, 로지스틱회귀에서도 인공신경망(뉴럴넷)으로 은근슬쩍 넘어간다.