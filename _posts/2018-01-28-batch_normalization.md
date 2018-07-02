---
layout:     post
title:      "Batch Normalization"
date:       2018-01-28 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/batch_normalization/header.png
---


<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


## Introduction

Batch Normalization(이하 BN)은 뭘까. BN은 무엇이길래 ResNet이나 DCGAN 등 다양한 딥러닝 모델에 광범위하게 사용되는걸까? 궁금해진 김에 논문과 Andrew Ng 교수의 강의를 찾아보고 Keras로 간단히 테스트해보았다.

## 논문과 강의 요약
2015년 나온 <a href="https://arxiv.org/pdf/1502.03167.pdf">Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</a> 논문과 <a href="https://www.youtube.com/watch?v=em6dfRxYkYU">Andrew Ng 교수의 deeplearning.ai 강의</a> 일부를 참조했다.


### Normalizing inputs to speed up training
뉴럴넷이나 머신러닝 모델을 학습시킬때 보통 입력값을 0~1 사이로, 혹은 평균과 표준편차를 사용해서 정규화(Normalize)한다. 왜 데이터를 정규화하는걸까? <a href="https://www.quora.com/Why-do-we-normalize-the-data">Quora</a>에 소개된 내용을 요약하자면 다음과 같은 효과가 있다.

1. 모델 학습이 피쳐의 스케일에 너무 민감해지는 것을 막는다.
2. Regularization이 스케일의 차이에 다르게 적용된다.
3. 모델간 결과 비교가 용이하다.
4. 모델 최적화에 용이하다.

링크에 4번 내용에 대한 부연이 붙어있는데, 실제로 loss 함수의 모양을 타원이 아닌 원으로 만들면 더 빨리 converge하는지 실험을 통해 간단히 살펴본다. (이려러고 한건 아닌데 궁금하니까.) '밑바닥부터 시작하는 딥러닝' 챕터 6에 첨부된 샘플코드를 수정해서 실험해보았다.

두가지 loss 함수를 준비한다.
1. $$ h(x) = \frac{x^2}{20} + y^2$$
2. $$ h(x) = x^2 + y^2$$

h(x)를 최소화시키는 x와 y는 1과 2 모두 (0, 0)으로 동일하나, 1은 x에 붙은 분모 50으로 인해 x가 loss에 미치는 영향은 y에 비해 크게 줄어든다. loss 함수의 타원이 가로로 길쭉해지기 때문에, 최적점 (0, 0)에 도달하기 위해서는 가로로 먼 길을 가야 한다. 

실제로 최적점 수렴에 걸리는 시간도 차이가 날까? 두 함수 모두 동일한 Stochastic Gradient Descent 함수를 적용하고 x와 y가 모두 0.01보다 낮아질때까지 시간이 얼마나 걸리는지 측정해보았다. (속도를 빠르게 하기 위해 learning rate은 0.9로 지정한다.)

![convergence](/assets/batch_normalization/normalization_test.png)

x와 y의 스케일이 달랐던 1번 함수는 최적점 도달까지 27초가 걸린 반면, 스케일이 같은 2번 함수는 9초만에 학습을 완료했다.


### Normalizing hidden layer inputs

앞서 간략한 실험을 통해 인풋의 스케일을 정규화시키는 것이 gradient descent를 얼마나 빠르게 만드는지 확인해보았다. 그런데 인풋만이 아니라 다층 신경망의 중간 중간에 위치한 히든 레이어의 인풋도 정규화시킨다면 딥 뉴럴넷의 학습도 더 효율적으로 할 수 있지 않을까? 하는 생각이 Batch Normalization과 맞닿아있다. 단, Batch Normalization에서는 직전 레이어의 출력값 a가 아닌, relu activation 직전의 z에 대해서 Batch Normalization을 수행한다. 

논문에서는 여러 실험을 통해 BN을 사용한 아키텍쳐가 그렇지 않은 모델에 비해 더 안정적이고 빠르게 학습함을 증명했다. 왜 그런걸까?


### Covariate Shift
딥러닝이나 머신러닝 모델은 학습(training) 데이터를 기반으로 학습힌다. 고양이와 개를 분류하는 모델을 예로 들자. 모종의 이유로 학습에 사용한 고양이 데이터가 모두 `러시안 블루` 종이었다고 가정하자. `페르시안` 고양이 이미지를 이 모델에 넣으면 모델은 어떤 대답을 내놓을까? 형태를 보고 고양이라 판단할 수도 있지만, 털 색상으로 보아 개라고 판단할 수 도 있을 것이다.

![CAT vs. DOG classifier](/assets/batch_normalization/cat_dog_classifier.png)
이미지 출처: Google

이처럼 트레이닝에 사용한 학습 데이터와 인퍼런스에 사용한 테스트 데이터의 차이가 생기는 것을 `Covariate Shift`라 한다. 도메인 어댑테이션(<a href="http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html">링크</a>)에서는 다음과 같이 설명한다. 

> Another assumption one can make about the connection between the source and the target domains is that given the same observation $$ X = x $$, the conditional distributions of Y are the same in the two domains. However, the marginal distributions of $$ X $$ may be different in the source and the target domains. 
> Formally, we assume that 
> $$P_{s}(Y|X=x) = P_{t}(Y|X=x)$$
> for all $$ x \in X $$ , but $$P_{s}(X) \neq P_{t}(X)$$.
> This difference between the two domains is called <strong>covariate shift</strong> (Shimodaira, 2000).


### Internal Covariate Shift

논문에서는 Covariate Shift 문제를 레이어 레벨로 확장한다. 2개의 히든 레이어($$F_1, F_2$$)를 통해 이미지를 처리하는 뉴럴 네트워크를 생각해보자. 인풋 u가 네트워크를 통과하는 걸 식으로 간단히 표현하면 아래와 같다.

$$y_{hat} = F_{2}(f_{1}(u))$$

$$F_2$$는 두번째 히든 레이어이지만, 이 레이어 역시 인풋과 아웃풋을 갖는 하나의 작은 네트워크라고 볼 수 있다. $$F_{1}(u)$$를 $$x$$라 둔다면 아래와 같이 다시 쓸 수 있다.

$$y_{hat} = F_{2}(x)$$

위에서 실험을 통해 인풋을 정규화시키는 것이 더 빠른 gradient descent를 확인했다. 이를 적용하면, $$F_2$$ 네트워크도 입력값을 정규화시키면 더 효율적인 학습을 할 수 있다. 그런데 문제는 $$x = F_{1}(u)$$다. 뉴럴넷이 학습하면서 backpropagation을 통해 $$F_1$$ 레이어의 weight와 bias의 값이 업데이트된다. 그리고 업데이트된 값은 같은 u가 들어오더라도 다른 값을 리턴하게 된다.

또한 미니배치 역시 입력값을 다르게 하는 요소가 된다. 뉴럴넷을 학습할때 모든 데이터를 한번에 모델에 넣어 gradient를 계산하기보다는 더 효율적으로 그 추정량을 구할 수 있는 미니배치를 쓴다. 즉, 첫번째 배치를 통해 수정된 weight와 bias는 분포가 다를 수 있는 두번째 배치를 가공해 뒷 레이어에 전달하게 되므로, 운나쁘게 배치간 데이터 분포가 매우 다른 경우 레이어가 일관된 학습을 하기 어려워지게 된다. 

이를 논문에서는 Internal Covariate Shift라 명명한다.

> ... the notion of covariate shift can be extended beyond the learning system as a whole, to apply to its parts, such as a sub-network of a layer. 

> We refer to the change in the distributions of internal nodes of a deep network, in the course of training, as <strong>Internal Covariate Shift</strong>.


## Batch Normalization in Action

앞에서는 레이어를 기준으로 입력 데이터를 정규화하는 것을 Batch Normalization이라고 했으나, 실제로는 직전 레이어의 activation의 입력값에 BN을 적용한다.

![BN architecture](/assets/batch_normalization/bn_architecture.png)

입력값($$x$$)에 대해서는 다음과 같은 방식으로 정규화가 이루어진다. $$x^{(k)}$$는 x의 k번째 차원이다.

1. 배치($$X$$)를 그 평균과 분산을 이용하여 정규화한다.

$$\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$$

실제 구현시에는 분모에 아주 작은 상수 e를 더해 분모가 0이 되는 것을 방지한다.

2. $$\hat{x}$$에 학습가능한 파라미터인 $$\gamma$$를 곱하고 $$\beta$$를 더한다.

$$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

평균과 분산을 이용한 정규화 외에 별도의 파라미터를 사용하는 걸까? $$\gamma$$와 $$\beta$$없이 정규화만 하는 BN을 통해 거친 데이터가 sigmoid activation에 전달된다고 생각해보자.

![sigmoid function + normalized input](/assets/batch_normalization/sigmoid_normal.png)

-5부터 5까지 데이터(x)를 생성한다음, 이를 sigmoid함수에 넣은 결과(y)를 산포도로 뿌리면 위 그래프의 붉은 선을 형성한다. 그리고 오차 역전파시 y의 x에 대한 도함수를 구하면 녹색 분포를 그린다. 원 함수의 기울기가 0인 양 극단은 역전파할때 전달하는 정보가 소실된다. 재밌는 점은 x를 평균과 분산을 이용해 정규하고 그 분포를 뿌리면 아래 푸른색 히스토그램과 같은 모양이 그려진다.

> Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. 

단순하게 정규화한 값을 넣어 sigmoid activation에 전달하면, 그 값들은 그림과 같이 언제나 sigmoid 함수의 중간(선형적인 부분)에 위치하게 되므로 그 출력값과 gradient가 해당 구간에만 한정된다. 

고양이 얼굴을 학습한 어떤 필터를 생각해보자. 이미지에서 고양이가 있는 부분에서는 필터의 출력을 강하게 전달하고, 없는 부분에서는 넘어가야 한다. 그런데 위 그림처럼 행렬 연산을 한 값이 평균이 0이고 분산이 1인 값으로 고정되면, 가장 옵티멀한 신호를 전달한다고 볼 수 없다.

정규화는 입력값의 분포를 규격화시킨다. 만약 규격화가 잘못된 것이라면 어떻게 해야 할까? 규격화를 다시 원래대로 되돌려 원본 입력값을 그대로 출력으로 내보내면 되지 않을까? 만약 그대로 내보내는 identity function이 모델이 학습해야 할 옵티멀이라면 우리는 어떤 파라미터를 사용해서 BN을 수정하면 될 것이다. 여기서 사용하는 파라미터가 $$\gamma$$와 $$\beta$$다.

식 2를 다시 보면..

$$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

학습을 통해 $$\gamma^{(k)} = \sqrt{Var[x^{(k)}]}$$로, $$\beta^{(k)} = E[x^{(k)}]$$로 파라미터가 설정되었다고 하자. 이를 식 1과 함께 다시 써보면..


$$y^{(k)} = \sqrt{Var[x^{(k)}]} \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}} + E[x^{(k)}]$$

식 1의 분모와, 분자에서 빼준 평균이 $$\gamma$$와 $$\beta$$에 의해 상쇄되면서 결국 $$y^{(k)} = x^{(k)}$$가 되어 BN의 입력값이 그대로 출력값으로 넘어간다.


## Inference with Batch Normalization

BN을 통해 학습시 발생하는 Internal Covariate Shift 문제를 해결할 수 있다는 것은 확인했다. 그런데 이상한 점이 하나 있다. BN은 모델에 들어온 데이터 배치별로 정규화를 하는 개념인데, 그렇다면 학습이 끝난 후 테스트 데이터에 대한 인퍼런스는 어떻게 수행하는걸까? 또 인퍼런스하는 데이터의 수가 1개라면? 여러개라면 학습할 때 처럼 평균과 분산이라도 구할텐데 1개라면 답이 없다.

테스트 데이터를 처리할때 BN은 학습시 배치별로 뽑았던 평균과 분산을 평균내거나 이동평균(moving average)하는 식으로 계산한다음, 테스트 데이터에 적용하는 방식으로 사용한다. 

이러한 이유로 Keras에서는 ResNet50처럼 BN을 사용하는 아키텍쳐를 불러와 인퍼런스를 할때 K.set_learning_phase(False)를 지정하지 않으면 BN의 모드가 정해지지 않기 때문에 모델이 오류를 뱉는다.

tf.keras.layers에 구현되어있는 Batch Normalization 코드를 보면 아래와 같이 해당 코드가 training 상황에서 돌아가는지에 대한 정보를 처리한다. 

{% highlight python %}
def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()
    output = super(BatchNormalization, self).call(inputs, training=training)
    if training is K.learning_phase():
      output._uses_learning_phase = True  # pylint: disable=protected-access
    return output

{% endhighlight%}

## BN in Action!

설명이 매우 장황하고 중간중간에 삼천포로 좀 빠졌지만, 결론은 이렇다. BN을 사용하면 학습시 모델에 들어오는 인풋을 안정시켜 각 레이어가 더 안정적으로 학습할 수 있게 된다. (입력값 정규화로 인한 부수적인 regularization 효과도 있다.) 그렇다면 정말 효과가 있는지 간략한 실험을 통해 확인해보자.

cifar10 데이터를 사용해서 다음과 같은 2가지 모델을 만들었다.

![test architecture](/assets/batch_normalization/test_architecture.png)

CONV 레이어 뒤에 들어간 Batch Normalization 레이어말고는 모든 레이어의 규격과 순서가 같다. 

배치 사이즈를 32, epoch을 30으로 설정하고 두 모델의 training, validation 에러가 어떻게 달라지는지 Tensorboard를 통해 살펴보자.

![test result: default-Blue, BN-Red](/assets/batch_normalization/test_result.png)

### Training accuracy & loss
두 모델 모두 training accuracy는 비슷한 수준으로 수렴했으나, BN 모델(붉은 선)이 같은 시점에서 더 높은 accuracy를 찍었으며 더 빠르게 loss를 낮췄다. 

### Validation accuracy & loss
Training accuracy는 비슷하게 수렴한데 반해 Validation Accuacy는 BN 모델이 기존 모델에 비해 약 4% 정도 높은 수치를 보였다. loss 곡선이 특히 인상적인데, 기존 모델(푸른선)은 15th epoch 이후 validation loss가 되려 상승하면서 전형적인 오버피팅의 증상을 보였다. 이에 반해 BN 모델의 validation loss는 약간 상승한 수준에 그쳐, 논문에서 주장한 바와 같이 약간의 regularization 효과가 있음을 확인할 수 있었다.


## Reference  
<a href="https://arxiv.org/pdf/1502.03167.pdf">https://arxiv.org/pdf/1502.03167.pdf</a>  
<a href="http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html">http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html</a>  
<a href="https://www.quora.com/Why-do-we-normalize-the-data">https://www.quora.com/Why-do-we-normalize-the-data</a>  