---
layout:     post
title:      "DCGAN - 럭키짱 얼굴 생성 실험"
date:       2017-07-23 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/Learning_DCGAN_files/Header.png
---


## DCGAN

<a href="http://jsideas.net/python/2017/07/01/GAN.html">앞선 포스팅</a>에서 GAN(Generative Adversarial Network)의 구조에 대해 간단히 살펴보았다. 가장 기본적인 GAN은 Generator와 Discriminator에 densely-connected network가 사용된다. 예를 들어 28x28 사이즈인 MNIST 손글씨 이미지를 길게 이어붙여 784개의 숫자로 구성된 벡터로 만들고, 이를 그 다음 은닉층의 모든 노드와 연결시키는 방식이다. 이미지의 특성을 고려하지 않고도 꽤 괜찮은 결과가 나왔는데, 이미지 분류에 자주 쓰이는 CNN을 엮으면 어떻게 될까? GAN에 CNN을 더한 모델을 보통 DCGAN(Deep Convolutional Generative Adversarial Network)라 한다.

![gif](/assets/Learning_DCGAN_files/celeba.gif)

DCGAN을 사용하면 간단한 손글씨 이미지를 만드는 것을 넘어, 위와 같이 얼굴까지 만들 수 있게 된다. 200,000장이 넘는 <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">유명인의 얼굴 데이터셋 CelebA</a>을 사용해서 새로운 얼굴을 만들어본 것인데, 3,000 ~ 4,000번만 배치를 돌아도 꽤 그럴듯한 얼굴이 만들어진다. 학습 속도를 높이기 위해 이미지 사이즈를 28x28로 줄여놓고 학습을 하다보니 누구를 속일 수 있을 정도로 고화질 사진이 나오지는 않았지만, 그래도 개인이 20분 정도 ec2 위에서 돌려본 정도로는 괜찮은 수준이다.

![step 100](/assets/Learning_DCGAN_files/celeba_step100.png)
![step 5000](/assets/Learning_DCGAN_files/celeba_step5000.png)

<hr>

## 만화 얼굴도 생성할 수 있을까?

사람 얼굴이 꽤 그럴듯하게 생성된다면.. 사진이 아닌 만화의 얼굴도 생성할 수 있지 않을까? <s>(평점을)</s> 즐겨보는 웹툰 중 하나인 김성모 화백의 '돌아온 럭키짱'에 나오는 극실사체의 얼굴을 적당히 생성할 수 있다면 꽤 재밌는 드립을 칠 수 있을 것 같았다.

![돌아온 럭키짱](/assets/Learning_DCGAN_files/naver_lucky.png)

<hr>

## 데이터 확보 및 가공

전체 프로세스는 다음과 같았다.

+ 원본 웹툰 이미지 확보: 
  + <a href="https://gist.github.com/allieus/13c1a80ef5648c2b9b112e1c58f9727b">allieus님이 작성한 네이버 웹툰 크롤러</a>를 활용해 럭키짱의 회차별 이미지 데이터를 확보했다. 
  + 280화나 되는데다, 보안 등의 이슈로 가장 어려운 단계라고 생각했었는데 오히려 가장 쉽게 풀렸다.
+ 이미지에서 얼굴 추출:
  + openCV를 사용해 전체 이미지에서 얼굴을 추출했다.
  + 간단히 구글링해보니, 애니메이션에서 얼굴을 추출하는 사례가 몇개 있었다. <a href="https://gist.github.com/allieus/13c1a80ef5648c2b9b112e1c58f9727b">아이돌마스터</a>
  ![애니메이션에서 얼굴 추출하기](/assets/Learning_DCGAN_files/opencv_anime.png)
  + 여기에서 학습된 classifier를 럭키짱 이미지에 대고 돌렸으나, 전혀 얼굴을 찾지 못했다..
  + <a href="https://github.com/mrnugget/opencv-haar-classifier-training">Train your own OpenCV Haar classifier</a>를 참조해 럭키짱에 특화된 얼굴 추출기를 만들었다.
  + 얼추 작동하기는 했으나, 전체 이미지를 대상으로는 성능이 매우 좋지 않았다. 그래서 먼저, 웹툰 이미지에서 이미지가 들어있는 사각 박스를 추출한 다음, 추출한 박스 내에서 얼굴을 찾는 방식을 취했다. 
  + 최초 시도보다는 성능이 괜찮았지만, 놓치는 얼굴이 많아보였다. 탐지 성능을 최대한 높여 가능한 많은 이미지를 만든 후, 이 중 얼굴에 해당하는 것을 수동으로 걸러냈다. 약 7만장 중 2,700장 정도를 건졌다.
  ![운이 좋은 사례](/assets/Learning_DCGAN_files/face_detected.png) 
  + 재미있는 시도였지만, Haar classifier를 학습시키는데 로컬 머신에서 매우 오랜 시간이 걸렸다. 이틀은 꼬박 터미널을 켜두었다.
  + 이 부분에 대해서는 별도로 포스팅을 써보고자 한다.
+ DCGAN을 활용한 얼굴 생성:
  + CelebA에서 활용한 DCGAN을 튜닝하여 적용한다.

<hr>

## DCGAN 로직

GAN과 마찬가지로 DCGAN은 Generator와 Discriminator로 구성되지만, 이들이 보다 더 복잡한 방식으로 만들어진다. 

+ CNN - Convolutional Layer를 사용한다. 이미지 분류를 위해 Discriminator가 'tf.layers.conv2d'를 사용해 이미지에서 여러 피쳐를 뽑는다면, Generator는 tf.layers.conv2d_transpose를 써서 여러 피쳐를 이미지화한다. 
+ Batch Normalization - DCGAN에서 자주 쓰이는 테크닉으로, Discriminator의 입력층과 Generator의 출력층을 제외한 모든 레이어의 conv2d_transpose 다음에 쓰인다. 레이어에 들어온 인풋을 평균이 0이고 분산이 1인 값으로 정규화를 시켜버리는 것. 네트워크에 입력하는 값을 정규화하는 것이 아니라, 네트워크 내에서 각 레이어에 들어가는 인풋을 정규화시키는 개념이다. Batch Normalization은 DCGAN의 필수 구성 로직 중 하나로, internal covariate shift를 줄여 네트워크가 더 잘 학습하도록 돕는다고 한다. <a href="https://arxiv.org/pdf/1502.03167.pdf">관련 논문</a>
+ Dropout - <a href="https://github.com/soumith/ganhacks">How to Train a GAN</a>에서 찾은 팁 중 하나로, Generator의 레이어에 dropout(0.5)을 추가하여 학습/테스트를 한다. 
+ Label Smoothing - 진짜 데이터인 경우 Discriminator에 들어갈 값을 1이 아닌 0.9 정도로 노이즈를 섞어 학습한다. 오류를 일부 섞어넣으면 Generator 학습에 도움이 된다고 한다.

<hr>

## 실험 결과

확보한 데이터셋을 32x32 사이즈로 줄여서 DCGAN을 테스트해보았다. 32x32 크기로 다운사이즈하니 같은 크기의 다른 데이터셋에 비해 더 알아보기가 어려운 듯 하다. 얼굴에 그라데이션이 들어가 있지 않은데다 이목구비가 얇은 선으로 구성되다보니 그런 듯 하다.
![128x128](/assets/Learning_DCGAN_files/faces128.png) 
![32x32](/assets/Learning_DCGAN_files/faces32.png)

적용한 하이퍼파라미터는 다음과 같다.  

+ z_size = 100  
+ learning_rate = 0.0002  
+ batch_size = 256  
+ epochs = 200  
+ alpha = 0.2 (leaky ReLU)
+ beta1 = 0.9 (AdamOptimizer)

100 step마다 출력된 테스트 결과는 아래와 같다.

![gif](/assets/Learning_DCGAN_files/luckyZzang.gif)

CelebA 결과와는 다르게 학습 초기부터 얼굴형이 명확히 드러나지 않았으며, 꽤 많은 step이 지난 후에도 얼굴형만 간신히 보이는 수준이다. 

학습과정에서 기록한 Generator_loss와 Discriminator_loss로도 학습이 제대로 되지 않았음을 확인할 수 있었다.

![CelebA - 초기](/assets/Learning_DCGAN_files/celeba_early.png)
![CelebA - 말기](/assets/Learning_DCGAN_files/celeba_late.png)

CelebA는 처음에는 Generator Loss가 0.5에서 6까지 널을 뛰었으나, 말기에는 1 정도에 수렴하며 안정적인 결과를 놓았다. Discriminator Loss 역시 0.5 밑으로 계속 내려가지 않는 것으로 보아 Generator가 만들어낸 그럴듯한 가짜 이미지가 Discriminator를 잘 속였음을 알 수 있다.

이번에는 럭키짱 데이터를 보자.
![럭키짱 - 초기](/assets/Learning_DCGAN_files/lucky_early.png)
![럭키짱 - 말기](/assets/Learning_DCGAN_files/lucky_late.png)

초기에는 Discriminator와 Generator 모두 안정적인 Loss를 기록하지만, 말기로 갈수록 Generator는 수렴에 실패했다. (loss를 기록한 값을 ec2를 날리기 전에 저장을 못해 그래프가 없다). Generator는 200 step 이후부터 학습이 종료될때까지 2~4의 Loss를 기록했다. 만들어진 이미지의 퀄리티 더 나아지지 않는 탓인지 Discriminator의 Loss 역시 200 step 이후 줄곧 0.4 ~ 0.6 사이를 오갔다. 결론적으로 DCGAN을 사용한 럭키짱 얼굴 생성은 실패했다.

실패한 원인에는 여러가지가 있겠으나, 가장 큰 원인은 데이터에 있다고 본다. 배치 사이즈, learning rate와 같은 하이퍼파라미터를 조정하거나, 최초 이미지 사이즈를 늘이거나 줄여보는 등 여러 시도를 해보았으나, Generator가 수렴에 실패하는 것은 매한가지였다. 예제로 사용된 MNIST나 CelebA 데이터셋에 비해 럭키짱 데이터셋이 2,700장에 불과한 것이 실패의 근본적인 이유이지 않나 싶다. 존재하는 어떤 현상의 본질을 뽑고, 그 본질에 노이즈를 섞어 새로운 무언가를 만드는 것이 GAN의 기본 개념인데, 본질을 추출하기 위한 현상의 관찰 수가 적다면, 모델의 베이스가 되는 본질 자체가 불안정할 수 밖에 없다는 생각이 들었다. 그리고 애초에 딥러닝 모델 자체가 대규모 데이터를 가진 문제를 해결하려는 것이기도 하고.

어떻게든 얼굴 같아 보이는 결과를 뽑고 싶었는데, 지금 수준에서는 ec2 비용만 나갈 뿐 더 진전을 이루기는 어려워보여서 여기서 실험을 마친다.
