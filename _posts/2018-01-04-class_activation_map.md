---
layout:     post
title:      "CAM: 대선주자 얼굴 위치 추적기"
date:       2018-01-04 00:00:00
author:     "Jun"
img: 20180104.png
tags: [python, deep learning, image, weekly supervised learning]
---

작년에 올린 여러 포스팅을 통해 2017년 대통령 선거에 출마한 주요 후보 5인의 얼굴을 <a href="http://jsideas.net/python/2017/05/07/transfer_learning.html">분류하는 CNN 모델</a>을 만들고, 또 <a href="http://jsideas.net/python/2017/11/26/transfer_learning_with_keras_on_floydhub.html">여러 모델간의 성능을 비교</a>해보았다. 
그 결과 <strong>[얼굴 사진을 넣어 누구인지 맞추는]</strong> 문제를 CNN을 통해 해결할 수 있다는 사실은 확인할 수 있었다. 
그런데 왜 모델이 잘 동작하는지, 홍준표와 심상정은 어떻게 다르게 분류하는지에 대해서는 알 수 없었다. 내가 알 수 있는 건 각 클래스에 대해 모델이 출력하는 확률값 뿐이었다.

정확히 맞추기만 하면 장땡은 아니다. 직관적으로 이해할 수 없는 부분에 대해 설명을 요구하는 인간적인 욕구 충족을 넘어서는 문제다. 왜 동작하는지 이해해야 모델의 문제점을 보완하고, 더 잘 동작할 수 있는 또다른 문제에 모델을 적용할 수 있다. 
예컨대 과거 김성모작가의 <a href="http://jsideas.net/python/2017/07/23/DCGAN_luckyZzang_experiment.html">돌아온 럭키짱의 얼굴</a> 추출을 위해 <a href="https://github.com/nagadomi/lbpcascade_animeface">일본 만화로 학습된 얼굴 추출기</a>를 사용해본 적이 있다. 
학습 데이터와 테스트 데이터의 성질과 분포가 다르니 당연히 결과는 실패였다. 
내 모델이 특정 데이터를 대상으로 왜 잘 동작하지 않는지를 이해해야 모델의 사용 환경과 개선점을 찾을 수 있다.

![잘 안되는 이유](/assets/materials/20180104/luckyZZang_idol.png)


AlexNet 등장 이후 CNN이 대세로 굳어지면서 많은 연구자들이 '왜 잘 동작하는가'에 대한 질문에 답을 찾으려 노력하고 있다. 
그 중에 하나가 최근에 읽은 <a href="https://arxiv.org/abs/1512.04150">Learning Deep Features for Discriminative Localization(2015)</a>이다. 
기존에 많이 쓰는 pretrained network를 약간만 변형해서 놀라운 결과를 내는 `Class Activation Mapping`을 제안한다. 
논문에 나온 수식과 구현방식을 대선주자 데이터셋을 사용해 간략히 정리해보았다.

<hr>

## Learning Deep Features for Discriminative Localization

### 주요 내용 요약
- <strong>[핵심] 바운딩 박스를 인풋으로 주지 않아도, 오브젝트 디텍션 용도로 학습된 모델을 조금만 튜닝해서 오브젝트 위치 추적이 가능하다.</strong>
- 탐지 대상인 사물의 위치 정보가 없어도 CNN은 오브젝트 디텍터로서 기능한다.
- 그런데 분류를 위해 fully connected layer를 사용함으로써 이 기능이 사라진다.
- Network In Network나 GoogLeNet에서는 파라미터 수 최소화를 위해 fully connected 대신 `Global Averag Pooling`(이하 GAP)을 썼다.
- 그런데 이 GAP는 파라미터 수를 줄여 오버피팅을 방지하는 기능 외에도, 오브젝트의 위치 정보를 보존하는데 사용할 수 있다.
- GAP를 통해 특정 클래스에 반응하는 영역을 맵핑하는 `Class Activation Mapping`(이하 CAM)을 제안한다.
- 당시 제안된 다른 위치 추적 방식들에 비해 CAM은 한번에 (single forward pass) end-to-end로 학습할 수 있다.
- FC를 GAP로 대체해도 성능 저하가 크게 일어나지 않았으며 이를 보정하기 위한 아키텍쳐도 제안한다.
- Global Max Pooling은 탐지 사물을 포인트로 짚는 반면, GAP는 사물의 위치를 범위로 잡아내는 장점이 있다.
- 다른 데이터셋을 활용한 분류, 위치 특정, 컨셉 추출에도 쉽게 대입해서 사용할 수 있다.


### Class Activation Mapping
CNN은 보통 사람의 시각 처리와 연결지어 설명하는 경우가 많다. 
fully connected로만 연결된 DNN은 28x28 크기의 손글씨 이미지를 784차원으로 찌그러뜨린 다음, 각 픽셀값에 웨이트를 곱해 다음 레이어로 넘긴다. 자연히 이 과정에서 서로 인접한 픽셀간의 관계 정보가 소실된다.

이와 달리 CNN은 Convolution Filter를 사용해 인접 픽셀간의 정보를 그대로 뒤 레이에 넘긴다. 
그렇게 함으로써 처음에는 직선이나 곡선 정보를 얻고, 이를 조합해서 숫자 2나 5의 형상 정보를 얻는다. 

![형상을 서서히 학습하는 CNN](/assets/materials/20180104/cnn_features.png)
이미지 출처: https://stats.stackexchange.com/questions/146413/why-convolutional-neural-networks-belong-to-deep-learning/146477

이전에 만들어본 대선주자 얼굴 분류기도 마찬가지다. 
어떤 이미지가 들어왔을 때 그 사진의 주인공이 문재인인지 홍준표인지 안철수인지 꽤 높은 정확도로 모델이 분류했다면, 위 그림과 마찬가지로 그 얼굴을 알아보는 `필터`를 학습했다고 생각할 수 있다. 
그럼 그 필터가 활성화되는 위치를 역으로 추적한다면, 그 이미지에서 얼굴이 위치한 부분을 알아낼 수 있지 않겠느냐고 생각해볼 수 있다!

하지만 일반적인 CNN 분류 모델처럼 마지막 convolutional layer의 아웃풋을 flatten해서 fully connected에 넘기는 순간, 필터가 들고있는 정보가 사라진다. 
여기서 논문이 제안한 것은 flatten해서 fc를 쓰지 말고 Global Average Pooling을 쓰자는 거다.

![CNN - flatten](/assets/materials/20180104/flatten_architecture.png)

이미지 출처: https://it.mathworks.com/discovery/convolutional-neural-network.html

#### Global Average Pooling
Global Average Pooling은 위 그림에서 flatten이 일어나기 직전, 마지막 convolutional layer에 적용하는 방식으로, 각 피쳐맵의 평균값을 뽑아 벡터를 만든다. 
이해를 돕기 위해 마지막 conv layer의 피쳐맵 개수가 3개고 각각의 크기가 3x3이라고 해보자.

![GAP, GMP](/assets/materials/20180104/GAP_GMP.png)

위 예시에서 GAP는 각 피쳐맵에 대해 모든 값을 더하고, GMP는 모든 값 중 최대값을 골라 벡터를 만든다. 
사실 평균(Average)을 취하면 1+2+1을 한 후 9로 나눠줘야 맞겠지만, 논문 상에서는 합으로 처리되어있다. 
어차피 같은 레이어의 모든 피쳐맵은 x, y 갯수가 동일하므로 굳이 나눠주지 않아도 되는 것으로 이해했다. <a href="https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/keras/_impl/keras/layers/pooling.py">tf.keras 상에 구현된 코드</a>를 보면 `return K.mean(inputs, axis[1, 2])`로 평균을 구하는 것으로 구현되어있기는 하다.

#### Class Activation Map

논문에서 GAP를 사용해서 CAM을 도출하는 일련의 과정을 살펴보자. 
> For a given image, let $$f_{k}{(x, y)}$$ represent the activation of unit k in the last convolutional layer at spatial location (x, y). 

$$f_{k}{(x, y)}$$는 마지막 conv layer의 k번째 유닛의 액티베이션을 표현한다. 즉, 위 그림에서 첫번째 유닛인 붉은색 표가 $$f_1(x, y)$$가 된다.

> Then, for unit $$k$$, the result of performing global average pooling, $$F^k$$ is $$\sum_{x, y}f_{k}(x,y)$$.

k번째 유닛에 대해서 GAP를 씌운 값은 $$F^k$$가 된다. 즉, GAP를 통해 계산한 붉은색 값 4가 $$F^1$$이 된다. 

> Thus, for a given class $$c$$, the input to the softmax $$S_c$$ is $$\sum_{k}w^{c}_{k}F_{k}$$ where $$w^{c}_{k}$$ is the weight corresponding to class $$c$$ for unit $$k$$.

여기부터 조금 복잡해지는데, $$w^{c}_{k}$$를 $$F_{k}$$에 곱하고 이를 모두 더해 소프트맥스 레이어에 집어넣을 input인 $$S_c$$를 구한다. 우리가 예측할 클래스의 갯수가 3개라고 가정해보면 사실 다음과 같이 간단히 생각해볼 수 있다.

![$$S_c$$](/assets/materials/20180104/S_c.png)

클래스 1, 2, 3 중에서 우리가 관심있는 클래스가 1이라고 생각해보자. 그러면 우리의 소프트맥스 인풋은 $$S_1$$이 되고, 이를 산출하는 공식은 $$\sum_{k}w^{1}_{k}F_{k}$$이 된다. 
여기서 k는 1, 2, 3이므로 이 summation을 다시 풀어서 쓰면 $$w^{1}_{1}F_{1} + w^{1}_{2}F_{2} + w^{1}_{3}F_{3}$$이 되고, 위 그림에서 각 수식에 맞는 값을 끼워보면 $$2 * 4 + 1 * 3 + 0 * 1$$이 되어서 결국 $$S_1$$은 11이 된다. 

$$S_c$$를 구했다면 이제 이를 소프트맥스 공식에 넣어서 클래스 $$c$$의 분류 확률을 구하게 된다.

논문에서는 소프트맥스 직전까지의 공식을 변형해서 CAM을 구하는데 그 과정이 흥미롭다. $$S_c$$를 구하는 공식을 다시 써보면..

$$S_c = \sum_{k}w^{c}_{k}F_{k}$$

$$F_k$$가 $$\sum_{x, y}f_{k}(x,y)$$이므로,

$$S_c = \sum_{k}w^{c}_{k}\sum_{x, y}f_{k}(x,y)$$

두 시그마의 곱은 다음과 같이 <a href="http://functions.wolfram.com/GeneralIdentities/12/">변형</a>할 수 있다.

$$\sum^{n}_{k=0}a_{k}\sum^{n}_{j=0}b_{j} == \sum^{n}_{k=0}\sum^{n}_{j=0}a_{k}b_{j}$$

그러므로 위 공식을 다시 쓰면 (시그마 순서를 바꿨다)

$$S_c = \sum_{x, y}\sum_{k}w^{c}_{k}f_{k}(x,y)$$

여기서 우리가 구하고자하는 Class Activation Map인 $$M_{c}(x, y)$$를 2번째 시그마부터 떼어서 정의한다.

$$M_{c}(x, y) = \sum_{k}w^{c}_{k}f_{k}(x,y)$$

$$S_c = \sum_{x, y}M_{c}(x, y)$$


#### Imprementing CAM
구현은 수식보다는 더 간단해보인다. 수식을 말로 풀어보자면, 맞추고자 하는 클래스가 1이라고 가정했을때 weight matrix의 1번째 횡벡터와 피쳐맵 $$f_{k}(x, y)$$를 곱하면 된다. 
그래서 $$w^1$$ 횡백터의 인덱스 $$k$$를 돌면서 $$k$$번째 weight element를 $$k$$번째 피쳐맵 매트릭스에 곱한다음 이를 다 더해주기만 하면 된다. 

구현상 편의를 위해서 먼저 그 마지막 conv layer의 가로세로 크기만큼의 numpy 매트릭스를 만들고 cam이라는 변수에 할당한다.

weight matrix의 횡벡터는 class_weights에, 피쳐맵은 conv_outputs라는 변수에 할당한다.

그리고 class_weights로 선언된 횡벡터를 loop로 돌면서 해당 weight element인 w를 conv_outputs[:, :, k]에 곱해주고 이를 cam에 계속 더해주면 수식대로 구현이 된다. 
([:, :, k]인 이유는 3번째 차원이 채널이기 때문이다. 채널이 0번째 인덱스인 경우 [k, :, :]이 된다.)

{% highlight python %}

cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
for k, w in enumerate(class_weights[:, class_idx]):
    cam += w * conv_outputs[:, :, k]

{% endhighlight %}

그래서 클래스 1에 해당하는 CAM을 구하려면 class_idx를 1로 지정하고 위 코드를 실행하면 된다. 

이미지 경로와 class_idx를 받아 CAM을 추출하는 함수를 다음과 같이 구현해보았다.

{% highlight python %}
def generate_cam(img_path, class_idx, tl_model):  
    """
    parameter:
    ----------  
    img_path(string): image path
    class_idx(integer): class index for class activation map
    tl_model: network to perform inference

    return:
    -------
    img_arr : numpy array of the given image
    cam : numpy array of class activation map
    predictions : numpy array of classification result
    """

	## read image and preprocess it
    img = pil_image.open(img_path).resize((224, 224))
    img_arr = np.asarray(img)[:, :, :3] / 255.
    img_array = np.expand_dims(img_arr, 0)
    
    ## get prediction result and conv_output values
    get_output = K.function([tl_model.layers[0].input], [tl_model.layers[-3].output, tl_model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img_array])
    conv_outputs = conv_outputs[0, :, :, :]
    class_weights = tl_model.layers[-1].get_weights()[0]
    
    ## calculate cam
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, class_idx]):
        cam += w * conv_outputs[:, :, i]

    ## normalise cam and resize to fit the orginal image size
    ## in this case (224, 224)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (224, 224))
    
    return img_arr, cam, predictions

{% endhighlight %}

#### finding bounding boxes

논문에서는 추출한 CAM으로 로컬라이징을 수행한 후, 이를 실제 바운딩 박스의 영역과 비교해 얼마나 정확하게 맞추었는지 평가하였다. 
논문에서는 심플한 thresholding 기법을 사용했다. 
먼저 CAM의 최대값의 20% 이상인 값만 걸러 활성화된 여러 덩어리를 남긴다. 이 덩어리 중 가장 큰 덩어리를 커버할 수 있는 바운딩 박스를 만든다.

논문에서는 가장 큰 덩어리만을 남겼으나, 나는 모델이 어디에 반응했는지를 모두 판단하고 싶어 일단 반응한 모든 덩어리에 바운딩 박스를 씌우는 쪽으로 구현했다. 
skimage.measure의 regionprops를 사용하면 바운딩 박스 영역을 아주 편리하게 뽑을 수 있다. 
추출한 props에 덩어리 갯수만큼의 박스가 떨어지므로, 후에 matplotlib을 사용해서 이미지 위에 박스를 for loop으로 그리는 방식으로 구현했다.

```python
def generate_bbox(img, cam, threshold):
    labeled, nr_objects = ndimage.label(cam > threshold)
    props = regionprops(labeled)
    return props

## ...

for b in props:
    bbox = b.bbox
    xs = bbox[1]
    ys = bbox[0]
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    rect = patches.Rectangle((xs, ys), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
```

<hr>

### Model Architecture
예전에 작업한 대선주자 분류모델에서도 GAP를 사용해봤지만, 논문을 보면서 그대로 구현해보지는 않았었다. 
이번에는 논문에서 표현한대로 모델을 수정해보았다.

#### VGG16
논문에서 사용한 3가지 모델 중 VGG를 써보았다. 구체적으로 16을 썼는지 19를 썼는지 명시해두지 않았는데, 모델 아키텍쳐를 보면 VGG16을 쓴 것으로 보인다.

> For VGGnet, we removed the layers after conv5-3 (i.e., pool5 to prob) resulting in a mapping resolution of 14 x 14.

근데 어차피 16이나 19나 conv5-3까지의 구조가 동일하므로 둘다 써도 무방하다.

VGG16의 원래 구조를 찍어보면, `block5_conv3`의 컨볼루션 레이어의 Output Shape이 14 x 14임을 확인할 수 있다. 
처음에 input shape를 224, 224로 지정하지 않으면 conv5-3의 사이즈가 14 x 14로 떨어지지 않으므로 주의하자.
```
### vgg16 model architecture
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    ...      
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 25088)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              102764544 
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 4096)              16781312  
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 5)                 20485     
    =================================================================
    Total params: 134,281,029
    Trainable params: 119,566,341
    Non-trainable params: 14,714,688
    _________________________________________________________________
```

CAM을 얻기 위해서 논문에서는 다음과 같이 모델 아키텍쳐를 수정한다.
1. conv5-3 이후의 레이어는 사용하지 않는다. 
2. conv5-3 이후에 filter 크기가 3x3, stride 1, pad 1, channel이 1024인 컨볼루션 레이어를 더한다.
3. GAP 레이어를 더한다.
4. Softmax 레이어를 더한다.
5. 이미지넷 데이터로 fine-tuning 한다.

이를 구현하기 위해 다음과 같이 작업하였다.

0. pretrained된 VGG16 모델을 가져온다.  
이때 include_top = False를 넣어서 fully connected를 날린다.
```python
base_model = vgg16.VGG16(include_top = False, input_shape=(img_width, img_height, img_channel))
```

1. conv5-3 이후의 레이어는 사용하지 않는다.  
fc를 날려도 마지막 레이어는 conv가 아닌 pooling layer다. pooling layer도 필요가 없다. 모델의 layers 프로퍼티에 접근해서 pop을 하면 마지막 레이어가 날아간다. 그리고 output을 아래와 같이 가장 마지막 레이어(conv5-3)로 새로 지정해야 모델의 아웃풋이 수정된다.
```python
base_model.layers.pop()
base_model.outputs = [base_model.layers[-1].output]
```

2. conv5-3 이후에 filter 크기가 3x3, stride 1, pad 1, channel이 1024인 컨볼루션 레이어를 더한다.  
tf.keras에서 제공하는 Conv2D는 padding이 `valid`아니면 `same` 둘 중 하나다. 
논문에서는 padding을 1로 주라고 되어있으므로, ZeroPadding2D를 사용해서 padding을 먼저 주고, Conv2D를 붙였다. 
또 논문에서는 activation function에 대한 설명이 없는데, VGG16의 conv layer가 이를 `relu`로 사용하고 있어 이를 따르기로 했다. 
(activation function을 None으로 넣고 학습시켜봤는데 loss가 엄청 폭발한다.) Stride 디폴트가 (1,1)이므로 생략했다.
```python
last = base_model.outputs[0]
x = ZeroPadding2D(padding=(1, 1))(last)
x = Conv2D(filters=1024, kernel_size=(3, 3), activation='relu')(x)
```

3. GAP 레이어를 더한다.  
```python
x = GlobalAveragePooling2D()(x)
```

4. Softmax 레이어를 더한다.
```python
preds = Dense(nb_classes, activation='softmax')(x) ## nb_classes=5
```

5. 이미지넷 데이터로 fine-tuning 한다.
이미지넷 대신에 대선주자 5인 데이터를 활용해 fine-tuning을 실시했다.

<hr>

## 대선주자 얼굴 위치 추적기
2017년 대선에 참여한 주요 후보 5인의 얼굴 데이터셋을 사용해 CAM으로 분석을 진행해본다.

![대선주자 5인](/assets/materials/20180104/cands.png)


### Training

구글 크롤링으로 모은 안철수(ahn), 홍준표(hong), 문재인(moon), 심상정(sim), 유승민(you) 데이터를 train, validation, test셋으로 랜덤 분리했다.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l"></th>
    <th class="tg-yw4l">ahn</th>
    <th class="tg-yw4l">hong</th>
    <th class="tg-yw4l">moon</th>
    <th class="tg-yw4l">sim</th>
    <th class="tg-yw4l">you</th>
  </tr>
  <tr>
    <td class="tg-yw4l">train</td>
    <td class="tg-yw4l">344</td>
    <td class="tg-yw4l">309</td>
    <td class="tg-yw4l">282</td>
    <td class="tg-yw4l">275</td>
    <td class="tg-yw4l">273</td>
  </tr>
  <tr>
    <td class="tg-yw4l">validation</td>
    <td class="tg-yw4l">43</td>
    <td class="tg-yw4l">38</td>
    <td class="tg-yw4l">35</td>
    <td class="tg-yw4l">34</td>
    <td class="tg-yw4l">34</td>
  </tr>
  <tr>
    <td class="tg-yw4l">test</td>
    <td class="tg-yw4l">37</td>
    <td class="tg-yw4l">30</td>
    <td class="tg-yw4l">30</td>
    <td class="tg-yw4l">29</td>
    <td class="tg-yw4l">28</td>
  </tr>
</table>

논문에서는 fully connected를 GAP로 바꿔도 성능 저하가 크게 나타나지 않았으며, GAP로 인해 발생하는 성능 저하를 추가적인 conv layer를 더해 해소하였다고 한다.

> We observe that AlexNet is the most affected by the removal of the fully connected layers. To compensate, we add two convolutional layers just before GAP resulting in the AlexNet*_GAP network. We find that AlexNet*-GAP preforms comparably to AlexNet. 

VGGnet 역시 VGGnet-GAP로 명명한 것으로 보아, 마찬가지로 conv layer를 더해 성능을 보정한 것으로 보인다.

그러면 대선주자 데이터셋에 대해서 기본 아키텍처 그대로 fc를 사용한 모델과 GAP 모델간에는 어떤 차이가 있었을까? 비교를 위해 다음과 같은 3가지 모델을 만들었다.

1. VGGnet-ORG: 기본 VGG16 구조  
~conv5_c + pool5 + fc(4096) + dropout + fc(4096) + dropout + softmax

2. VGGnet-ORG1024: 기본 VGG16 구조에 fc의 유닛 수를 1024로 줄인 모델   
~conv5_c + pool5 + fc(1024) + dropout + fc(1024) + dropout + softmax

3. VGGnet-ORG512: 기본 VGG16 구조에 fc의 유닛 수를 1024로 줄인 모델   
~conv5_c + pool5 + fc(512) + dropout + fc(512) + dropout + softmax

4. VGGnet-GAP: 논문에서 제안한 CAM 구조  
~conv5_c + conv + GAP + softmax

최초 실험에서는 1과 4번 모델만 테스트해보았으나, 기존 VGG16의 구조가 대규모 이미지넷 데이터셋 분류에 특화되어 만들어졌다는 점을 감안하여 (fc의 파라미터 수가 너무 많았음) fc의 유닛 수를 줄인 2번과 3번 모델을 추가적으로 만들었다.


### Model Evaluation

이들에 대한 성능 평가표는 아래와 같다.

![performance evaluation](/assets/materials/20180104/performance_evaluation.png)

test accuracy를 기준으로 봤을때 VGGnet-GAP의 성능이 다른 모델들보다 압도적으로 좋았다. 클래스별 이미지 수가 비슷하므로 적어도 정확도가 20%를 넘어야 하는데, VGG16-ORG는 기준을 밑돈 것을 보아 아예 학습이 제대로 이루어지지 않았다고 볼 수 있다. 
4096짜리 fc를 2개를 중첩하다보니 파라미터 수가 폭증했다. 
우측 차트를 보면 그 차이를 현격하게 확인할 수 있다. 이보다 더 적은 유닛을 사용한 VGGnet-ORG1024는 원래 모델에 비해서는 성능이 더 낫기는 했으나, 논문에서처럼 GAP보다 낫거나 비슷한 성능을 보이지는 않았다. 
2번 모델의 성능 개선에 고무되어 fc의 파라미터를 반으로 줄인 ORG512를 만들었으나, 오히려 정확도는 떨어지는 결과가 나왔다.

![performance evaluation table](/assets/materials/20180104/performance_evaluation_table.png)


### Visualising Class Activation Map

성능 평가를 통해 fc를 GAP으로 대체하는 방식에서 큰 성능 저하가 일어나지 않았음을 확인하였다. (혹은 논문에서와는 달리 대선주자 데이터셋에서는 GAP가 가장 나은 성능을 보여주었다.) 
그렇다면 이제는 이 모델이 이미지의 어떤 부분에서 결과를 추론했는지 살펴보자.

![CAM result](/assets/materials/20180104/cam_result.png)

Voila! 예전에 처음 이미지 분류 모델을 만들때만 하더라도, 분류 정확도는 좋게 나왔지만 혹 각 후보들의 상징색을 보고 분류를 내리는게 아닐까 의심스러웠었다. 
특히 대선기간을 앞두고 데이터를 크롤링하다보니 더더욱 그런 경향성이 두드러지지 않을까 싶었다. 결과를 보자면 일단 크게 우려하지는 않아도 좋을 듯 하다. 
먼저 각 열의 최상단에는 각 이미지(테스트셋 중 택 1)에 대한 분류 확률을 출력했다. 모두 0.9 이상으로 올바르게 분류되었다. 모델이 정상적으로 동작했음을 알 수 있다.

2번째 행 CAM 이미지를 보면 녹색으로 발광하는 부분들이 보인다. 예측한 클래스에 대해서 모델이 반응한 이미지 영역이다. 
이에 투명도를 0.5정도 주고 이미지위에 올린 3번째 행을 보면, 모델이 대략 얼굴을 보고 판단을 내렸음을 어렵지 않게 유추할 수 있다. 

마지막 4번째 행은 바운딩 박스를 씌운 결과다. 두가지 흥미로운 점이 발견된다. 먼저, 이 모델은 바운딩 박스가 처리되지 않은 이미지와 레이블만 넣고 학습했음에도 불구하고, 각 후보의 얼굴을 꽤 정확하게 찾아냈다. 
바로 바운딩 박스를 잘라내서 얼굴 데이터로 쓰기에는 규격이 제각각이지만, 이정도 결과가 어딘가 싶다. 

두번째는 3번째 열 심상정 후보의 결과다. 모델이 어떻게 동작했는지 아는 것은 어떻게 개선해야할지에 대한 실마리를 준다. 
심상정 후보의 CAM과 바운딩 박스를 보면, 모델이 얼굴 외에도 '정의'라는 글자에 반응했음을 알 수 있다. 실제로 트레이닝 데이터셋을 보자.

![심상정 - 일부 트레이닝 셋](/assets/materials/20180104/sims.png)

(딱히 '정의'가 빗발치진 않는데...) 일부 이미지에서 당의 로고 부분이 보인다. 확실치는 않지만 모델이 '정의'라는 단어나 주변 색을 '심성정' 레이블과 연결지은 듯 하다. 
심상정 후보의 이미지 갯수가 다른 세 후보보다 약간 적은 편인데, 가급적 당의 색이나 '정의' 단어가 배경에 없는 이미지를 추가해서 데이터셋을 보강하는 쪽으로 개선할 수 있을 듯 하다.


### Inferece on video

독사진에 대해서는 성능이 그럴듯 하다는 것은 잘 알았다. 혹시 더 어려운 데이터를 넣어도 모델이 잘 작동할까? 
만약 한 이미지에 같은 대선주자 2명이 등장하면 모델은 어떤 판단을 내릴까? 여전히 CAM은 잘 작동할까?

이를 알아보기 위해 한 대통령선거 TV 토론회의 동영상을 가져와 모델을 돌려보았다. 


[![대선토론회 클립](https://img.youtube.com/vi/9TapvvLTNQ0/0.jpg)](https://www.youtube.com/watch?v=9TapvvLTNQ0 "Video Title")


이미지 분류용 모델을 영상에 씌워본 경험이 없어 일단 프레임을 잘라서 인퍼런스하고 이를 gif로 바꾸는 편법을 써봤다. 아래는 각 주자별 주요 클립에 대한 분류 확률 및 바운딩 박스 결과다.

![안철수](/assets/materials/20180104/ahn_bbox.gif)

![홍준표](/assets/materials/20180104/hong_bbox.gif)

![문재인](/assets/materials/20180104/moon_bbox.gif)

![심상정](/assets/materials/20180104/sim_bbox.gif)

![유승민](/assets/materials/20180104/you_bbox.gif)

기존 테스트셋 정확도에 비해 영상에 인퍼런스를 수행한 결과는 다소 실망스럽다. 트레이닝 데이터와 영상 프레임의 이미지 특성이 달라서, 혹은 224 x 224 사이즈로 리사이징 하는 과정에서 정확도가 희생되었을 수도 있겠다. 
안철수, 홍준표, 유승민 후보에 대해서는 그나마 분류 확률이나 바운딩 박스가 괜찮게 출력되었으나, 심상정, 문재인 후보에 대해서는 모델이 제대로 동작하지 않았다. 특히 독사진에 대해서도 분류 정확도가 높지 않았다. 원인은 아직 잘 모르겠다.

또, 두명의 주자가 동시에 나오거나 아무도 나오지 않은 장면에 대해서 모델이 제대로 분류를 수행하거나 박스를 그리지 못했다. 
두명의 주자가 분명히 나온 프레임에 대해 특정 주자에게 확률을 몰아주거나, 아무것도 없는 화면에서도 특정 주자를 검출하는 현상이 발견되었다. 
두번째 문제를 해결하기 위해서는 [아무도 등장하지 않음]이라는 레이블을 단 데이터를 학습에 포함시킴으로써 해결할 수 있을 듯 하다. 
하지만 한명에 대해 분류를 수행하는 모델이 2명 이상이 등장하는 이미지에 대해서 잘 동작하게 하도록 하는 방안은 아직 떠오르지 않는다. 
이미지넷 분류나 세그멘테이션 결과를 보면 여러 이미지에 대해서 바운딩 박스를 그리던데, 그쪽 연구 결과를 살펴보면 힌트를 얻을 수 있지 않을까 싶다.


end.


## Related Posts
- <a href="https://jsideas.net/python/2018/01/12/grad_cam.html">Grad-CAM: 대선주자 얼굴 위치 추적기</a>  
- <a href="https://jsideas.net/python/2018/07/03/acol.html">Adversarial Complementary Learning</a>



## Reference
- https://arxiv.org/abs/1512.04150
- https://github.com/jacobgil/keras-cam
- <a href="https://stackoverflow.com/questions/16937158/extracting-connected-objects-from-an-image-in-python
">stack overflow</a>