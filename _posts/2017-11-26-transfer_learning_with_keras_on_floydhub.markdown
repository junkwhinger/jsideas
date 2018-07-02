---
layout:     post
title:      "Transfer Learning wtih Keras on FloydHub"
date:       2017-11-26 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/keras/Header.png
---

# Transfer Learning wtih Keras on FloydHub

## tf.keras
딥러닝 라이브러리로는 `tensorflow`가 가장 널리 사용되지만 다른 편리한 선택지도 많다. `Keras`는 2016년 9월 기준으로 텐서플로우에 이어 두번째로 많이 쓰이는 라이브러리이며, 빠르고 간결하게 딥러닝 모델을 구현할 수 있다는 큰 장점이 있다. 또 2017년 들어 텐서플로우 라이브러리 안에서도 keras를 사용할 수 있게 되면서 사용상 번거로움도 줄었다. 텐서플로우 최신 버전 1.4에서는 <a href='https://www.tensorflow.org/api_docs/python/tf/keras'>tf.keras</a>로 바로 사용할 수 있다.

![Keras + TensorFlow](/assets/transfer_learning_with_keras_on_floydhub/keras-tensorflow.jpg)

## FloydHub
회사 업무에서는 컴퓨터에 내장된 고사양 GPU나 EC2 인스턴스를 자유롭게 쓸 수 있으나, 집에서는 그만큼 좋은 딥러닝 환경을 구축하기 어려운 경우가 많다. 특히 집에서는 맥북에서 코딩을 하는데 내장된 GPU가 딥러닝을 지원하지도 않고, 또 찾아본 바로는 external GPU를 붙이는 것도 가성비가 좋지는 않다고 한다. 그래서 최근 쉽게 사용할 수 있는 Cloud GPU를 알아보다 <a href="https://www.floydhub.com">FloydHub</a>를 사용하게 되었다.

평소에 나는 tensorflow가 깔린 conda 환경에서 jupyter notebook을 실행시켜 딥러닝을 돌려보는데, 이런 실행 환경에서 FloydHub를 매우 자연스럽게 사용할 수 있다. `pip install floyd-cli`를 깔고 간단한 인증 과정을 거치면 터미널에서 바로 FloydHub를 사용할 수 있게 된다. 허브를 띄울 때 몇가지 파라미터를 넣을 수 있다.

* `--gpu` : GPU 머신을 돌린다.
* `--mode jupyter` : 주피터 환경을 실행한다.
* `--env tensorflow-1.4` : 텐서플로우 버전 1.4로 환경을 설정한다.
* `--data myacc/datasets/mydata` : 사용할 데이터셋의 위치를 설정한다.
* `--tensorboard` : 텐서보드를 실행한다.

```
$ floyd run --gpu --env tensorflow-1.4 --tensorboard --data junkwhinger/datasets/presidential_candidates_2017 --mode jupyter
```
이런식으로 터미널에서 실행시키면 브라우저 창에서 FloydHub 페이지가 뜨면서 localhost에서 jupyter notebook을 쓰는 것처럼 바로 딥러닝을 돌려볼 수 있다. 랩탑에 최신 GPU가 없어도, 귀찮게 환경설정이나 텐서보드를 실행하지 않고도 말이다.

프라이싱은 무료회원이나 한달에 9달러를 내는 Data Scientist 회원은 GPU를 시간당 0.59달러에 빌릴 수 있다. jupyter notebook을 실행하면 running이라고 뜨는데, 그게 활성화된 시간만큼 잔여 시간에서 깎는 구조다. AWS EC2 인스턴스랑 쓰는 것은 비슷하지만 그보다 훨씬 간편하게 사용할 수 있어 좋았다.

![FloydHub 실행화면](/assets/transfer_learning_with_keras_on_floydhub/floydhub.png)


## 대선주자 얼굴 분류기 revisited
큰맘 먹고 GPU 100 시간을 지른 기념으로 올초 벚꽃대선을 맞아 만들어본 <a href="http://jsideas.net/python/2017/05/07/transfer_learning.html">대선주자 얼굴 분류기</a>를 Keras와 FloydHub를 사용해서 다시 돌려봤다. 당시에는 VGG16을 사용해서 여러 transfer learning 방법을 적용해보았는데, 이번에는 Keras에서 제공하는 여러 다른 pre-trained 모델을 FloydHub 환경에서 사용해보았다.


## keras의 pre-trained 모델들
Transfer Learning을 할 때는 보통 이미 학습이 끝난 모델을 불러와서 끝단에 새로 만든 분류 레이어를 더한 다음 fine-tuning을 한다. 예전에 텐서플로우로 짰을 때는 Model 파일을 내려받아서 모델을 새로 짜고 가공하기가 좀 번거로운 점이 있었는데, Keras로는 이 작업을 매우 쉽고 오류 적게 처리할 수 있다. 게다가 2017년 들어 구글 텐서플로우 팀이 Keras를 텐서플로우 코어 라이브러리에서 제공하면서 더 사용하기 편해졌다. tf.keras에서 제공하는 pre-trained 모델 중에 VGG, Inception_V3, Xception, ResNet50을 사용해보았다. 공부하면서 알게된 내용을 간략히 정리해본다.

- VGG16 & 19
  - 2014년 이미지넷 대회에서 두각을 보인 모델로, Oxford VGG 그룹에서 창안함.
  - 컨볼루션 레이어와 FC를 16 혹은 19층으로 깊게 쌓은 모델로, 층이 깊어질수록 성능이 좋아진다는 것을 증명함.
  - 11x11과 같이 큰 컨볼루션 필터보다는 여러개의 작은(3x3) 필터의 조합이 파라미터 수 등에서 더 효과적임을 증명함.
  - Image Segmentation 등 다른 이미지 태스크에도 자주 활용됨. 데뷰 2017에서 본 자율주행 차선변경 프로젝트에서도 VGG16을 사용함.
  - 단순하고 직관적인 구조로 이해와 구현이 쉬움.
  - 반면 학습시간이 굉장히 오래 걸리고, 모델 용량이 약 540~570MB로 꽤 큼.
- Inception_v3
  - 2014년 이미지넷에서 두각을 보인 GoogleNet의 후속 버전 중 하나. 
  - v3는 v2에 Batch Normalization과 여러 마이너 개선이 추가된 버전임.
  - 단순히 깊이만 늘린 VGG와 달리 Inception 모델은 한번에 여러 크기의 필터를 동시에 사용하는 Inception 아키텍쳐를 채택함. 인셉션 모듈은 이미지에서 여러 차원의 피쳐를 추출하는 방식으로, 하나의 모듈 안에서 1x1, 3x3, 5x5 등 여러 필터로 이미지를 처리하고 그 결과를 묶은 다음 다음 레이어로 넘김.
  - 컨볼루션 레이어는 가로-세로(2차원)와 채널(+1차원)에서 필터를 학습함. 이때 각 필터는 채널간 상관성(cross channel correlation)과 공간상의 상관성(spatial correlation)을 동시에 맵핑하게 됨. 인셉션은 이 과정을 더 효율적으로 처리하기 위해 채널간 상관성과 공간상 상관성을 따로따로 처리할수 있는 구조를 취함. 그래서 1x1 컨볼루션으로 차원을 줄이고 3x3이나 5x5 컨볼루션을 뒤이어 실행하는 방식임. 이를 더 익스트림하게 분리한 케이스가 Xception.
  - Inception_v3 모델은 VGG나 ResNet에 비해 용량이 96MB 정도로 작음.
- XCEPTION
  - 인셉션 모델의 Extreme 버전으로, 2017년 Keras 라이브러리의 창시자 Francois Chollet가 제안함.
  - 채널간, 공간상 상관성 분리를 기조로 하는 인셉션 모델을 극단으로 밀어붙여 depthwise separable convolution 구조를 만듦. 인풋에 1x1 컨볼루션을 씌운 후 나오는 모든 채널에 3x3 컨볼루션 연산을 수행하는 개념.
  - separable convolution은 크로스 뎁스 피쳐와 2d 피쳐를 구분해서 처리할 수 있음. 이러면 cross depth 피쳐가 노멀 컨볼루션에 의해 파괴되지 않음. 익스트림 인셉션 가설이라고 함. 이 부분에 대해서는 다른 포스팅에서 더 자세하게 공부해보는 것으로 함.
  - 91MB로 모델 중 가장 가벼움.
- RESNET50
  - 층을 더 많이 쌓을수록 성능이 개선될 듯 하지만 그만큼 학습이 더 어려워지고 트레이닝 에러가 커지는 문제가 발생함. 이 문제를 해결하고 더 깊은 층을 쌓기 위해 residual connection을 이용한 것이 ResNet.
  - ResNet을 구성하는 residual block은 일반적인 컨볼루션 연산과는 다름. 입력 x를 일련의 컨볼루션 연산을 통해 처리한 F(x)에 다시 x를 더해, H(x)를 만들어 다음 레이어에 넘김.
  - 그래서 H(x) = F(x) + x를 F(x) = H(x) - x로 볼 때, ResNet의 residual block은 잔차(residual)를 학습하는 것이라고도 보여짐
  - 논문에서는 각 레이어를 학습시켜 우리가 바라는 이상적인 목표에 피팅하는 것보다, 잔차를 학습하고 이 잔차가 0이 되도록 푸시하는 것이 더 최적화하기 쉬운 문제라고 함.
  - residual block 덕분에 깊이가 엄청 깊어지면서도 학습이 잘 되는 장점을 가져감.

  
## Transfer Learning with Keras
tf.keras의 applications에서 위에 언급한 다양한 pre-trained 모델을 가져다 쓸 수 있다. 입출력이 깔끔하게 정리된 모듈이어서 모델명만 바꿔가면서 매우 쉽게 여러 모델을 테스트할 수 있는 장점이 있다.

Keras에서 transfer learning은 다음과 같은 방식으로 진행했다.

1. 사용할 pre-trained 모델을 가져온다. 모델 끝부분에 달린 FC 분류레이어를 파인튜닝해야 하므로 이를 제외하기 위해 include_top 파라미터에 False를 넘긴다.
```python
base_model = vgg19.VGG19(include_top = False, 
                           input_shape=(img_width, img_height, img_channel))
```
{% highlight python %}

{% endhighlight%}

2. pre-trained 모델에 새로운 분류 레이어를 만들어붙인 새로운 모델을 정의한다. 여기서는 FC 대신에 Global Average Pooling을 사용해봤다. FC는 오버피팅에 취약하고 Flatten하는 과정에서 피쳐의 위치정보가 소실되는 단점이 있는 반면, Global Average Pooling은 피쳐맵의 평균값을 softmax에 넘겨 피쳐 영향력을 해석하기 용이하고 구조 자체가 오버피팅을 방지하는 효과가 있다. 
```python
last = base_model.output
x = GlobalAveragePooling2D()(last)
x = Dense(512, activation='relu')(x)
preds = Dense(nb_classes, activation='softmax')(x)
```

3. 불러온 원래 모델의 웨이트를 학습하지 않도록 설정하고 컴파일한다.
```python
for layer in base_model.layers:
    layer.trainable = False
tl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
![마지막 dense 레이어의 파라미터만 학습하도록 지정되었다](/assets/transfer_learning_with_keras_on_floydhub/trainable.png)

4. callback을 사용해서 EarlyStopping과 텐서보드를 사용해보자.
```python
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(patience=10, monitor='val_acc')
tb = TensorBoard(log_dir='vgg19_tl')
```
학습 epoch 수가 많아지면 어느 순간 과적합이 발생하면서 training accuracy는 높아지지만 validation accuracy가 꺾이는 순간이 온다. validation accuracy가 좋은 시점에서 적절하게 학습을 그만 두어야 하는데, 이를 자동으로 처리해주는 것이 EarlyStopping 콜백이다. patience 파라미터를 10로 지정하면, validation accuracy가 직전 최고점수에 비해 떨어지는 걸 연속 10번까지 견디게 된다. Tensorboard 콜백은 지정한 log_dir로 loss, accuracy 등의 정보를 저장해준다. 여기서 정의한 콜백들을 뒤에 fit을 수행할때 넘기기만 하면 된다.

5. 모델을 학습한다.
```python
tl_model.fit_generator(generator=train_generator, 
                       epochs=300, 
                       validation_data=validation_generator, 
                       steps_per_epoch=50, 
                       validation_steps=10,
                       callbacks=[es, tb])
```
텐서플로우를 쓸 때는 for문으로 epoch을 돌면서 행할 행동을 지정해야 했는데, keras에서는 fit만으로 간단히 학습을 실행할 수 있어 매우 좋다. 또 generator를 넘길 수 있어 사용이 매우 간편하다.

6. 여러 노트북을 만든 후, 모델만 다르게 지정하기만 하면 된다.
```python
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras.applications import inception_v3
from tensorflow.python.keras.applications import xception
from tensorflow.python.keras.applications import resnet50
```
resnet50를 제외한 모든 모델은 가로세로 150, 150 이미지를 입력으로 받을 수 있다. resnet50는 197, 197만 받도록 되어있어 이 모델만 따로 이미지 사이즈를 지정했다.

## Running on FloydHub

FloydHub에 GPU 모드로 진입한 다음, 위 5개 모델을 각각의 노트북으로 만들고 TensorBoard를 callback으로 넣어 학습시켰다. Floyd 문서에 보니 GPU는 Tesla K80를 쓴다고 하는데, 아마존에서 얼마인지 대강 보니 3천불 정도 하는 듯 하다. 

![Tesla K80 on Amazon](/assets/transfer_learning_with_keras_on_floydhub/tesla_K80.png)


속도 비교를 위해 같은 VGG19 모델 코드를 FloydHub GPU 환경과 내 로컬 맥북 CPU로 돌려보았다.

![FloydHub GPU(Tesla K80) / VGG19 Fine-Tuning 속도](/assets/transfer_learning_with_keras_on_floydhub/floydhub_gpu.gif)

![local CPU(Intel i7) / VGG19 Fine-Tuning 속도](/assets/transfer_learning_with_keras_on_floydhub/local_cpu.gif)

FloydHub GPU에서는 epoch 하나가 거의 10초 내에 끝나는데 반해 로컬 CPU에서는 360초, 그러니까 전자가 대략 36배 빠르다고 볼 수 있다.

이번에는 돌려본 5가지 모델의 성능 비교를 위해 텐서보드를 실행시켜보았다. FloydHub를 실행시킬때 `--tensorboard` 옵션을 주면 아래 그림과 같이 텐서보드 링크가 뜬다.

![텐서보드 링크](/assets/transfer_learning_with_keras_on_floydhub/tensorboard_link.png)


![텐서보드 모델 loss & accuracy](/assets/transfer_learning_with_keras_on_floydhub/tensorboard.png)

Validation Accuracy를 기준으로 보자면 ResNet > VGG16 > Xception & VGG19 > InceptionV3 순으로 성능이 좋았다. ResNet이 가장 성능이 좋을거라고 예상하기는 했는데, 그 과정이 흥미롭다. 나머지 모델들은 초반부터 validation accuracy의 개선 속도가 급격한데 반해, ResNet은 10번째 Epoch을 돌때까지만해도 그 정확도가 0.2 정도로 오지선다를 랜덤으로 찍는 수준을 기록했다. 또 validation loss 역시 다른 모델과 달리 급격하게 상승하는 추세를 보인다. 보통 training loss가 감소할 때 validation loss가 증가한다면 training dataset에 대한 오버피팅으로 보기에, 처음에는 이 데이터셋에 적용하기에는 ResNet의 구조가 너무 복잡한게 아닐까하는 의심도 들었다. 그런데 12번째 Epoch부터 Validation Accuracy가 거의 10%p씩 오르더니 이내 다른 모델을 추월해버렸다. 결과적으로 5가지 모델 중 가장 낮은 validation loss를 기록했다. 하지만 여전히 training result에 비해 validation result가 좋지 않은 점은 긍정적으로 해석하기 어려우며 그 원인 중 하나는 적은 데이터셋일 수도 있겠다는 생각이 든다. (Global Average Pooling 뒷단에 FC를 제거하고 바로 softmax로 넘겨보기도 했는데 결과는 비슷했다.)

또 한가지는 VGG16의 선방이다. 깊이로 따지자면 VGG19이, 구조적으로는 InceptionV3나 Xception이 더 나은 성능을 보일거라 예상했지만, ResNet에 가장 근접한 성과를 낸 것은 VGG16이었다. 층이 깊어질수록 학습이 어려워지는 것을 ResNet에서 residual connection으로 보완했다는 것을 생각해볼 때 그런 보완 구조 없이 단순히 컨볼루션 레이어를 깊게 쌓은 VGG의 경우 16이 19보다 더 나은 성능을 보이는 것일지도 모르겠다. 

## Inference with ResNet
좀 수상하지만 강력한 성능을 보인 ResNet을 사용해서 얼굴 분류를 시도해보자. Keras를 이용해 간단히 학습을 진행해본 것처럼, 간단하게 모델을 저장하고 불러와서 인퍼런스를 돌려볼 수 있다. 모델을 저장할때 2개 파일을 만드는데, 하나는 모델 아키텍쳐가 저장된 json 파일이고, 다른 하나는 weight 정보를 담은 h5 파일이다. h5파일은 90.4MB 정도로 VGG에 비해 확실히 가벼웠다.
```python
model_json = tl_model.to_json()
with open("resnet_model.json", "w") as json_file:
    json_file.write(model_json)
tl_model.save_weights("resnet.h5")
```

인퍼런스를 수행할 노트북에서는 아래처럼 파일을 불러와서 model을 만들면 된다.
```python
with open('resnet_model.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('resnet.h5')
```

urllib과 cv2를 적절히 활용해 구글 이미지 링크에서 이미지를 따와서 numpy array로 만든 다음,
이를 resnet에 맞게 197, 197 사이즈로 변환하고 인퍼런스를 수행한 다음 결과를 디스플레이하는 코드를 만든다.
```python
def predict_who_is_who(url):
    img = url_to_image(url)
    
    ## to fit ResNet
    resized_img = imresize(img, size=(197, 197))
    adj_img = np.expand_dims(resized_img / 255, 0)
    
    ## make prediction
    res = np.squeeze(loaded_model.predict(adj_img))
    
    ## ground_truth
    gt = np.array(['ahn', 'hong', 'moon', 'sim', 'you'])
    result = pd.DataFrame(res, index=gt)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ## present image
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.axis("off")

    ## present bar plot
    result.plot(ax=ax2, kind='bar')

    ## annotate bar pot
    rects = ax2.patches
    labels = [round(res[l], 3) for l in range(len(rects))]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label, ha='center', va='bottom')

    ax2.set_title("Face recognition test with ResNet50", fontsize=20)
    ax2.tick_params(labelsize=20)
    plt.xticks(rotation='horizontal')
    plt.show()
```

결과는 아래와 같다.

![문재인 - 성공](/assets/transfer_learning_with_keras_on_floydhub/moon.png)
![안철수 - 성공](/assets/transfer_learning_with_keras_on_floydhub/ahn.png)
![홍준표 - 실패](/assets/transfer_learning_with_keras_on_floydhub/hong.png)
![심상정 - 성공](/assets/transfer_learning_with_keras_on_floydhub/sim.png)
![유승민 - 성공](/assets/transfer_learning_with_keras_on_floydhub/you.png)

마지막 번외 - 유담씨
![유담 - 실패?](/assets/transfer_learning_with_keras_on_floydhub/youdam.png)



## 참고자료
- https://en.wikipedia.org/wiki/Keras
- http://nicolovaligi.com/history-inception-deep-learning-architecture.html
- https://datascience.stackexchange.com/questions/15328/what-is-the-difference-between-inception-v2-and-inception-v3
- https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
- https://www.quora.com/Why-does-a-depth-wise-separable-convolution-model-like-Keras-Xception-perform-exceptionally-well-compared-to-GoogleNet-Inception-or-any-other-TL-models
- https://arxiv.org/pdf/1610.02357.pdf
- https://blog.waya.ai/deep-residual-learning-9610bb62c355
- http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/
- http://openresearch.ai/t/inception-v1-going-deeper-with-convolutions/40
- http://openresearch.ai/t/xception-deep-learning-with-depthwise-separable-convolutions/49
- https://www.quora.com/What-is-global-average-pooling
