---
layout:     post
title:      "Transfer learning: 대선주자 얼굴 분류기"
date:       2017-05-07 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/cand_faces/header.png
tags: featured
---

## Transfer learning: 대선주자 얼굴 분류기

CNN(Convolutional Neural Network)은 이미지 처리에 가장 널리 사용되는 알고리즘이다. 숫자 손글씨 데이터셋인 MNIST는 28x28 픽셀로 되어 있다. 이를 일반적인 DNN(Deep Neural Network)로 처리하려면, 먼저 이미지 하나를 784(28x28)개의 소수점 숫자로 된 벡터로 변환한 다음, 이를 뉴럴넷에 집어넣어 학습시켜 분류를 한다. 꽤 괜찮은 성능이 나오기는 하지만, CNN만큼은 아니다. 

CNN이 왜 DNN보다 이미지 처리에 강할까? 우리 눈과 뇌가 이미지를 어떻게 처리하는지 생각해보면 직관적으로 이해할 수 있다. 숲 사진을 본다고 할 때, 우리는 나무, 풀, 동식물 등 숲을 이루는 작은 부분을 묶어서 숲을 인지한다. 나무 뿌리를 하나의 픽셀로 본다면, 그 픽셀 위에는 줄기가, 또 그 위에는 잎파리가 있을 것이다. DNN이 이미지를 기계적으로 줄세워서 처리한다면, CNN은 (마찬가지로 기계적이지만) 근접한 픽셀끼리 묶어서 처리한다는 점에서 데이터의 공간적인 특성을 살린 처리가 가능하다. 

![숲](/assets/cand_faces/forest.jpg)

하나의 필터가 동일한 `weight`로 전체 이미지를 훑어 `feature map`을 만드는 CNN의 특성은 학습해야 할 파라미터의 수를 줄여 학습 효율이 좋긴 하지만, 좋은 분류기를 만드는데 꽤 오랜 학습 시간이 걸린다. MNIST는 `convolution layer` 2개로도 금방 결과가 나오지만, 애초에 데이터의 크기가 작고 (28x28), 컬러 채널이 1개 뿐이다. 하지만 우리가 분류하기를 원하는 현실 세계의 데이터는 그보다 크고 색깔도 다양하다.

![MNIST](/assets/cand_faces/mnist.png)

여기서 transfer learning이라는 재미있는 개념이 등장한다. 처음부터 CNN을 학습시키자니 너무 시간이 오래걸리고 효율이 낮다면, 성능이 입증된 CNN을 가져다가 피쳐를 추출하고, 이를 바탕으로 우리가 원하는 분류를 수행하도록 만드는 것이다. 개인적으로는 렌즈를 갈아끼우는 카메라와 비슷하다는 느낌을 받았다. 바디가 같더라도 어떤 렌즈를 끼우느냐에 따라 시야가 넓은 사진이나 멀리서도 줌인해서 사진을 찍는.

![렌즈 교환식 카메라](/assets/cand_faces/camera.png)

<a href="http://cs231n.github.io/transfer-learning/">CS231n</a>에 더 자세히 설명되어 있는데, 요약하자면 이러하다. 실질적으로 전체 CNN을 처음부터 끝까지 학습하는 경우는 별로 없다. 왜냐하면 그만큼 큰 데이터셋을 구하기가 어렵기 때문이다. 그 대신에 이미 대규모 데이터를 대상으로 학습이 끝난 ConvNet을 가져다가 초기값으로 사용하거나 고정된 피쳐 추출기로 사용할 수 있다. 

ConvNet을 고정된 피쳐 추출기로 사용하는 경우에는 ConvNet 끝에 달린 fully-connected layer를 없애고, convolutional layer를 통해 처리되는 값만 얻으면 된다. 이를 얻어진 피쳐를 `CNN codes`라 한다. 보통 ImageNet에서 데이터를 처리할 때 `ReLU`를 사용하므로, 반드시 이 `codes`가 `ReLU`로 처리되도록 신경쓰자. 

ConvNet을 fine-tuning하는 경우에는 끝의 fully-connected layer를 제거할 뿐 아니라 앞선 convolutional layer를 새로운 데이터를 사용해서 다시 학습시키고, 역전파를 통해 `weight`도 업데이트해야 한다. 전체 convolutional layer를 업데이트하거나, 뒷단만 업데이트하는 것도 가능하다. (왜냐햐면 앞단의 레이어를 통해 얻어지는 것은 직선이나 곡선같은 원시적인 피쳐이므로 쓸모가 많으나, 뒷단의 고차원적 피쳐는 특정 도메인에 종속될 수 있으므로)

그럼 언제 어떻게 transfer learning을 사용해야 할까?

1. 작고 (ImageNet과) 비슷한 데이터셋
=> 오버피팅의 위험이 있으므로 CNN codes에 분류기만 학습시킨다.

2. 크고 비슷한 데이터셋
=> 데이터가 많으므로 fine-tuning해도 좋겠다.

3. 작고 다른 데이터셋
=> 데이터셋이 작으므로 분류기만 학습시키는게 좋겠다. 데이터셋 자체가 매우 다르므로, ConvNet을 모두 통과한 결과보다는 그 앞에서 나온 결과를 가지고 분류기를 학습시키는 것이 더 나을 수 있다. 

4. 크고 다른 데이터셋
=> 이 경우 그냥 CNN을 처음부터 만들수도 있다. 하지만 현실적으로는 pre-trained된 모델의 `weight`로 초기값을 설정하고 학습시키는 것이 더 나은 경우가 많다. 데이터셋이 많은 경우 Convolution layer를 처음부터 끝까지 파인튜닝하는 것도 가능하다.


### VGG16
VGG16은 University of Oxford의 <a href="http://www.robots.ox.ac.uk/~vgg/research/very_deep/">Visual Geometry Group</a>에서 만든 모델로 ImageNet Challenge 2014에서 `localisation and classification` 부문 1, 2등을 수상했다고 한다. 이미지 분류 성능이 좋을 뿐만 아니라, 분류에 사용된 `weight`를 공개했기 때문에 더 의미있는 학문적 성취라 인정받고 있다. 

![VGG performance](/assets/cand_faces/vgg_performance.png)

VGG는 생각보다 간단한 구조로 되어있다. 일반적인 CNN의 구조처럼 convolution layer와 pooling layer로 구성된 세트가 여러겹 쌓여 마지막에는 4096개의 아이템을 가진 벡터가 출력되고 이를 바탕으로 분류를 수행한다. 

![VGG16](/assets/cand_faces/vgg16.png)

### Transfer learning 실습 방향
이번 실습에서는 때가 때이니만큼 대한민국 19대 주요 대선주자들의 얼굴을 실습 데이터로 사용해보았다. 구글에서 후보 이름으로 각각 500장 씩 이미지를 가져왔다. 간혹 다른 후보가 들어가있거나, 단체 사진 등으로 얼굴을 명확히 인식할 수 없는 경우에는 수작업으로 데이터를 걸러냈다. 데이터 수집은 어느 github의 능력자 덕에 노가다를 피했다. <a href="https://github.com/jonnyhsy/imagecrawler">github.com/jonnyhsy</a>에 들어가 repo를 다운받고 설명에 쓰인대로 명령어를 돌리면 된다. 한글도 잘 돌아간다. 덕분에 시간을 엄청 절약했다.

![주요 대선 후보 5인](/assets/cand_faces/cands.png)

위에서 소개한 4가지 방향에 비추어봤을 때 어떻게 transfer learning을 사용해보면 좋을까? 일단 데이터 사이즈는 한 클래스당 약 380개(500개에서 노이즈 제거) 정도이므로, 데이터셋은 작다. ImageNet에 사용된 데이터셋과 얼마나 유사한지를 판단해야 하는데, 아래 그림과 같이 ImageNet은 일반적인 사물에 대한 분류라면, 이번 실습은 사람 얼굴을 판별하는 것이므로 아주 비슷하다고는 볼 수 없겠다. 따라서 1번과 3번을 모두 시도해보는 것으로 결정했다.

![ImageNet data](/assets/cand_faces/imagenet.png)

<hr>

### Preprocessing
수집한 이미지를 VGG에 넣기 전에 반드시 전처리를 거쳐야 한다. 실습에 사용한 VGG16은 224x224 픽셀의 이미지를 처리한다. 이를 위해 이미지 중심을 기준으로 정사각형으로 이미지를 자르는데, 이 과정에서 간혹 얼굴이 날아갈 수도 있다. 이미지를 크롭한 후 눈으로 다시 확인하여 머리나 얼굴이 잘려 알아보기 어려운 이미지는 제외했다. 아직 할 줄 몰라서 못하긴 했지만, 나중에는 얼굴만 자동으로 추출해서 MNIST처럼 깔끔한 데이터셋을 만드는 과정을 앞단에 넣어주면 매우 편할 것 같다. (일단 이렇게 잘린 이미지가 몇장 안되긴 해서 패스.)

![잘린 이미지들](/assets/cand_faces/headless.png)

<hr>

### Model & Evaluation Metric Design
이번 실습에서는 4가지 모델을 만들고, Accuracy를 기준으로 성능을 평가해보자. 구상한 모델은 다음과 같다.

1. [M1] Transfer Learning: VGG16-relu6 + 2 * FC  
 * VGG16을 고정된 피쳐 추출기로 사용 (전략 1)
 * relu6의 결과를 인풋으로 2개의 FC에 전달 (FC - dropout - FC2)

2. [M2] Transfer Learning: VGG16-pool3 + 2 * FC  
 * VGG16의 early 결과인 pool3를 피쳐로 추출 (전략 3)
 * 2개의 FC에 전달 (FC - dropout - FC2)

3. [M3] Transfer Learning: VGG16-pool3 + CNN * 2 + 2 * FC
 * VGG16의 early 결과인 pool3를 피쳐로 추출 (전략 3)
 * 추가로 2개의 Convolutional Layer (3x3 filter)
 * 2개의 FC layer를 더해 학습 (FC - dropout - FC2)

3. [M4] Convolutional Neural Network: 2 * CNN + 2 * FC  
 * 5x5 필터 + 2x2 풀링으로 구성된 CNN을 2개 중첩
 * 2개의 FC에 전달 (FC - dropout - FC2)

<hr>

### How to get VGG16
앞서 언급했듯, VGG16을 내려받는다는 것은 공개된 VGG16의 `weight`를 가져오는 것과 같다. 공개된 데이터 저장소에 올려진 `.npy` 파일을 아래와 같이 내려받는다. 용량이 500메가쯤 되서 시간이 좀 걸린다.

{% highlight python %}

vgg_dir = 'vgg/'
if not isdir(vgg_dir):
    os.makedirs(vgg_dir)
    
class DLProgress(tqdm):
    last_block = 0
    
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
        
if not isfile(vgg_dir + "vgg16.npy"):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='VGG16 Params') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
            vgg_dir + 'vgg16.npy',
            pbar.hook
        )
else:
    print("VGG16 already exists.")

{%endhighlight%}

<hr>

### Preprocess through VGG16
정리한 이미지를 VGG16에 집어넣어, 피쳐를 뽑는다. 코드 중간에 run(vgg.relu6, ...)라고 된 부분이 VGG16의 중간과정인 relu6에서 피쳐를 얻는 부분이다. 여기를 vgg.pool3로 바꾸면 [M2]에서 사용할 인풋 데이터를 얻을 수 있다.

{% highlight python %}

def process_images_thru_vgg(classes, batch_size=10):
    start_time = datetime.now()

    labels = []
    file_list = []
    batch = []

    codes = None

    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)

        for each in classes:
            print("Starting {} images".format(each))
            class_path = image_dir + each
            files = os.listdir(class_path)
            files.remove('.DS_Store')
            for ii, file in enumerate(files, 1):
                file_list.append(each + "/" + file)
                # Add images to the current batch
                img = utils.load_image(os.path.join(class_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))
                labels.append(each)

                # Running the batch through the network to get the codes
                if ii % batch_size == 0 or ii == len(files):
                    images = np.concatenate(batch)

                    feed_dict = {input_: images}
                    ## run vgg.relu6 to extract fixed features
                    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                    # Here I'm building an array of the codes
                    if codes is None:
                        codes = codes_batch
                    else:
                        codes = np.concatenate((codes, codes_batch))

                    # Reset to start building the next batch
                    batch = []
                    print('{} images processed'.format(ii))

    # write files to file
    import csv
    with open('files_v2', 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(file_list)

    # write codes to file
    with open('codes_v2', 'w') as f:
        codes.tofile(f)

    # write labels to file
    with open('labels_v2', 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(labels)
        
    end_time = datetime.now()
    ts = end_time - start_time
    time_spent = int(ts.total_seconds())
    print("processing images thru vgg16: {} secs".format(time_spent))

{% endhighlight%}

총 1,821장의 이미지를 넣고 돌렸다. [M1]에서는 927초, [M2]에서는 704초가 걸렸다. [M1]에서는 이후에 convolution layer를 2개 더 거치고 relu까지 하다보니 시간이 더 걸리는 듯 하다. 전처리에 10~13분 정도 소요되므로, 매번 돌릴때마다 반복하지 않기 위해 파일로 중간 결과를 저장해두었다.

![M1](/assets/cand_faces/prep_thru_vgg.png)

### values through VGG
[M1]에서 VGG16에 이미지를 집어넣으면, 이미지 하나당 4096개 소수점으로 된 벡터가 출력된다. 이를 보기쉽도록 64x64 이미지로 표현하면 아래와 같다.

![원본이미지](/assets/cand_faces/vgg_train_original.png)![처리된 결과](/assets/cand_faces/vgg_train_res.png)

VGG16으로 이미지가 처리되는 중간결과는 Tensorboard를 사용해 확인할 수 있으나.. Tensorboard를 사용하는 방법 및 결과는 다음에 공부를 해서 올리기로 한다.

여튼, relu6를 통한 결과는 이미지로 표현하기 위한 결과가 아니므로 보기에 이해가 가지 않는 것이 당연하다. [M2]를 통해 얻은 결과는 [M1]보다 훨씬 저차원의 데이터로 하나의 벡터에 200,704개의 소수가 들어있다.

![M1](/assets/cand_faces/prep_thru_vgg.png)

<hr>

### Adding my own classifier
VGG16을 통해 이미지의 피쳐를 추출했으므로, 이제 내가 원하는대로 분류기를 만들어 끝에 붙이기만 하면 된다. fully-connected 레이어를 2개를 이어 붙이고, 첫 레이어 뒤에는 dropout을 붙여 overfitting을 방지해본다. 

{% highlight python %}

def model_run(epochs, keep_prob, checkpoint_dir):

    start_time = datetime.now()

    training_loss_list = []
    val_acc_list = []

    if not isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    iteration = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for x, y in get_batches(train_x, train_y):
                feed = {inputs_: x,
                        labels_: y,
                        keep_prob_0: keep_prob}
                loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                training_rec = {'epoch': e+1, 'training_loss': loss}
                training_loss_list.append(training_rec)

                print("Epoch: {}/{}".format(e+1, epochs),
                      "Iteration: {}".format(iteration),
                      "Training loss: {:.5f}".format(loss))
                iteration += 1

                if iteration % 50 == 0:
                    feed = {inputs_: val_x,
                            labels_: val_y,
                            keep_prob_0: 1.0}
                    val_acc = sess.run(accuracy, feed_dict=feed)

                    val_rec = {'epoch': e+1, 'val_acc': val_acc}
                    val_acc_list.append(val_rec)

                    print("Epoch: {}/{}".format(e+1, epochs),
                          "Iteration: {}".format(iteration),
                          "Validation Acc: {:.4f}".format(val_acc))
        saver.save(sess, checkpoint_dir + "cand_faces_cleaned.ckpt")

    end_time = datetime.now()
    ts = end_time - start_time
    time_spent = int(ts.total_seconds())
    print("fine-tuning vgg16 with my dataset: {} secs".format(time_spent))
    
    return training_loss_list, val_acc_list

{% endhighlight%}

<hr>

### Performance Evaluation

#### [M1]
위 함수에 epochs, keep_prob(dropout), checkpoint_dir(저장위치)를 넘겨 분류기 학습을 수행했다. [M1]에 먼저 epochs:300, keep_prob: 50을 넣고 돌렸는데 그 결과는 아래와 같았다.

Epoch: 1/300 Iteration: 0 Training loss: 6.09721  
Epoch: 1/300 Iteration: 1 Training loss: 4.44061  
Epoch: 1/300 Iteration: 2 Training loss: 3.81262  
...  
Epoch: 300/300 Iteration: 2997 Training loss: 0.00034  
Epoch: 300/300 Iteration: 2998 Training loss: 0.00028  
Epoch: 300/300 Iteration: 2999 Training loss: 0.00032  
Epoch: 300/300 Iteration: 3000 Validation Acc: 0.7912  
fine-tuning vgg16 with my dataset: 60 secs  

60초만에 validation accuracy가 거의 80%가 나왔다. 바이너리 분류 문제도 아니고 분류 클래스가 5개이기 때문에 꽤 좋은 성능을 보였다고 판단했다. 300 epochs를 돌렸는데 60초가 걸렸으므로, 1 epoch당 0.2초가 걸린 셈이다. GPU도 아닌 CPU에서 돌렸다. training_loss가 빠르고 안정적으로 떨어지는 것으로 보아 학습이 잘 되었다. 또 epoch을 많이 돌릴 것도 없이 100번 정도만 돌려도 같은 결과를 얻을 수 있었던 것 같다. 최종 테스트셋을 대상으로는 Test accuracy: 0.7432를 기록했다.

![M1_res](/assets/cand_faces/m1_res.png)

후보별 accuracy를 보면 0.9에서 0.6 정도 사이에 분포한 가운데, 심상정 > 안철수 > 홍준표 순으로 높았다. 심상정 후보만 성이 달라 더 정확도가 높지 않았을까 추측해본다.

![M1_acc](/assets/cand_faces/m1_acc.png)

맞춘 케이스와 못맞춘 케이스를 몇개 살펴보자.
![M1_정답](/assets/cand_faces/m1_correct.png)
![M1_오답](/assets/cand_faces/m1_incorrect.png)

왜 틀린지는 잘 모르겠다..


#### [M2]
M1의 결과에 고무되어 M2의 결과를 더 기대했다. 데이터셋이 ImageNet 데이터셋과 조금 상이하다고 생각했기 때문에 더 저차원의 필터를 가져와서 돌리면 좋겠다고 생각했다. 

Epoch: 1/100 Iteration: 0 Training loss: 881.21344  
Epoch: 1/100 Iteration: 1 Training loss: 17106.92773  
Epoch: 1/100 Iteration: 2 Training loss: 21967.27539  
...   
Epoch: 100/100 Iteration: 998 Training loss: 1.60844  
Epoch: 100/100 Iteration: 999 Training loss: 1.60662  
Epoch: 100/100 Iteration: 1000 Validation Acc: 0.2692  
fine-tuning vgg16 with my dataset: 758 secs  

하지만 저차원의 데이터를 바로 fully-connected layer에 연결해서인지 성능도 매우 별로였고, 시간도 12배 이상 걸렸다. 특이한 점은 초반에 training loss가 매우 크게 발생했었는데, 저런 수치는 그동안 텐서플로우를 돌려보면서 처음 본 거라 좀 놀랐다. 물론 epoch 몇번을 돈 이후에는 10 이하로 떨어졌지만, M1만큼 낮은 loss까지 떨어지지는 않았다. validation accuracy 역시 초반에 급격히 올라 기대가 컸으나, 30%를 넘어가지 못했다. 최종 테스트셋을 대상으로는 Test accuracy: 0.1967를 기록했다.

![M2_res](/assets/cand_faces/m2_res.png)

더 웃긴 것은 분류 결과였는데..
![M2_acc](/assets/cand_faces/m2_acc.png)
안철수 후보만 모두 맞추고 나머지는 하나도 못 맞춘 이상한 결과가 나왔다. 즉, 안철수 후보의 데이터에만 피팅이 된 비정상적인 결과가 나왔다.

#### [M3]
M3는 초기 계획에는 없었으나, M2가 실패하면서 이를 보완한 모델로 만들어봤다. M2가 너무 저차원 데이터여서 학습이 제대로 되지 않았다는 생각이 들어, M3에는 VGG에서 추출한 저차원 피쳐에 convolution layer를 2개 더하고 FC 레이어에 집어넣는 방식을 취해보았다. 이로서 M1보다는 CNN의 성능 자체는 낮을수 있으나 내가 입력한 데이터셋을 반영해서 M2보다는 나은 결과물을 예상했다.

그러나..

Epoch: 1/20 Iteration: 0 Training loss: 40095.62500  
Epoch: 1/20 Iteration: 1 Training loss: 122916.09375  
Epoch: 1/20 Iteration: 2 Training loss: 259003.64062  
...  
Epoch: 20/20 Iteration: 198 Training loss: 1.60973  
Epoch: 20/20 Iteration: 199 Training loss: 1.60940  
Epoch: 20/20 Iteration: 200 Validation Acc: 0.1703  
fine-tuning vgg16 with my dataset: 3912 secs  

작은 긍정적인 점 하나에 여러 단점이 따라왔다. 장점은 VGG의 앞단 결과물에 내가 만든 CNN을 더해봤다는 시도일 뿐, 학습시간이나 성능 모두 M2에 비해 좋지 않았다. validation accuracy는 epoch을 돌면 돌수록 반대로 떨어졌다. 이번에는 유승민 후보에만 다 맞추고 나머지는 다틀렸다.

![M3_acc](/assets/cand_faces/m3_acc.png)

![M3_res](/assets/cand_faces/m3_res.png)

게다가 고작 20 epochs를 돌렸을 뿐인데 3,912초, 거의 48분이 걸렸다. M1이 300 epochs에 60초가 걸렸다는 걸 생각해보면 너무나 느린 속도다. GPU를 사용하면 더 빨라지겠으나, loss가 줄어드는 패턴을 보면 epoch을 많이 돌린다고 해서 나아질 것 같지는 않다. 사실 CS231에서 4가지 방법에 대해 소개하면서 데이터셋이 많은 경우에만 CNN `weight`을 업데이트하라고 했는데, 괜히 한 얘기가 아니다 싶다.

#### [M4]
마지막으로 VGGNet을 사용하지 않고 내가 간단히 만든 CNN이 출력한 결과를 보자. 

Epoch: 1/10 Iteration: 0 Training loss: 1.71266  
Epoch: 1/10 Iteration: 1 Training loss: 1.69945  
Epoch: 1/10 Iteration: 2 Training loss: 1.72151  
...  
Epoch: 100/100 Iteration: 998 Training loss: 1.73242  
Epoch: 100/100 Iteration: 999 Training loss: 1.69291  
Epoch: 100/100 Iteration: 1000 Validation Acc: 0.2308  
NN with my dataset: 2201 secs  

일단 training loss 자체는 M2, M3에 비해 낮고 안정적이지만, loss가 전혀 줄어들지 않았다. Validation accuracy는 20%를 넘겼지만, Test셋을 대상으로는 Test accuracy: 0.1366를 기록했다. 언더피팅과 오버피팅이 절묘하게 섞인, 그런 말도 안되는 결과가 나온 것 같다. 

![M4_acc](/assets/cand_faces/m4_acc.png)

![M4_res](/assets/cand_faces/m4_res.png)

분류 결과도 유승민 후보만 100%가 나오고 나머지는 다 틀린 것으로 봐서 학습 자체가 제대로 안된듯 하다. 역시 괜찮은 CNN모델을 처음부터 만들기에는 데이터셋 자체가 너무 적었던 것 같다.

<hr>

### 마무리
대선주자 얼굴 분류기를 통해 VGGNet 기반 transfer learning을 테스트해보았다. 데이터셋이 다 합해서 2,000개도 되지 않는 초소형 데이터셋이었지만, VGGNet에 간단한 fully-connected layer를 더하는 것만으로도 나쁘지 않은 성능을 얻은 듯 하다. 대선주자 얼굴을 분류해서 어디 쓸데가 있는 것은 아니지만, 현재 하는 회사 업무처럼 더 많은 이미지를 얻을 수 있는 곳에서 더 유용하게 쓸 수 있지 않을까 기대가 된다.