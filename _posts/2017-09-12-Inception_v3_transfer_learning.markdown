---
layout:     post
title:      "Inception V3: Transfer Learning"
date:       2017-09-12 00:00:00
author:     "Jun"
img: 20170912.png
tags: [python, deep learning, image]
---

## Inception v3를 활용한 Transfer Learning
<br>

### Transfer Learning: Inception v3
요새 패스트캠퍼스에서 딥러닝 영상인식 수업을 듣고 있다. 두번째 강의에서 구글이 만든 Inception v3 모델을 사용한 Transfer Learning 코드를 살펴보았는데, 복습도 할겸 일부 코드를 들고와서 Jupyter notebook용으로 옮겨보았다.   
(원본 코드: <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py">링크</a>)

Inception 구조는 2014년 구글에서 펴낸 Going deeper with convolutions, Szegedy et al. (2014) 논문에 나오는 CNN 구조로, GoogleLeNet에 사용되어 그해 이미지넷 대회에서 VGG 모델을 누르고 우승한다. 기존의 CNN 모델들이 인풋 이미지에 같은 크기의 Convolution 필터를 하나씩 계속 덧대는 구조라면, Inception은 한번에 여러 크기의 필터를 동시에 사용한다. 덕분에 (상대적으로) 간단한 VGG 모델에 비해 Inception 구조는 직관적으로 잘 와닿지 않는 느낌이었다. 이미지를 처리할 때 여러 크기의 필터를 복합적으로 사용해 이미지의 특징을 더 잘 잡아낸다고 개념적으로 이해하고 있다.

![Inception 모듈 구조(https://adeshpande3.github.io/assets/GoogLeNet3.png)](/assets/materials/20170912/inception_module.png)

여튼 <a href="http://nicolovaligi.com/history-inception-deep-learning-architecture.html">Short history of the Inception deep learning architecture</a>에 의하면, 구글에서 2015년에 다시 한번 Rethinking the inception architecture for computer vision, Szegedy et al. (2015) 이라는 논문을 낸다. 여기서 2014년 논문에서 구현한 Inception v1을 개량하고 v2를 거쳐 기존 GoogleLeNet의 성능을 압도하는 v3를 만들어낸다. 3x3보다 큰 필터는 그보다 더 작은 필터 여러개로 더 효율적으로 표현할 수 있으며, 심지어 7x7 필터는 1x7과 7x1 컨볼루션 레이어로 대체하는 것을 제안한다.

2016년에는 v4, Inception-ResNet까지 확장된 논문이 나오는데, 일단 이 포스팅은 v3에 관한 것이니 여기까지만 하도록 한다..

### 라이브러리 임포트
일단 필요한 필요한 라이브러리를 불러온다. tqdm과 같은 라이브러리는 원문 코드에는 사용되지 않았다.  
또 원문 코드는 터미널 환경에서 유저가 하이퍼파라미터를 설정하도록 FLAGS를 사용했는데, 노트북 환경에서는 이를 제외하였다.


```python
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
from tqdm import tqdm
import time

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import matplotlib.pyplot as plt
%matplotlib inline
```

<hr>

### 하이퍼파라미터 설정 (일부)
모델 경로, 보틀넥 텐서 등 일부 하이퍼파라미터를 설정한다. 이 코드에서는 보틀넥(Bottleneck)이라는 게 등장하는데, 이는 이미지가 v3를 거쳐 나온 일종의 중간결과라고 생각하면 된다. Transfer Learning에서는 Inception이나 VGG같은 이미 검증된 모델의 네트워크 파라미터를 그대로 사용하고 끝단의 classifier만 학습한다. 그러므로 (당연히) 이미지를 고정된 네트워크에 넣어 텐서로 변환하는 `전처리`과정을 매번 에폭을 돌 때마다 반복할 이유가 전혀 없다. 그래서 본 코드에서는 이를 보틀넥에 저장하는데, 그 텐서의 크기가 2048이 된다.


```python
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
```

<hr>

### 데이터 전처리
이미지 전처리에 사용되는 함수가 엄청 많다. 보틀넥을 만들고 관리하고 불러오는 잡다한 함수가 서로 꼬리를 물고 얽혀있다.  
- `def create_image_lists` - 이미지 디렉토리에서 인풋 데이터를 찾아 데이터로 변환한다.  
- `class TqdmUpTo` - 파일 다운로드할 때 예쁜 프로그레스바 띄워주는 도구  
- `def maybe_download_and_extract` - Inception 그래프가 없다면 다운로드받는다.  
- `def create_inception_graph` - 파일에서 텐서플로우 그래프를 읽어 리턴한다.  
- `def should_distort_images` - 이미지 처리시에 좌우 변환 등 왜곡을 줄지 결정한다.  
- `def ensure_dir_exists` - 디렉토리 경로가 있는지 체크하고 없으면 만든다.  
- `def cache_bottlenecks` - 만든 보틀넥을 임시 저장한다.  
- `def get_or_create_bottleneck` - 보틀넥을 만든다.  
- `def get_bottleneck_path` - 만든 보틀넥 파일의 저장경로를 가져온다.  
- `def create_bottleneck_file` - 보틀넥을 파일에 저장한다.  
- `def run_bottleneck_on_image` - 이미지에서 보틀넥을 추출한다.  


```python
def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """이미지 디렉토리에서 인풋 데이터를 찾아 데이터로 변환한다"""
    
    ## image_dir가 존재하지 않는다면 오류 출력
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    
    result = {}
    
    ### image_dir 내 하위 디렉토리(label)를 가져온다 
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
            
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        
        ## 파일이 없거나 데이터가 작으면 예외 처리
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print("WARNING: Folder has less than 20 images, which may cause issues.")
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print("WARNING: Folder {} has more than {} images. Some images will never be selected".format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        
        ## 트레이닝 / 밸리데이션 / 테스트셋으로 나눈다.
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                               (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                              (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
        
    return result
```


```python
## 데이터를 다운로드받을 때 사용할 Tqdm 클래스를 정의한다.
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
```


```python
def maybe_download_and_extract():
    dest_directory = model_dir
    ensure_dir_exists(dest_directory)
    
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    
    if not os.path.exists(filepath):
        
        print("그래프 파일이 없습니다. 다운로드를 시작합니다.")
        
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=DATA_URL) as t:
            urllib.request.urlretrieve(DATA_URL, filepath, reporthook=t.update_to, data=None)
        
        statinfo = os.stat(filepath)
        print("다운로드 완료: ", filename, statinfo.st_size, 'bytes.')
    else:
        print("그래프 파일이 이미 존재합니다.")
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
```


```python
def create_inception_graph():
    """
    저장된 GraphDef 파일에서 그래프를 만들고
    Graph 오브젝트를 리턴한다.
    """
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
    
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor
```


```python
def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
    """이미지 데이터에 변화를 줄지 결정한다."""
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or (random_brightness != 0))
```


```python
def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
```


```python
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, \
                                         bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    print('{} bottleneck files created'.format(how_many_bottlenecks))
```


```python
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, \
                             bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                    bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        print('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values
```


```python
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'
```


```python
def get_image_path(image_lists, label_name, index, image_dir, category):
    
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.', label_name, category)
        
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    
    return full_path
```


```python
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir,
                          category, sess, jpeg_data_tensor, bottleneck_tensor):
    print('보틀넥 파일 생성 시작 - {}'.format(bottleneck_path))
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    
    if not gfile.Exists(image_path):
        tf.logging.fata('File does nto exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    except:
        raise RuntimeError('파일 처리 중 에러 발생: %s' % image_path)
        
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)
```


```python
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(
        bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values
```

<hr>

### Transfer Learning
이제 비로소 모델에 마지막 classifer를 붙이는 작업을 한다.  
- `def add_final_training_ops` - 마지막 레이어를 정의하고 loss과 optimizer를 정의한다.  
- `def add_evaluation_step` - 성능 평가를 위한 지표 (accuracy)를 정의한다.  
- `def get_random_cached_bottlenecks` - 랜덤 혹은 전체 보틀넥을 가져와 모델에 집어넣는다.  


```python
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')
        
        ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')
        
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.01)
            
            layer_weights = tf.Variable(initial_value, name='final_weight')
        
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            
    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)
        
    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)
```


```python
def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return evaluation_step, prediction
```


```python
def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir,
                                 jpeg_data_tensor, bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # 샘플링한 보틀넥을 가져온다.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                      image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                image_index, image_dir, category,
                                                bottleneck_dir, jpeg_data_tensor,
                                                bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # 보틀넥을 모두 가져온다.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                      image_index, image_dir, category,
                                                      bottleneck_dir, jpeg_data_tensor,
                                                      bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


```

<hr>

### 하이퍼파라미터 설정 (모델 관련)
파일 경로 및 모델 구현에 필요한 일부 파라미터를 설정한다. 테스트에 사용할 데이터로, 패스트캠퍼스 강의에서 제공한 3가지 종류의 고양이 이미지를 사용해보았다.


```python
image_dir = 'cat_photos'
output_graph = '/tmp/output_graph.pb'
output_labels = '/tmp/output_labels.txt'
summaries_dir = '/tmp/retrain_logs'
how_many_training_steps = 300
learning_rate = 0.01
testing_percentage = 10
validation_percentage = 10
eval_step_interval = 10
train_batch_size = 100
test_batch_size = -1
validation_batch_size = 100
print_misclassified_test_images = False
model_dir = '/tmp/imagenet'
bottleneck_dir = '/tmp/bottleneck'
final_tensor_name = 'final_result'
flip_left_right = False
random_crop = 0
random_scale = 0
random_brightness = 0 
log_frequency = 10
log_device_placement = False
```

<hr>

### 모델 다운로드 및 준비


```python
## inception_v3를 다운받아 압축을 푼다.
maybe_download_and_extract()
```

    그래프 파일이 이미 존재합니다.



```python
## 그래프와 보틀넥 텐서, 이미지데이터 텐서, 리사이즈 이미지 텐서를 불러온다.
graph, bottleneck_tensor, jpeg_data_tensor, resize_image_tensor = (create_inception_graph())
```


```python
## 재학습할 폴더를 가져와서 레이블화한다.
image_lists = create_image_lists(image_dir, testing_percentage, validation_percentage)
```

    Looking for images in 'chartreux'
    Looking for images in 'persian'
    Looking for images in 'ragdoll'



```python
class_count = len(image_lists.keys())
```


```python
if class_count == 0:
    print('이미지가 해당 경로에 없습니다: ' + image_dir)
    
elif class_count == 1:
    print('해당 경로에 클래스가 1개만 발견되었습니다: ' + image_dir + ' - 분류를 위해 2개 이상의 클래스가 필요합니다.')
    
else:
    print("클래스가 2개 이상 있습니다. 학습을 시작합니다.")
```

    클래스가 2개 이상 있습니다. 학습을 시작합니다.



```python
## Image distortion // 현재 설정: False
do_distort_images = should_distort_images(flip_left_right, random_crop, random_scale, random_brightness)
```

<hr>

### 모델 학습 시작
텐서플로우 세션을 열고, 보틀넥 파일을 가져오고, Inception_v3 끝에 학습시킬 마지막 classifier 레이어를 붙인다.  
이후 에폭을 돌면서 이미지를 넣어 Training / Validation Accuracy를 산출한다.  
학습이 완료되면 테스트셋 이미지를 넣어 최종 테스트셋 정확도를 평가한다.


```python
acc_list = []

with tf.Session(graph=graph) as sess:
    if do_distort_images:
        (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortion(
            flip_left_right, random_crop, random_scale, random_brightness)
    else:
        cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
    
    ## 네트워크의 끝에 우리가 원하는 분류 레이어를 붙인다.
    (train_step, cross_entropy, bottleneck_input, 
     ground_truth_input, final_tensor) = add_final_training_ops(len(image_lists.keys()), 
                                                                final_tensor_name, 
                                                                bottleneck_tensor)
        
    ## 정확도 평가를 위한 새로운 오퍼레이션
    evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)
    
    ## 가중치 초기화
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(how_many_training_steps):
        
        ## 보틀넥과 정답지를 준비한다.
        if do_distort_images:
            (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(
                sess, image_lists, train_batch_size, 'training', image_dir, distorted_jpeg_data_tensor,
                distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
            (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                sess, image_lists, train_batch_size, 'training', bottleneck_dir, image_dir,
                jpeg_data_tensor, bottleneck_tensor)
        
        
        ## 보틀넥과 정답지를 모델에 집어넣어 학습시킨다.
        _ = sess.run(
            [train_step],
            feed_dict={bottleneck_input: train_bottlenecks,
                      ground_truth_input: train_ground_truth})
    
        ## 특정 구간마다 트레이닝 정확도와 cross entropy 로그, 밸리데이션 정확도를 출력한다.
        is_last_step = (i + 1 == how_many_training_steps)
        if (i % eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict = {bottleneck_input: train_bottlenecks,
                            ground_truth_input: train_ground_truth})
            
            print('%s: Step %d: Train accuracy = %.1f%%'% (datetime.now(), i, train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
            
            validation_bottlenecks, validation_ground_truth, _ = (
                get_random_cached_bottlenecks(
                    sess, image_lists, validation_batch_size, 'validation', bottleneck_dir,
                    image_dir, jpeg_data_tensor, bottleneck_tensor))
            
            validation_accuracy = sess.run(
                evaluation_step,
                feed_dict = {bottleneck_input: validation_bottlenecks,
                            ground_truth_input: validation_ground_truth})
            print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)'% (datetime.now(), i,
                                                                       validation_accuracy * 100, 
                                                                       len(validation_bottlenecks)))
            
            ## 시각화를 위해 로그를 한벌 더 저장한다.
            acc_list.append({"epoch": i, "train_accuracy": train_accuracy, "validation_accuracy": validation_accuracy})
    
    ## 테스트셋에 사용할 보틀넥과 정답지를 가져온다.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(sess, image_lists, test_batch_size, 'testing', bottleneck_dir,
                                     image_dir, jpeg_data_tensor, bottleneck_tensor))
    
    ## 테스트셋 정확도와 예측 분류값을 가져온다.
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction], 
        feed_dict={bottleneck_input: test_bottlenecks,
                  ground_truth_input: test_ground_truth})
    print('최종 학습 정확도 = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))
    
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_tensor_name])
    with gfile.FastGFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')
```

    2017-09-12 22:57:31.651697: Step 0: Train accuracy = 90.0%
    2017-09-12 22:57:31.651842: Step 0: Cross entropy = 0.971989
    2017-09-12 22:57:31.728077: Step 0: Validation accuracy = 50.0% (N=100)
    2017-09-12 22:57:32.482163: Step 10: Train accuracy = 97.0%
    2017-09-12 22:57:32.482298: Step 10: Cross entropy = 0.531070
    2017-09-12 22:57:32.557460: Step 10: Validation accuracy = 69.0% (N=100)
    2017-09-12 22:57:33.302371: Step 20: Train accuracy = 98.0%
    2017-09-12 22:57:33.302512: Step 20: Cross entropy = 0.378725
    2017-09-12 22:57:33.373815: Step 20: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:34.101410: Step 30: Train accuracy = 98.0%
    2017-09-12 22:57:34.101545: Step 30: Cross entropy = 0.314494
    2017-09-12 22:57:34.173382: Step 30: Validation accuracy = 92.0% (N=100)
    2017-09-12 22:57:34.913235: Step 40: Train accuracy = 99.0%
    2017-09-12 22:57:34.913372: Step 40: Cross entropy = 0.230641
    2017-09-12 22:57:34.985273: Step 40: Validation accuracy = 96.0% (N=100)
    2017-09-12 22:57:35.723743: Step 50: Train accuracy = 99.0%
    2017-09-12 22:57:35.723963: Step 50: Cross entropy = 0.220344
    2017-09-12 22:57:35.795600: Step 50: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:36.530771: Step 60: Train accuracy = 99.0%
    2017-09-12 22:57:36.530906: Step 60: Cross entropy = 0.187575
    2017-09-12 22:57:36.602992: Step 60: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:37.342191: Step 70: Train accuracy = 100.0%
    2017-09-12 22:57:37.342325: Step 70: Cross entropy = 0.172496
    2017-09-12 22:57:37.415287: Step 70: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:38.148254: Step 80: Train accuracy = 100.0%
    2017-09-12 22:57:38.148388: Step 80: Cross entropy = 0.166109
    2017-09-12 22:57:38.220542: Step 80: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:38.949337: Step 90: Train accuracy = 98.0%
    2017-09-12 22:57:38.949470: Step 90: Cross entropy = 0.150583
    2017-09-12 22:57:39.020142: Step 90: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:39.763395: Step 100: Train accuracy = 99.0%
    2017-09-12 22:57:39.763533: Step 100: Cross entropy = 0.121292
    2017-09-12 22:57:39.835280: Step 100: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:40.562501: Step 110: Train accuracy = 96.0%
    2017-09-12 22:57:40.562636: Step 110: Cross entropy = 0.157512
    2017-09-12 22:57:40.633806: Step 110: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:41.378612: Step 120: Train accuracy = 98.0%
    2017-09-12 22:57:41.378750: Step 120: Cross entropy = 0.112991
    2017-09-12 22:57:41.455019: Step 120: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:42.182392: Step 130: Train accuracy = 99.0%
    2017-09-12 22:57:42.182529: Step 130: Cross entropy = 0.107235
    2017-09-12 22:57:42.254900: Step 130: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:42.986459: Step 140: Train accuracy = 100.0%
    2017-09-12 22:57:42.986595: Step 140: Cross entropy = 0.102031
    2017-09-12 22:57:43.058726: Step 140: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:43.786503: Step 150: Train accuracy = 98.0%
    2017-09-12 22:57:43.786641: Step 150: Cross entropy = 0.106734
    2017-09-12 22:57:43.858639: Step 150: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:44.587175: Step 160: Train accuracy = 99.0%
    2017-09-12 22:57:44.587308: Step 160: Cross entropy = 0.101420
    2017-09-12 22:57:44.662408: Step 160: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:45.409826: Step 170: Train accuracy = 97.0%
    2017-09-12 22:57:45.409960: Step 170: Cross entropy = 0.115423
    2017-09-12 22:57:45.482110: Step 170: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:46.211915: Step 180: Train accuracy = 99.0%
    2017-09-12 22:57:46.212053: Step 180: Cross entropy = 0.074815
    2017-09-12 22:57:46.283751: Step 180: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:47.015308: Step 190: Train accuracy = 100.0%
    2017-09-12 22:57:47.015443: Step 190: Cross entropy = 0.071336
    2017-09-12 22:57:47.087443: Step 190: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:47.820748: Step 200: Train accuracy = 100.0%
    2017-09-12 22:57:47.820886: Step 200: Cross entropy = 0.067409
    2017-09-12 22:57:47.891961: Step 200: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:48.625402: Step 210: Train accuracy = 100.0%
    2017-09-12 22:57:48.625540: Step 210: Cross entropy = 0.062077
    2017-09-12 22:57:48.697570: Step 210: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:49.427996: Step 220: Train accuracy = 100.0%
    2017-09-12 22:57:49.428132: Step 220: Cross entropy = 0.078230
    2017-09-12 22:57:49.505181: Step 220: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:50.232927: Step 230: Train accuracy = 100.0%
    2017-09-12 22:57:50.233061: Step 230: Cross entropy = 0.068900
    2017-09-12 22:57:50.304175: Step 230: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:51.033848: Step 240: Train accuracy = 100.0%
    2017-09-12 22:57:51.033982: Step 240: Cross entropy = 0.077747
    2017-09-12 22:57:51.105312: Step 240: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:51.838966: Step 250: Train accuracy = 100.0%
    2017-09-12 22:57:51.839107: Step 250: Cross entropy = 0.052831
    2017-09-12 22:57:51.910628: Step 250: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:52.646291: Step 260: Train accuracy = 100.0%
    2017-09-12 22:57:52.646428: Step 260: Cross entropy = 0.064428
    2017-09-12 22:57:52.717543: Step 260: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:53.451311: Step 270: Train accuracy = 100.0%
    2017-09-12 22:57:53.451445: Step 270: Cross entropy = 0.050011
    2017-09-12 22:57:53.525890: Step 270: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:54.267649: Step 280: Train accuracy = 100.0%
    2017-09-12 22:57:54.267786: Step 280: Cross entropy = 0.056949
    2017-09-12 22:57:54.340720: Step 280: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:55.068551: Step 290: Train accuracy = 100.0%
    2017-09-12 22:57:55.068688: Step 290: Cross entropy = 0.060681
    2017-09-12 22:57:55.140574: Step 290: Validation accuracy = 100.0% (N=100)
    2017-09-12 22:57:55.797798: Step 299: Train accuracy = 100.0%
    2017-09-12 22:57:55.797934: Step 299: Cross entropy = 0.046153
    2017-09-12 22:57:55.870064: Step 299: Validation accuracy = 100.0% (N=100)
    최종 학습 정확도 = 100.0% (N=10)
    INFO:tensorflow:Froze 2 variables.
    Converted 2 variables to const ops.

<hr>

### 정확도 시각화
중간에 따로 저장한 트레이닝과 벨리데이션 정확도가 에폭에 따라 얼마나 개선되는지 살펴보았다.  
몇 에폭 지나지 않아 밸리데이션 정확도가 1을 찍는 것으로 보아 학습이 빠르게 잘 되었으며, 굳이 1000번 돌리지 않고 약 300번 정도에서 끊었다.


```python
import pandas as pd
acc_df = pd.DataFrame.from_dict(acc_list)
acc_df.set_index('epoch', inplace=True)
```


```python
f, ax = plt.subplots(figsize=(10, 5))
acc_df.plot(ax=ax)
plt.show()
```


![에폭에 따른 정확도 차트](/assets/materials/20170912/retrain_36_0.png)

<hr>

### 추론
마지막 레이어까지 모두 학습이 끝났다. 이제는 새로운 이미지를 잘 분류하는지 라이브에 태워본다.


```python
imagePath = 'tmp/test_chartreux.jpg'                                      
modelFullPath = '/tmp/output_graph.pb'                                    
labelsFullPath = '/tmp/output_labels.txt'                                 


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
```


```python
def run_inference_on_image(imagePath):
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('파일이 존재하지 않습니다: %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()

    with tf.Session() as sess:

        ## 학습이 끝난 마지막 소프트맥스 텐서를 가져온다.
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        ## 이미지 데이터를 네트워크 맨 앞에 넣어 분류를 실행한다.
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        ## [[]]로 중첩된 어레이가 떨어지는데 np.squeeze로 하나의 어레이로 만든다.
        predictions = np.squeeze(predictions)
        
        ## 가장 높은 확률 값을 가진 5개를 선택한다. 여기서는 클래스가 3개 뿐이라 3개만 출력된다.
        top_k = predictions.argsort()[-5:][::-1]
        print(top_k)
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer
```



![라이브 테스트1: chartreux](/assets/materials/20170912/test_chartreux.jpg)

```python
run_inference_on_image('tmp/test_chartreux.jpg')
```

    [0 2 1]
    b'chartreux\n' (score = 0.99032)
    b'ragdoll\n' (score = 0.00569)
    b'persian\n' (score = 0.00399)





    "b'chartreux\\n'"


이번에는 우리집 러시안 블루도 잘 분류하는지 살펴보았다. 러시안 블루는 애초 분류 레이블에 없지만, 외형적으로 가장 유사한 샤트룩스가 선택된 것으로 보아 학습이 꽤 잘되었다고 평가할 수 있겠다. 또, 아이폰에서 바로 찍은 큰 JPG을 모델이 집어넣고 돌렸는데 바로 잘 돌아가는 것으로 보아.. 라이브 환경에 돌아가는 모델 구현에 참고할 만한 점이 많은 코드라 하겠다.
![라이브 테스트2: russian blue](/assets/materials/20170912/test_russian_blue.jpg)


```python
run_inference_on_image('tmp/test_russian_blue.jpg')
```

    [0 2 1]
    b'chartreux\n' (score = 0.94632)
    b'ragdoll\n' (score = 0.03988)
    b'persian\n' (score = 0.01380)





    "b'chartreux\\n'"
