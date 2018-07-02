---
layout:     post
title:      "Which Continent Does Pyeongchang Belong To?"
date:       2018-02-19 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/cityFinder/worldmap.jpg
---


## Intro

A couple of months ago, I found an abandoned world map in my new office space, and put it on the wall.

![World Map](/assets/cityFinder/worldmap.jpg)

One day I stumbled upon an interesting idea when looking at the atlas. The names of the cities are similar to each other when they are geographically close to each other. For example, Stratford, Wilford, and Bradford are in England, Europe. Pyeongchang, Pyongyang are in the Korean peninsula, Asia. 
![pyeongchang](/assets/cityFinder/pyeongchang.png)

On hearing Pyeongchang and Pyongyang, one could think that they are in the same area without any difficulties. Like this poor pilot (http://www.telegraph.co.uk/news/2017/04/02/airport-mix-up-sees-winter-olympics-delegation-land-pyongyang/).

![poor pilot](/assets/cityFinder/poor_pilot.png)

If we can tell where the cities are just by hearing their names, can we train a machine to do that too?

<br>

## Dataset

To train a complicated Deep Learning model, one would need a lot of data. The more, the merrier! 

Wikipedia provides a list of cities with 100,000+ residents for free, but it wasn't enough. I found a free source that contains as many as 10,000 cities. As it didn't have the continent label I needed, I preprocessed it a little bit to get the dataset I wanted.

You can download it from <a href="https://github.com/junkwhinger/city_finder">my GitHub repo</a>. It looks like the following table, and it has 7,221 cities and their corresponding country and continent information.


{% highlight python %}
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


raw_df = pd.read_csv('city_continent.csv')
raw_df.head()
{% endhighlight %}


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_ascii</th>
      <th>country</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Qal eh-ye</td>
      <td>Afghanistan</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chaghcharan</td>
      <td>Afghanistan</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lashkar Gah</td>
      <td>Afghanistan</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Zaranj</td>
      <td>Afghanistan</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tarin Kowt</td>
      <td>Afghanistan</td>
      <td>Asia</td>
    </tr>
  </tbody>
</table>
</div>



## Approach

The goal is to find and optimize a function that maps the city names to their continents. It is a simple text-classification problem. I used PyTorch which I've heard a lot about recently. 

Here's how I designed the process.  

### 1. Prepare Dataset
split the dataset into train, validation, and test set.

### 2. Define Vocabulary
build a vocabulary to encode text into vectors

### 3. Define LSTM model
uni/bidirectional, number of embeddings, number of hidden units, number of layers, dropout

### 4. Train 
run training

### 5. Evaluate
compare model performance on test set


## 1. Prepare Dataset
`sklearn`'s `train_test_split` really comes in handy when chopping the dataset nicely. I made a function named `prepare_dataset` to split the raw dataset into train, validation, and test set, and to save them in the target directory.


{% highlight python %}
import os
from sklearn.model_selection import train_test_split

def prepare_dataset(path, target_dir):
    df = pd.read_csv(path)[['city_ascii', 'continent']]

    X_train, X_test, y_train, y_test = train_test_split(df.city_ascii, 
                                                        df.continent, 
                                                        test_size=0.1, 
                                                        random_state=42, 
                                                        stratify=df.continent)

    X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=0.1, 
                                                        random_state=42, 
                                                        stratify=y_train)

    trainSet = pd.concat([X_train, y_train], axis=1)
    valSet = pd.concat([X_val, y_val], axis=1)
    testSet = pd.concat([X_test, y_test], axis=1)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    trainSet.to_csv(os.path.join(target_dir, '_train.csv'), index=None)
    valSet.to_csv(os.path.join(target_dir, '_val.csv'), index=None)
    testSet.to_csv(os.path.join(target_dir, '_test.csv'), index=None)


prepare_dataset('city_continent.csv', 'dataset')
{% endhighlight %}

## 2. Define Vocabulary

When building a text-based deep learning model, it's cumbersome to transform text into vectors. We would have to make a word or character set and make a dictionary that maps a word to an integer. And you would want to give space to `<unk>` in case the model stumbles upon an unseen character.

Luckily a group of kind-minded geniuses made a publicly available tool for it, and its name is `torchtext`<a href="https://github.com/pytorch/text">(GitHub)</a>. It is a fantastic tool that enables quick and easy vocab-construction. 


{% highlight python %}
from torchtext import data
{% endhighlight %}

First, we need to define the text and label columns. `CITY` is a column that has city name text. Because it is a sequential data type, I give it `sequential=True`. Sequence processing is easily done just by passing the tokenizer function I defined below.

The column `CONTINENT` is a categorical data type, yet it is not a numerical datatype yet. So I pass `use_vocab=True` for it.


{% highlight python %}
## City name tokenizer
## 'Seoul' -> ['S', 'e', 'o', 'u', 'l']
def tokenizer(text):
    return list(text)

tokenizer("Seoul")

>>    ['S', 'e', 'o', 'u', 'l']
{% endhighlight %}



{% highlight python %}
CITY = data.Field(sequential=True, pad_first=True, tokenize=tokenizer)
CONTINENT = data.Field(sequential=False, use_vocab=True)
{% endhighlight %}

`data.TabularDataset.splits` is a super conveninent function that loads the preprocessed datasets I splited above. `CITY` and `CONTINENT` fields are passed into the function with the actual column name.


{% highlight python %}
train, val, test = data.TabularDataset.splits(
    path='dataset', skip_header=True, train='_train.csv',
    validation='_val.csv', test='_test.csv', format='csv',
    fields=[('city_ascii', CITY), ('continent', CONTINENT)]
)
{% endhighlight %}

`some_field.build_vocab(dataset)` literally builds vocab. Before the construction, our data field doesn't have its vocab.


{% highlight python %}
CITY.vocab

>>    ---------------------------------------------------------------------------
>>
>>    AttributeError                            Traceback (most recent call last)
>>
>>    <ipython-input-10-ae92b3eb66a0> in <module>()
>>    ----> 1 CITY.vocab
>>    
>>
>>    AttributeError: 'Field' object has no attribute 'vocab'
{% endhighlight %}


    



{% highlight python %}
CITY.build_vocab(train, val, test)
CONTINENT.build_vocab(train, val, test)
{% endhighlight %}

And now it does. `field.vocab.stoi` maps characters to their integers, `field.vocab.itos` the other way round. 


{% highlight python %}
CITY.vocab.stoi['A']

>>    26


CITY.vocab.itos[26]

>>    'A'



CONTINENT.vocab.freqs

>>    Counter({'Africa': 1284,
             'Asia': 2334,
             'Europe': 847,
             'North America': 1418,
             'Oceania': 297,
             'South America': 1041})
{% endhighlight %}








Let's turn the loaded datasets into generators which are handy when training a neural network. `data.BucketIterator.splits` uses `sort_key` to group city names that are of similar lengths, which minimizes the number of paddings.


{% highlight python %}
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(16, 16, 16), repeat=False,
    sort_key=lambda x: len(x.city_ascii), device=-1
)
{% endhighlight %}

Here's a vectorized city name and continent label a train iterator would deliver to the model.


{% highlight python %}
sample = next(iter(train_iter))
{% endhighlight %}


{% highlight python %}
list(sample.city_ascii[:, 0].data)

>>    [28, 6, 25, 8, 10, 12, 6]


"".join([CITY.vocab.itos[i] for i in list(sample.city_ascii[:, 0].data)])

>>    'Kipushi'


sample.continent[0].data

>>     3
>>    [torch.LongTensor of size 1]


CONTINENT.vocab.itos[sample.continent[0].data.numpy()[0]]

>>    'Africa'

{% endhighlight %}







## 3. Define LSTM Models
A lot of tutorial codes that I referenced use Class to define their own neural network in PyTorch. I found this approach a little bit confusing at first because I was quite happy with Keras sequential but it has kind of grown on me, and I like how I can generate multiple models just by tweaking the parameters.

`ParameterGrid` from `sklearn` is a handy tool for building a parameter grid. Here I experimented with the number of LSTM layers, dropout in LSTM, dropout in the fully connected layer at the end, and whether the LSTM is bidirectional. 


{% highlight python %}
from sklearn.model_selection import ParameterGrid

grid = {'BATCH_SIZE': [16], 'EMBEDDING_DIM': [100], 'HIDDEN_DIM': [100], 
 'nb_1stm_layers':[1, 2, 3], 'lstm_dropout': [0, 0.5], 
 'lstm_bidirectional': [True, False], 'fc_dropout': [0, 0.5]}

param_grids = list(ParameterGrid(grid))

print("Number of LSTM models to generate: {}".format(len(param_grids)))

>>    Number of LSTM models to generate: 24
{% endhighlight %}



Building such versatile model taught me some lessons.
- it's better to pass `use_gpu` to the model because the Variables need `.cuda()` to run in the GPU environment.
- when the LSTM layer is `bidirectional` the `hidden_dim` in LSTM gets doubled
- if you want to deactivate `dropout`, just pass `0` to it.



{% highlight python %}
import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(1000)


use_gpu = torch.cuda.is_available()


class LSTMModel(nn.Module):
    def __init__(self, batch_size, vocab_size, label_size,
                 embedding_dim, hidden_dim,
                 nb_lstm_layers, lstm_dropout, lstm_bidirectional,
                 fc_dropout, use_gpu):
        super(LSTMModel, self).__init__()
        
        self.use_gpu = use_gpu
        
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.label_size = label_size
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.nb_lstm_layers = nb_lstm_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_bidirectional = lstm_bidirectional
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            num_layers=self.nb_lstm_layers, 
                            dropout=self.lstm_dropout,
                            bidirectional=self.lstm_bidirectional)
        
        if self.lstm_bidirectional:
            self.hidden2label = nn.Linear(hidden_dim * 2, label_size)
        else:
            self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(fc_dropout)
        
        
    def init_hidden(self):
        if self.lstm_bidirectional:
            if self.use_gpu:
                return (Variable(torch.zeros(self.nb_lstm_layers * 2, self.batch_size, self.hidden_dim).cuda()),
                        Variable(torch.zeros(self.nb_lstm_layers * 2, self.batch_size, self.hidden_dim).cuda()))
            else:
                return (Variable(torch.zeros(self.nb_lstm_layers * 2, self.batch_size, self.hidden_dim)),
                        Variable(torch.zeros(self.nb_lstm_layers * 2, self.batch_size, self.hidden_dim)))
        else:
            if self.use_gpu:
                return (Variable(torch.zeros(self.nb_lstm_layers, self.batch_size, self.hidden_dim).cuda()),
                        Variable(torch.zeros(self.nb_lstm_layers, self.batch_size, self.hidden_dim).cuda()))
            else:
                return (Variable(torch.zeros(self.nb_lstm_layers, self.batch_size, self.hidden_dim)),
                        Variable(torch.zeros(self.nb_lstm_layers, self.batch_size, self.hidden_dim)))
                
    
    def forward(self, sentence):
        x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        y = self.dropout(y)
        log_probs = F.log_softmax(y)
        return log_probs
{% endhighlight %}


I defined `load_params` function that returns a model given a param set.


{% highlight python %}
def load_params(modelClass, pg, use_gpu):
    model = modelClass(batch_size = pg['BATCH_SIZE'],
                  vocab_size = len(CITY.vocab),
                  label_size = len(CONTINENT.vocab) -1, ## b.c of <unk>
                  embedding_dim = pg['EMBEDDING_DIM'],
                  hidden_dim = pg['HIDDEN_DIM'],
                  nb_lstm_layers = pg['nb_1stm_layers'],
                  lstm_dropout = pg['lstm_dropout'],
                  lstm_bidirectional=pg['lstm_bidirectional'],
                  fc_dropout = pg['fc_dropout'],
                  use_gpu = use_gpu)
    if use_gpu:
        return model.cuda()
    else:
        return model
{% endhighlight %}


{% highlight python %}
a_model = load_params(LSTMModel, param_grids[0], use_gpu)
print(a_model)

>>    LSTMModel (
      (embeddings): Embedding(58, 100)
      (lstm): LSTM(100, 100, bidirectional=True)
      (hidden2label): Linear (200 -> 6)
      (dropout): Dropout (p = 0)
    )


nb_trainable_params = sum([param.nelement() for param in a_model.parameters()])
print("Number of parameters to train: {}".format(nb_trainable_params))

>>    Number of parameters to train: 168606
{% endhighlight %}



## 4. Train
Although it doesn't take a long time to train a model thanks to the petit dataset, 24 models could be quite hefty for my cpu. To hasten the training process I ran the following code in GPU environment on FloydHub.


{% highlight python %}
import numpy as np
from torch import optim
import torch.nn.functional as F

out_dir = os.path.abspath(os.path.join(os.path.curdir, "result_dir"))

## make checkpoint directory if it doesn't exist    
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def save_checkpoint(state, out_dir, filename):
    torch.save(state, out_dir + '/' + filename)


def get_accuracy(truth, pred):
    tr = np.array(truth)
    pr = np.array(pred)
    return sum(tr == pr) / len(tr)
{% endhighlight %}

f1_score is a useful metric to monitor when the label is not balanced. `Asia` is nearly 10 times the size of `Oceania`.


{% highlight python %}
CONTINENT.vocab.freqs

>>     Counter({'Africa': 1284,
             'Asia': 2334,
             'Europe': 847,
             'North America': 1418,
             'Oceania': 297,
             'South America': 1041})

{% endhighlight %}


{% highlight python %}
from sklearn.metrics import f1_score
def get_f1(truth, pred):
    tr = np.array(truth)
    pr = np.array(pred)
    return f1_score(tr, pr, average='weighted')
{% endhighlight %}


`train_epoch_progress` is for the training dataset, and `evaluate` for the validation and test set.

{% highlight python %}
def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch):
    
    model.train() ## train mode
    
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
        
    for batch in train_iter:
        if use_gpu:
            city, continent = batch.city_ascii.cuda(), batch.continent.cuda()
        else:
            city, continent = batch.city_ascii, batch.continent
        continent.data.sub_(1) ## -1 to make index start from 0 (0 is <unk> in the vocab)
        truth_res += list(continent.data)
        
        model.batch_size = len(continent.data)
        model.hidden = model.init_hidden()
        
        pred = model(city)
        pred_label = pred.data.max(1)[1].cpu().numpy() ## .cpu() to get it from gpu env
        pred_res += [x for x in pred_label]
        
        model.zero_grad()
        loss = loss_function(pred, continent)
        avg_loss += loss.data[0]
        count += 1
        
        loss.backward()
        optimizer.step()
        
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    f1 = get_f1(truth_res, pred_res)
    
    print('Train: loss %.2f | acc %.1f | f1-score %.3f' % (avg_loss, acc * 100, f1))
        
    return avg_loss, acc, f1
{% endhighlight %}


{% highlight python %}
def evaluate(model, data, loss_function, name):
    
    model.eval() ## eval mode
    
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    
    for batch in data:
        if use_gpu:
            city, continent = batch.city_ascii.cuda(), batch.continent.cuda()
        else:
            city, continent = batch.city_ascii, batch.continent
        continent.data.sub_(1)
        truth_res += list(continent.data)
        
        model.batch_size = len(continent.data)
        model.hidden = model.init_hidden()
        
        pred = model(city)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, continent)
        avg_loss += loss.data[0]
        
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    f1 = get_f1(truth_res, pred_res)
    
    print(name + ': loss %.2f | acc %.1f | f1-score %.3f' % (avg_loss, acc * 100, f1))
    return avg_loss, acc, f1
{% endhighlight %}

Then I ran the models for 25 epochs and saved their best performing model when they beat their own previous records based on validation f1-score.


{% highlight python %}
EPOCHS = 25

import time

loss_function = nn.NLLLoss()
{% endhighlight %}


{% highlight python %}
for idx, pg in enumerate(param_grids):
    
    start_time = time.time()
    
    print('** Training pg:{} **'.format(idx))
    
    model = load_params(LSTMModel, pg, use_gpu)
    ## fixed Learning Rate as 0.001
    ## tried SGD with lr_scheduler
    ## but it was slower than Adam and showed inferior validation accuracy and f1-score.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()
    
    best_val_f1 = 0.0
    
    record = []
    
    for epoch in range(EPOCHS):
        print("Epoch: {}".format(epoch + 1))
        
        rec_dict = {}
        train_loss, acc, train_f1 = train_epoch_progress(model, 
                                         train_iter, 
                                         loss_function, 
                                         optimizer,
                                         CITY, 
                                         CONTINENT, 
                                         epoch)
        
        val_loss, val_acc, val_f1 = evaluate(model, val_iter, loss_function, 'Val')
        
        ## when the current validation f1 exceeds the best validation f1 so far,
        ## replace the best_val_f1 with val_f1
        ## and best_model with the current model
        ## and save the epoch, state_dict, and optimizer for the later use.
        if val_f1 > best_val_f1:
        
            best_val_f1 = val_f1
            best_model = model
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': best_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, out_dir=out_dir, filename='pg_{}__epoch_{}.pth.tar'.format(idx, epoch+1))

        ## logging for visulisation
        rec_dict['epoch'] = epoch + 1
        rec_dict['training_loss'] = train_loss
        rec_dict['training_acc'] = acc
        rec_dict['train_f1'] = train_f1

        rec_dict['val_loss'] = val_loss
        rec_dict['val_acc'] = val_acc
        rec_dict['test_f1'] = val_f1

        record.append(rec_dict)
    
    rec_df = pd.DataFrame(record)
    
    rec_df.to_csv(out_dir + '/pg_{}__record.csv'.format(idx), index=False)

    # eval on test
    test_loss, test_acc, test_f1 = evaluate(best_model, test_iter, loss_function, 'Final Test')
    
    end_time = time.time()
    seconds_elapsed = round(end_time - start_time, 0)
    print("pg:{} took {} seconds.".format(idx, seconds_elapsed))
    print("\n")
    
{% endhighlight %}

Each model roughly took 1 ~ 2 minutes in GPU environment.

## 5. Performance Evaluation
Time to choose the best performing model. In the training process above, my 24 LSTMModels left their own best records and checkpoints. Here are the top 5 models that showed the highest accuracy and f1-score on the test dataset.


{% highlight python %}
import glob

checkpoints = glob.glob(out_dir + '/*.tar')
{% endhighlight %}

I tweaked the `evaluate` function above and added some codes for logging.

{% highlight python %}
import re
def evaluate_on_test(path):
    print(path)
    pg_idx = int(re.findall('/pg_(.*)__epoch', path)[0])
    epoch = int(re.findall('__epoch_(.*).pth', path)[0])
    
    model = load_params(LSTMModel, param_grids[pg_idx], use_gpu)
    model.eval()
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['state_dict'])
    avg_loss, test_acc, f1 = evaluate(model, test_iter, loss_function, 'FINAL TEST')
    
    rec_dict = {}
    rec_dict['pg'] = pg_idx
    rec_dict['epoch'] = epoch
    rec_dict['test_avg_loss'] = avg_loss
    rec_dict['test_acc'] = test_acc
    rec_dict['f1_score'] = f1
    return rec_dict
{% endhighlight %}

As the final f1-score is the ultimate measuring stick for choosing the best model, I loaded all the checkpoints saved during the training and computed their test loss, accuracy and f1-score.

{% highlight python %}
rec_list = []
for ch in checkpoints:
    rd = evaluate_on_test(ch)
    rec_list.append(rd)

>>    /output/result_dir/pg_19__epoch_5.pth.tar
    FINAL TEST: loss 1.25 | acc 52.4 | f1-score 0.503
    /output/result_dir/pg_8__epoch_1.pth.tar
    FINAL TEST: loss 1.36 | acc 47.9 | f1-score 0.434
    /output/result_dir/pg_7__epoch_4.pth.tar
    FINAL TEST: loss 1.20 | acc 52.4 | f1-score 0.504
    /output/result_dir/pg_17__epoch_10.pth.tar
    FINAL TEST: loss 1.21 | acc 55.2 | f1-score 0.536
    /output/result_dir/pg_11__epoch_2.pth.tar
    FINAL TEST: loss 1.29 | acc 50.3 | f1-score 0.475
    /output/result_dir/pg_11__epoch_1.pth.tar
    ...
{% endhighlight %}

    



{% highlight python %}
res_df = pd.DataFrame(rec_list)[['pg', 'epoch', 'test_avg_loss', 'test_acc', 'f1_score']]

by_accuracy = res_df.sort_values(by='test_acc', ascending=False).reset_index(drop=True)
by_accuracy['rank'] = by_accuracy.index + 1
by_accuracy.head()
{% endhighlight %}


### top 5 models by test accuracy

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pg</th>
      <th>epoch</th>
      <th>test_avg_loss</th>
      <th>test_acc</th>
      <th>f1_score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>12</td>
      <td>1.189121</td>
      <td>0.580913</td>
      <td>0.565772</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>16</td>
      <td>1.158638</td>
      <td>0.579530</td>
      <td>0.563450</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>12</td>
      <td>1.171896</td>
      <td>0.578147</td>
      <td>0.559408</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>9</td>
      <td>1.183090</td>
      <td>0.568465</td>
      <td>0.550840</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>6</td>
      <td>1.195676</td>
      <td>0.565698</td>
      <td>0.545025</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


<br>

{% highlight python %}
by_f1 = res_df.sort_values(by='f1_score', ascending=False).reset_index(drop=True)
by_f1['rank'] = by_f1.index + 1
by_f1.head()
{% endhighlight %}


### top 5 models by test f1-score

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pg</th>
      <th>epoch</th>
      <th>test_avg_loss</th>
      <th>test_acc</th>
      <th>f1_score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>12</td>
      <td>1.189121</td>
      <td>0.580913</td>
      <td>0.565772</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>16</td>
      <td>1.158638</td>
      <td>0.579530</td>
      <td>0.563450</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>12</td>
      <td>1.171896</td>
      <td>0.578147</td>
      <td>0.559408</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>15</td>
      <td>1.183138</td>
      <td>0.565698</td>
      <td>0.556997</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>15</td>
      <td>1.170758</td>
      <td>0.565698</td>
      <td>0.554061</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
res_df['pg_epoch'] = "pg_" + res_df.pg.map(str) + "__" + "epoch_" + res_df.epoch.map(str)
res_df2 = res_df.set_index('pg_epoch')[['test_acc', 'f1_score']]
{% endhighlight %}

It seems that the test accuracy and f1-score are in a linear relationship. Taking into consideration that the dataset was imbalance, f1-score should be the proper metric to decide the best model. Thus, pg_17 at epoch 12 is the best performing model among 24 LSTM models I trained. (and in terms of test accuracy)


{% highlight python %}
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(res_df2.test_acc, res_df2.f1_score)

ax.set_xlabel('test accuracy')
ax.set_ylabel('test f1 score')

ax.set_title('checkpoint accuracy ~ f1-score scatterplot')
plt.show()
{% endhighlight %}


![png](/assets/cityFinder/output_65_0.png)


pg_17 has three bidirectional layers with lstm dropout 0.5 and fc dropout 0.5.


{% highlight python %}
param_grids[17]
{% endhighlight %}




    {'BATCH_SIZE': 16,
     'EMBEDDING_DIM': 100,
     'HIDDEN_DIM': 100,
     'fc_dropout': 0.5,
     'lstm_bidirectional': True,
     'lstm_dropout': 0.5,
     'nb_1stm_layers': 3}




{% highlight python %}
def visualise_checkpoint(pg, epoch):
    print(param_grids[pg])
    pg_df = pd.read_csv(out_dir + "/pg_{}__record.csv".format(pg))
    pg_df.set_index('epoch', inplace=True)
    best_epoch = epoch
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(pg_df.training_loss)
    ax[0].plot(pg_df.val_loss)
    ax[0].vlines(best_epoch, 0.75, 1.8, color='r', linewidth=3, label='epoch={}'.format(best_epoch), alpha=0.25)
    ax[0].legend()
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')

    ax[1].plot(pg_df.training_acc)
    ax[1].plot(pg_df.val_acc)
    ax[1].vlines(best_epoch, 0.4, 0.65, color='r', linewidth=3, label='epoch={}'.format(best_epoch), alpha=0.25)
    ax[1].legend()
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('epoch')

    ax[2].plot(pg_df.train_f1)
    ax[2].plot(pg_df.test_f1, label='val_f1')
    ax[2].vlines(best_epoch, 0.3, 0.65, color='r', linewidth=3, label='epoch={}'.format(best_epoch), alpha=0.25)
    ax[2].legend()
    ax[2].set_ylabel('f1')
    ax[2].set_xlabel('epoch')

    fig.suptitle("pg_{} metrics over epochs".format(pg), fontsize=15)

    plt.show()
{% endhighlight %}

It looks like pg_17 started suffering from overfitting after 14~15 epoch going by the rising validation loss. However, pg_17 managed to deal with overfitting better than others especially when compared to other models like pg_0 that has its dropout options deactivated.


{% highlight python %}
visualise_checkpoint(17, 12)

>    {'BATCH_SIZE': 16, 'EMBEDDING_DIM': 100, 'HIDDEN_DIM': 100, \
      'fc_dropout': 0.5, 'lstm_bidirectional': True, 'lstm_dropout': 0.5, 'nb_1stm_layers': 3}
{% endhighlight %}





![png](/assets/cityFinder/output_70_1.png)


pg_0 is terribly overfitted to the training dataset. It's training accuracy and f1-score nearly reaches 1 after 18th epoch.

{% highlight python %}
visualise_checkpoint(0, 7)

>    {'BATCH_SIZE': 16, 'EMBEDDING_DIM': 100, 'HIDDEN_DIM': 100, \
      'fc_dropout': 0, 'lstm_bidirectional': True, 'lstm_dropout': 0, 'nb_1stm_layers': 1}
{% endhighlight %}




![png](/assets/cityFinder/output_71_1.png)


## Inference Test & Confusion Matrix
Test f1-score is a great measurement, but I need to go deeper to see the strengths and weaknesses of the models I've trained. As stated above, this dataset is more or less imbalanced. Spitting out 'Asia' at any given city name would result in higher than 1/6 accuracy. Are the models trained properly? 

{% highlight python %}
best_model_path = out_dir + '/pg_{}__epoch_{}.pth.tar'.format(17, 12)
{% endhighlight %}


{% highlight python %}
best_model = load_params(LSTMModel, param_grids[17], use_gpu)
checkpoint = torch.load(best_model_path, map_location={'cuda:0': 'cpu'})
best_model.load_state_dict(checkpoint['state_dict'])
{% endhighlight %}


{% highlight python %}
def inference(model, target, use_gpu):
    model.eval()
    if use_gpu:
        targetTensor = Variable(torch.Tensor([CITY.vocab.stoi[c] for c in CITY.preprocess(target)]).cuda().type(torch.LongTensor).view(len(target), -1)).cuda()
    else:
        targetTensor = Variable(torch.Tensor([CITY.vocab.stoi[c] for c in CITY.preprocess(target)]).type(torch.LongTensor).view(len(target), -1))
    model.hidden = model.init_hidden()
    model.batch_size = 1
    pred = model(targetTensor)
    res_df = pd.DataFrame([(idx, pred_val) for idx, pred_val in enumerate(pred.data.cpu().numpy()[0])], columns=['res_idx', 'nll'])
    res_df['prob'] = res_df.nll.map(lambda x: round(np.exp(x), 3))
    res_df['continent'] = res_df.res_idx.map(lambda x: CONTINENT.vocab.itos[x+1])
    res_df = res_df.sort_values(by='prob', ascending=False)
    return res_df
{% endhighlight %}


{% highlight python %}
inference(best_model, 'Pyeongchang', use_gpu)
{% endhighlight %}




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>res_idx</th>
      <th>nll</th>
      <th>prob</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.014178</td>
      <td>0.986</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-5.159672</td>
      <td>0.006</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-5.615287</td>
      <td>0.004</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-6.272159</td>
      <td>0.002</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-6.410239</td>
      <td>0.002</td>
      <td>Oceania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-6.759290</td>
      <td>0.001</td>
      <td>South America</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
inference(best_model, 'Pyongyang', use_gpu)
{% endhighlight %}




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>res_idx</th>
      <th>nll</th>
      <th>prob</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.017561</td>
      <td>0.983</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-4.879560</td>
      <td>0.008</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-5.401738</td>
      <td>0.005</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-6.174614</td>
      <td>0.002</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-6.269334</td>
      <td>0.002</td>
      <td>Oceania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-6.627199</td>
      <td>0.001</td>
      <td>South America</td>
    </tr>
  </tbody>
</table>
</div>



Voila! pg_17 got the answers correct! What about some tricky cities from England that sound like the ones in North America?


{% highlight python %}
inference(best_model, 'Sheffield', use_gpu)
{% endhighlight %}




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>res_idx</th>
      <th>nll</th>
      <th>prob</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.321180</td>
      <td>0.725</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-2.430576</td>
      <td>0.088</td>
      <td>Oceania</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-2.543775</td>
      <td>0.079</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-2.955052</td>
      <td>0.052</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-3.102797</td>
      <td>0.045</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-4.496040</td>
      <td>0.011</td>
      <td>South America</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
inference(best_model, 'Bradford', use_gpu)
{% endhighlight %}




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>res_idx</th>
      <th>nll</th>
      <th>prob</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.838458</td>
      <td>0.432</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-1.361331</td>
      <td>0.256</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-2.097363</td>
      <td>0.123</td>
      <td>Oceania</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-2.284119</td>
      <td>0.102</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-3.026701</td>
      <td>0.048</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-3.265310</td>
      <td>0.038</td>
      <td>South America</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
inference(best_model, 'York', use_gpu)
{% endhighlight %}




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>res_idx</th>
      <th>nll</th>
      <th>prob</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.825349</td>
      <td>0.438</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-1.733889</td>
      <td>0.177</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-1.913949</td>
      <td>0.147</td>
      <td>Oceania</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-2.118887</td>
      <td>0.120</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-2.342693</td>
      <td>0.096</td>
      <td>Asia</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-3.835471</td>
      <td>0.022</td>
      <td>South America</td>
    </tr>
  </tbody>
</table>
</div>



As expected, the model shows suboptimal performance when given confusing city names. Let's quantify its performance by continent using sklearn's confusion matrix.

Are Asian cities
{% highlight python %}
def evaluate_by_continent(model, data, loss_function, name):
    
    model.eval() ## eval mode
    
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    
    for batch in data:
        if use_gpu:
            city, continent = batch.city_ascii.cuda(), batch.continent.cuda()
        else:
            city, continent = batch.city_ascii, batch.continent
        continent.data.sub_(1)
        truth_res += list(continent.data)
        
        model.batch_size = len(continent.data)
        model.hidden = model.init_hidden()
        
        pred = model(city)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, continent)
        avg_loss += loss.data[0]
        
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    return truth_res, pred_res

{% endhighlight %}


{% highlight python %}
truth_res, pred_res = evaluate_by_continent(best_model, test_iter, loss_function, 'Eval')
{% endhighlight %}


{% highlight python %}
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
{% endhighlight %}


{% highlight python %}
cf = confusion_matrix(truth_res, pred_res)
cf_norm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]

col_idx = np.arange(0, 6)
cols = [CONTINENT.vocab.itos[ci+1] for ci in col_idx]

cf_df = pd.DataFrame(cf, columns=cols, index=cols)
cf_norm_df = pd.DataFrame(cf_norm, columns=cols, index=cols)
{% endhighlight %}


{% highlight python %}
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(cf_df, annot=True, ax=ax[0], cmap='RdBu_r', cbar=False, fmt='g')
ax[0].set_xlabel('Prediction', fontsize=15)
ax[0].set_ylabel('Ground Truth', fontsize=15)
ax[0].set_title("Confusion Matrix", fontsize=20)

sns.heatmap(cf_norm_df, annot=True, ax=ax[1], cmap='RdBu_r', cbar=False)
ax[1].set_xlabel('Prediction', fontsize=15)
ax[1].set_ylabel('Ground Truth', fontsize=15)
ax[1].set_title("Normalised Confusion Matrix", fontsize=20)
plt.tight_layout()

plt.show()
{% endhighlight %}


![png](/assets/cityFinder/output_87_0.png)


The model's prediction power is pretty good with Asian cities. It might imply that the Asian city names are named quite differently from the cities in other continents. It could be far-fetched, but I think the world history during the Great Expansion might have a plausible answer to this phenomenon. 

## Would Ensemble improve performance?
I used simple voting mechanism by which the label that is chosen the most gets to be the output of the ensemble model.


{% highlight python %}
by_f1.head()
{% endhighlight %}




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pg</th>
      <th>epoch</th>
      <th>test_avg_loss</th>
      <th>test_acc</th>
      <th>f1_score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>12</td>
      <td>1.189121</td>
      <td>0.580913</td>
      <td>0.565772</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>16</td>
      <td>1.158638</td>
      <td>0.579530</td>
      <td>0.563450</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>12</td>
      <td>1.171896</td>
      <td>0.578147</td>
      <td>0.559408</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>15</td>
      <td>1.183138</td>
      <td>0.565698</td>
      <td>0.556997</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>15</td>
      <td>1.170758</td>
      <td>0.565698</td>
      <td>0.554061</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
best_checkpoints = ['/output/result_dir/pg_17__epoch_12.pth.tar',
                    '/output/result_dir/pg_22__epoch_16.pth.tar',
                    '/output/result_dir/pg_22__epoch_12.pth.tar',
                    '/output/result_dir/pg_23__epoch_15.pth.tar',
                    '/output/result_dir/pg_22__epoch_15.pth.tar']
{% endhighlight %}


{% highlight python %}
truth_res = None
pred_list = []
for path in best_checkpoints:
    print(path)
    pg_idx = int(re.findall('/pg_(.*)__epoch', path)[0])
    epoch = int(re.findall('__epoch_(.*).pth', path)[0])
    
    model = load_params(LSTMModel, param_grids[pg_idx], use_gpu)
    model.eval()
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['state_dict'])
    truth_res, pred_res = evaluate_by_continent(model, test_iter, loss_function, 'best_models')
    truth_res = truth_res
    pred_list.append(pred_res)
{% endhighlight %}

    /output/result_dir/pg_17__epoch_12.pth.tar
    /output/result_dir/pg_22__epoch_16.pth.tar
    /output/result_dir/pg_22__epoch_12.pth.tar
    /output/result_dir/pg_23__epoch_15.pth.tar
    /output/result_dir/pg_22__epoch_15.pth.tar



{% highlight python %}
pred_arr = np.asarray(pred_list)
from scipy.stats import mode
ensemble_pred = mode(pred_arr, axis=0)[0][0]
truth_res = np.array(truth_res)

print("True label       : ", truth_res[:10])
print("Ensemble         : ", ensemble_pred[:10])
print("-------------------------------------------------")
print("pg_17__epoch_16  : ", np.array(pred_list[0][:10]))
print("pg_21__epoch_9   : ", np.array(pred_list[1][:10]))
print("pg_5__epoch_10   : ", np.array(pred_list[2][:10]))
print("pg_17__epoch_14  : ", np.array(pred_list[3][:10]))
print("pg_16__epoch_16  : ", np.array(pred_list[4][:10]))

>>    True label       :  [4 0 3 1 2 0 2 4 0 2]
      Ensemble         :  [4 0 3 2 2 3 0 2 0 2]
      -------------------------------------------------
      pg_17__epoch_16  :  [4 0 4 2 2 0 0 4 0 2]
      pg_21__epoch_9   :  [4 0 3 2 2 3 0 2 0 0]
      pg_5__epoch_10   :  [4 0 4 2 2 3 0 0 0 0]
      pg_17__epoch_14  :  [4 0 2 2 2 3 0 2 0 2]
      pg_16__epoch_16  :  [4 0 3 2 2 3 0 2 0 2]


{% endhighlight %}




{% highlight python %}
ensemble = {'type': 'ensemble'}
single = {'type': 'single'}


ensemble['f1'] = round(get_f1(truth_res, ensemble_pred), 3)
ensemble['accuracy'] = round(get_accuracy(truth_res, ensemble_pred), 3)

single['f1'] = round(by_f1.iloc[0].f1_score, 3)
single['accuracy'] = round(by_accuracy.iloc[0].test_acc, 3)


print('f1-score >> ensemble: {} vs. single: {}'.format(ensemble['f1'], single['f1']))
print('accuracy >> ensemble: {} vs. single: {}'.format(ensemble['accuracy'], single['accuracy']))

>>    f1-score >> ensemble: 0.571 vs. single: 0.566
>>    accuracy >> ensemble: 0.589 vs. single: 0.581

{% endhighlight %}




{% highlight python %}
cf = confusion_matrix(truth_res, ensemble_pred)
cf_norm = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]

col_idx = np.arange(0, 6)
cols = [CONTINENT.vocab.itos[ci+1] for ci in col_idx]

cf_df_en = pd.DataFrame(cf, columns=cols, index=cols)
cf_norm_df_en = pd.DataFrame(cf_norm, columns=cols, index=cols)
{% endhighlight %}


{% highlight python %}
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(cf_df_en, annot=True, ax=ax[0], cmap='RdBu_r', cbar=False, fmt='g')
ax[0].set_xlabel('Prediction', fontsize=15)
ax[0].set_ylabel('Ground Truth', fontsize=15)
ax[0].set_title("Confusion Matrix(Ensemble)", fontsize=20)

sns.heatmap(cf_norm_df_en, annot=True, ax=ax[1], cmap='RdBu_r', cbar=False)
ax[1].set_xlabel('Prediction', fontsize=15)
ax[1].set_ylabel('Ground Truth', fontsize=15)
ax[1].set_title("Normalised Confusion Matrix(Ensemble)", fontsize=20)
plt.tight_layout()

plt.show()
{% endhighlight %}


![png](/assets/cityFinder/output_102_0.png)


### Single vs. Ensemble
Let's compare the heatmaps side by side.


{% highlight python %}
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(cf_df, annot=True, ax=ax[0], cmap='RdBu_r', cbar=False, fmt='g', vmin=0, vmax=200)
ax[0].set_xlabel('Prediction', fontsize=15)
ax[0].set_ylabel('Ground Truth', fontsize=15)
ax[0].set_title("Confusion Matrix(Single)", fontsize=20)

sns.heatmap(cf_df_en, annot=True, ax=ax[1], cmap='RdBu_r', cbar=False, fmt='g', vmin=0, vmax=200)
ax[1].set_xlabel('Prediction', fontsize=15)
ax[1].set_ylabel('Ground Truth', fontsize=15)
ax[1].set_title("Confusion Matrix(Ensemble)", fontsize=20)
plt.tight_layout()

plt.show()
{% endhighlight %}


![png](/assets/cityFinder/output_104_0.png)


### Single vs. Ensemble (Normalised)
The Ensemble model is better at Asian and North American cities, but not at African, European, and Oceanian cities.


{% highlight python %}
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(cf_norm_df, annot=True, ax=ax[0], cmap='RdBu_r', cbar=False, vmin=0, vmax=1)
ax[0].set_xlabel('Prediction', fontsize=15)
ax[0].set_ylabel('Ground Truth', fontsize=15)
ax[0].set_title("Confusion Matrix(Single)", fontsize=20)

sns.heatmap(cf_norm_df_en, annot=True, ax=ax[1], cmap='RdBu_r', cbar=False, vmin=0, vmax=1)
ax[1].set_xlabel('Prediction', fontsize=15)
ax[1].set_ylabel('Ground Truth', fontsize=15)
ax[1].set_title("Confusion Matrix(Ensemble)", fontsize=20)
plt.tight_layout()

plt.show()
{% endhighlight %}


![png](/assets/cityFinder/output_106_0.png)


### Inference with Ensemble model
I used the simple voting rule for choosing the prediction label.


{% highlight python %}
models = []
for path in best_checkpoints:
    print(path)
    pg_idx = int(re.findall('/pg_(.*)__epoch', path)[0])
    epoch = int(re.findall('__epoch_(.*).pth', path)[0])
    
    model = load_params(LSTMModel, param_grids[pg_idx], use_gpu)
    model.eval()
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['state_dict'])
    models.append(model)
{% endhighlight %}

    /output/result_dir/pg_17__epoch_12.pth.tar
    /output/result_dir/pg_22__epoch_16.pth.tar
    /output/result_dir/pg_22__epoch_12.pth.tar
    /output/result_dir/pg_23__epoch_15.pth.tar
    /output/result_dir/pg_22__epoch_15.pth.tar



{% highlight python %}
def ensemble_inference(models, target, use_gpu):
    if use_gpu:
        targetTensor = Variable(torch.Tensor([CITY.vocab.stoi[c] for c in CITY.preprocess(target)]).cuda().type(torch.LongTensor).view(len(target), -1)).cuda()
    else:
        targetTensor = Variable(torch.Tensor([CITY.vocab.stoi[c] for c in CITY.preprocess(target)]).type(torch.LongTensor).view(len(target), -1))
    
    pred_idx_list = []
    for cmodel in models:
        cmodel.eval()
        cmodel.hidden = cmodel.init_hidden()
        cmodel.batch_size = 1
        pred = cmodel(targetTensor)
        pred_idx = np.argmax(pred.data.cpu().numpy()[0])
        pred_idx_list.append(pred_idx)
        
        time.sleep(0.1)
                
    (res, cnt) = Counter(pred_idx_list).most_common(1)[0]
    prediction = CONTINENT.vocab.itos[res + 1]
    return cnt, target, prediction  

{% endhighlight %}


{% highlight python %}
cnt, target, prediction = ensemble_inference(models, 'Sheffield', use_gpu)
print("{} models predicted {} to be in {}.".format(cnt, target, prediction))

>>    5 models predicted Sheffield to be in North America.


ensemble_inference(models, 'Bratford', use_gpu)
print("{} models predicted {} to be in {}.".format(cnt, target, prediction))

>>    5 models predicted Sheffield to be in North America.


ensemble_inference(models, 'York', use_gpu)
print("{} models predicted {} to be in {}.".format(cnt, target, prediction))

>>    5 models predicted Sheffield to be in North America.

{% endhighlight %}

Sometimes it's not the number of models but the quality of their performance.
The ensemble model failed to make a difference given cities in the UK. Perhaps mixing models that are strong in each continent categories would produce better results.


## 6. GAME!

So now that I have a deep learning algorithm that predicts its continent given a name of a city, it would be of great fun to play a game against this AI!! Sounds super nerdy. The function `run_test_single` and `run_test_ensemble` randomly choose a record from the test dataset and return a machine-predicted label.


{% highlight python %}
test_set = pd.read_csv('dataset/_test.csv')
{% endhighlight %}


{% highlight python %}
def judge(m_pred, gt):
    if m_pred == gt:
        return "Correct!"
    else:
        return "Incorrect!"

def run_test_single():
    sample = test_set.sample()
    city = sample.city_ascii.values[0]
    continent = sample.continent.values[0]
    print("Which Continent does **{}** belong to?".format(city))
    print(">>> Correct Answer: {}".format(continent))
    print("-----------------------------------------------")
    res_df = inference(best_model, city, use_gpu).sort_values(by='prob', ascending=False)
    m_pred = res_df.iloc[0].continent
    m_pred_prob = res_df.iloc[0].prob
    
    print("Machine prediction: {}({}) -- {}".format(m_pred, m_pred_prob, judge(m_pred, continent)))
    
run_test_single()

>>    Which Continent does **Jining** belong to?
      >>> Correct Answer: Asia
      -----------------------------------------------
      Machine prediction: Asia(0.948) -- Correct!
{% endhighlight %}



{% highlight python %}
def run_test_ensemble():
    sample = test_set.sample()
    city = sample.city_ascii.values[0]
    continent = sample.continent.values[0]
    print("Which Continent does **{}** belong to?".format(city))
    print(">>> Correct Answer: {}".format(continent))
    print("-----------------------------------------------")
    cnt, target, prediction = ensemble_inference(models, city, use_gpu)
    
    print("{} Machine prediction: {} -- {}".format(cnt, prediction, judge(prediction, continent)))
    
run_test_ensemble()

>>    Which Continent does **Demba** belong to?
      >>> Correct Answer: Africa
      -----------------------------------------------
      5 Machine prediction: Africa -- Correct!

{% endhighlight %}
    

Making a web game with AWS Lambda or EC2 would be great, but it's too much for me to build everything on my own. To see how fun it is to play this game, I ran the above function 10 times and see if my parents-in-law, my wife and myself could beat the model!

The results are..

** Human vs. AI **  
Parents 4 vs 6 AI  
Wife    4 vs 6 AI  
Me      2 vs 5 AI  

Here's my score table. I mean.. where the hell is Tom Price? Tom Price is a city in Western Austrailia.

|    | Question              | Me | AI |
|----|-----------------------|----|----|
| 1  | Haeju                 | 1  | 1  |
| 2  | Fond Du Lac           | 0  | 1  |
| 3  | San Juan De Nicaragua | 0  | 0  |
| 4  | Altata                | 0  | 0  |
| 5  | Porvoo                | 1  | 1  |
| 6  | Almirante             | 0  | 1  |
| 7  | Cap-Haitien           | 0  | 1  |
| 8  | Raleigh               | 0  | 0  |
| 9  | Atqasuk               | 0  | 0  |
| 10 | Tom Price             | 0  | 0  |
|    | Total Score           | 2  | 5  |



## Reference

https://github.com/clairett/pytorch-sentiment-classification