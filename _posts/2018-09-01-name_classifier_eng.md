---
layout:     post
title:      "Self Attention: Name Classifier"
date:       2018-09-01 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/selfattention/variants_of_junsik.png
---



# Self Attention: Name Classifier



## Intro

On hearing a person's name, we can often correctly tell if it's he or she. Male names tend to have strong pronunciations like Mark, Robert, and Lucas. On the other hand, female names are likely to sound smoother like Lucy, Stella, and Valerie. Would a neural network be able to replicate our classification process? And what part of names would it pay attention to mainly?


## Dataset

For this personal research project, I crawled commonly used baby names that are freely available on the internet. Some of them had metadata like origin and popularities.

### example

| babyname | sex  | origin |
| -------- | ---- | ------ |
| Aakesh   | boy  | Indian |
| Aaren    | boy  | Hebrew |
| Abalina  | girl | Hebrew |
| ...      | ...  |        |



### Exploration

Let's dive into the dataset and find out some useful patterns.

#### Most frequently used first letter

| Rank | Total               | Girl               | Boy                 |
| ---- | ------------------- | ------------------ | ------------------- |
| 1    | A - 3,792건 (10.2%) | A - 2,118건 (9.7%) | A - 1,674건 (10.8%) |
| 2    | C - 2,887건 (7.7%)  | C - 1,880건 (8.6%) | B - 1,051건 (6.8%)  |
| 3    | S - 2,862건 (7.7%)  | S - 1,825건 (8.4%) | M - 1,051건 (6.8%)  |
| 4    | M - 2,615건 (7.0%)  | M - 1,564건 (7.2%) | S - 1,037건 (6.7%)  |
| 5    | K - 2,262건 (6.1%)  | K - 1,531건 (7.0%) | C - 1,007건 (6.5%)  |

A is the most commonly chosen first letter in both sexes. B stands out among the boys' first letters.


#### Most frequently used first two letters

| Rank | Total               | Girl              | Boy               |
| ---- | ------------------- | ----------------- | ----------------- |
| 1    | Ma - 1,454건 (3.9%) | Ma - 907건 (4.2%) | Ma - 547건 (3.5%) |
| 2    | Ka - 914건 (2.4%)   | Ka - 726건 (3.3%) | Ha - 368건 (2.4%) |
| 3    | Sh - 864건 (2.3%)   | Sh - 715건 (3.3%) | Al - 336건 (2.2%) |
| 4    | Ca - 856건 (2.3%)   | Ca - 624건 (2.9%) | Da - 305건 (2.0%) |
| 5    | Al - 842건 (2.3%)   | Ch - 600건 (2.8%) | De - 300건 (1.9%) |

M pops up in the top first two letters. The difference between sexes is a little bit more prominent than the first letter chart.



#### Most dominant first two letters by sex

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

Sh, Ka, Ch, La, Jo are used more frequently in female names than male names. Male names prefer Ha, Ba, Ga, Co as their opening sequences. But the ratio gaps don't seem to be widened significantly.



#### Most frequently used last letter

| Rank | Total                | Girl                 | Boy                 |
| ---- | -------------------- | -------------------- | ------------------- |
| 1    | a - 10,693건 (28.6%) | a - 10,198건 (46.8%) | n - 3,123건 (20.1%) |
| 2    | e - 6,471건 (17.3%)  | e - 4,989건 (22.9%)  | o - 1,519건 (9.8%)  |
| 3    | n - 4,813건 (12.9%)  | n - 1,690건 (7.8%)   | e - 1,482건 (9.5%)  |
| 4    | y - 1,992건 (5.3%)   | y - 1,036건 (4.8%)   | s - 1,448건 (9.3%)  |
| 5    | s - 1,825건 (4.9%)   | i - 927건 (4.3%)     | r - 1,123건 (7.2%)  |

The last letter might hold some clues.
Nearly half of the female names end with A. A fifth of them have E ending. These endings are not as dominant in male names as in female names.


#### Most dominant last letter by sex

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



#### Most frequently used last two letters

| Rank | Total               | Girl                 | Boy               |
| ---- | ------------------- | -------------------- | ----------------- |
| 1    | na - 2,419건 (6.5%) | na - 2,375건 (10.9%) | on - 890건 (5.7%) |
| 2    | ne - 1,713건 (4.6%) | ne - 1,476건 (6.8%)  | an - 763건 (4.9%) |
| 3    | ia - 1,510건 (4.0%) | ia - 1,472건 (6.8%)  | er - 558건 (3.6%) |
| 4    | ie - 1,022건 (2.7%) | la - 909건 (4.2%)    | in - 482건 (3.1%) |
| 5    | on - 1,000건 (2.7%) | ta - 837건 (3.8%)    | us - 476건 (3.1%) |



#### Most dominant last two letters by sex

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



The simple aggregations above revealed that the last one or two letters might influence how we tell male names from female names. Would a neural classifier perform better than simple rules? Would it also pay attention to how the names end?


## Model


Text classification models usually extract patterns from text via RNN or CNN layers before processing them through a single or multiple fully connected layers to obtain probability values for each class.

![basic_lstm_classifier](/assets/selfattention/basic_lstm_classifier.png) 

BiLSTM model above does perform well, but it does not tell us the evidence that it takes into account. I could use occlusion methods like Local Interpretable Model-Agnostic Explanations(LIME), but these methods run the model multiple times with masked inputs to figure out the part that contributes the most. There must be a more straight-forward and convenient way to do this.


### A Structure Self-Attentive Sentence Embedding

A Structured Self-Attentive Sentence Embedding (2017.03) introduces Self-Attention to instill visibility into the text model. 

![self_attention](/assets/selfattention/self_attention.png)

The self-attention model uses the same BiLSTM feature extractor as an ordinary text classifier. It passes the feature through two fully connected layers (W_s1, W_s2) to achieve Attention matrix whose shape is n_token x hops. `da` for W_s1 and `hops` for W_s2 are hyperparameters. Compared to the conventional attention mechanisms that output an attention vector, the attention matrix has the following benefits:
- Multiple attention vectors represent multiple features that a sentence has.
- The model does not need extra inputs to obtain attention. 
- Attention alleviates the burden of LSTM as it accesses all the time steps of the LSTM layer.
- Attention matrix is incredibly easy to visualize as a heat map whose size is hops x n_token.  


A softmax operation is included in the attention part to normalize each vector so that the visualized heat map would be understood intuitively. 

To obtain the classification result, the model multiplies the LSTM output with the attention matrix before flattening and fully connected layers. Some pruning methods are suggested by the paper to decrease the trainable parameters, but I skipped this part as my model itself was not big enough to be pruned.


### Loss Function

In addition to the cross-entropy loss for classification error, the paper introduces another loss function called `Penalization Term`. The attention matrix is composed of multiple vectors that focus on specific parts of the input sentence. It will be of a huge waste if several attention vectors end up looking at the same area. Penalization term prevents this from happening.

$ P = ||(AA^T - I )||_F^2 $

The attention matrix $A$ is multiplied with its own transposed matrix before subtracting an identity matrix. The penalization term is the Frobenius norm of the operation.



## Experiment

### Training Setting

The authors carried out some interesting experiments such as classifying emotions of reviews and predicting ages of Twitterians by their tweets. The fundamental component of the inputs used in these experiments is words.

However, in my experiment, the basic unit is alphabets. As the diversity of alphabet letters is far less complicated than of English word I thought the hyperparameters used in the paper would be much of an overkill to classify names. 

So I reduced some of the hyperparameters to 1/10 or 1/5 of the ones suggested by the paper.

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

In line with the experimental results of the paper, the self-attention model outperformed the baseline model regarding validation accuracy. The training accuracy was about the same, indicating that the model learned generalized representations.

By class...

| Models              | Precision(girl) | Precision(boy) | Recall(girl) | Recall(boy) |
| ------------------- | --------------- | -------------- | ------------ | ----------- |
| BiLSTM + MaxPooling | 0.901           | 0.860          | 0.900        | 0.862       |
| Self Attention      | 0.903           | 0.876          | 0.913        | 0.862       |

Neither of them produced biased predictions.


## Visualizing Self Attention

### Attention Heatmap

And here comes the hidden purpose of this blog post. Which part of the names did the model pay attention to before making the final decision? As explained above, each row vector of the attention matrix sums up to 1. All I have to do is to pass the numpy array to matplotlib.  

![attention_heatmap](/assets/selfattention/attention_heatmap.png)

The heat map above shows how each attention vector highlights different parts of the name. During hyperparameter tuning, I set `hops` as 30 and ended up seeing nearly all the letters highlighted. The model performance was not inferior, but unnecessarily large hops severely damaged the model interpretability.



### Attention on Names

To make it more visually appealing, I summed up all the row vectors and normalized it (softmax) to overlay on the text directly.

Here are some of the famous names from Happy Potter.

![harrypotter](/assets/selfattention/harrypotter.png)


Apart from Harry, Hermione, Albus, and Draco, attentions tend to locate at the end of the names. It agrees with my previous hypothesis and simple aggregation results.


The following are the characters from Marvel Cinematic Universe.

![mcu](/assets/selfattention/mcu.png)

The model thought Tony and Loki as girly names and pepper as a boy's name. 'er' must be an unusual ending for female names.



How would the model react to the various endings?

![variants_of_cat](/assets/selfattention/variants_of_cat.png)

'ne' and 'na' endings boost the probability for female.

![variants_of_chris](/assets/selfattention/variants_of_chris.png)
 
Chris and Christina are classified as female names by the model, but the ending 'o' flips the result. Christian is misclassified as a female name. 




Albeit some prediction mistakes, the model does do its job. Would it also work decently on the names that it has nearly never seen before? The majority of the names in my dataset have western origins. Only a handful are from South Korea. Let's see how the model works on my colleagues' names.


![koreanboys](/assets/selfattention/koreanboys.png)

Boys are all correctly classified.



![koreangirls](/assets/selfattention/koreangirls.png)

Girls are all incorrect. It seems that the typical Korean female name endings are perceived boyish by the model.



Lastly, what about my own name?

![variants_of_junsik](/assets/selfattention/variants_of_junsik.png)

The model predicted Jun as a girly name, but its probability is much lower than June's. So June did sound female to the neural network because of the `e` at the end.



### On Embedding Space

During the forward propagation, the model obtains the embedded representation of the input data (name in my case). If the embedding matrices are numerical versions of the input texts, would they be forming clusters based on their semantic meaning and features?

I flattened the embedding matrix into a single vector and reduced its dimensions to 2 using TSNE algorithm. It took too long to process all 30,000 names, so I picked the most popular 100 names from each sex for visualization.

First of all, by sex.

![bySex](/assets/selfattention/bySex.png)

TSNE works wonderfully like in many other cases. Boys' names form a big cluster on the top right corners and girls' names bottom left. It seems that the names that are located close to each other tend to end the same.



Would the origins of the names influence how they end? 

![byOrigin](/assets/selfattention/byOrigin.png)



It's not as apparent as sex, but the names tend to cluster by their origins. Grayson, Jackson, Brandon, Jameson from the UK, Valentina, Emilia, Victoria, Aurora, Olivia from Latin heritage.


male by origin-

![boysByOrigin](/assets/selfattention/boysByOrigin.png)



female by origin-

![girlsByOrigin](/assets/selfattention/girlsByOrigin.png)





### Matrix Computation: Emilia - Emily + Lucy?


King - Man + Woman = Queen is a textbook example of word embedding. Not only does it makes semantic sense, but it also satisfies numerical sense. 

![king_to_queen](/assets/selfattention/king_to_queen.jpeg)

source: https://medium.com/@thoszymkowiak/how-to-implement-sentiment-analysis-using-word-embedding-and-convolutional-neural-networks-on-keras-163197aef623  


If the name texts could be expressed in a numerical form, would it also make some interesting numerical computations like King and Queen? The semantic meanings of the name text are nowhere near as rich as the English word, but why not? Let's do it for fun.

I put all the names from the training and validation dataset to obtain name-embedding dictionary. Each embedding matrix is of size hops(5) x 2 hidden_dim(600). I picked three names to run element-wise plus(+) and minus(-) operation to get the query matrix. Then I calculated the matrix euclidean distance between the query matrix and all the embedding matrices listed in the dictionary. Then I printed the 5 names with the lowest distance.


1) Emilia - Emily  + Lucy = Lucia!

![emily](/assets/selfattention/emily.png)

Emilia - Emily gives me 'ia'. Adding Lucy to 'ia' gives me Lucia!

2) Susie - Susanne + Roxie = Roxie!

![susie](/assets/selfattention/susie.png)

3) Christina - Christine + Austine = Austina!

![christina](/assets/selfattention/christina.png)

The simple computation does output expected results but it's not as amazing as King - Man + Woman = Queen. It hardly contains any high-level symantic meanings. Let's say we subtract "John" from "Paul" and add "Hank".

4) Paul - John + Hank = ?

![paul](/assets/selfattention/paul.png)

We get "Bank" but does it mean anything meaningful?



## Outro

In this simple research project, I implemented LSTM + Self Attention model to classify boy/girl names and visualized where the model paid attention. In many example cases, the model tends to pay special attention to how the names end. 'e' at the end makes 'Jun' female, but 'sik' makes it masculine. It coincides more or less with how I predict if it's he or she on hearing a person's name. Interpretable deep learning makes me think about how I think.



## Reference
<a href="https://arxiv.org/pdf/1703.03130.pdf">A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING</a>  
<a href="https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding">An open-source implementation of the paper 'A Structured Self-Attentive Sentence Embedding' published by IBM and MILA.</a>
