---
layout:     post
title:      "OVERWATCH: MASTER CLASSIFIER"
date:       2016-10-01 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/overwatch/overwatch_header.jpg
---

## Intro
<br>

Gone are the glorious days of League of Legends. Blizzard's Overwatch has been a worldwide hit since its debut back in April this year. It's has been tremendously successful over here in South Korea, to the extent that even some primary school kids sneakly played Overwatch (and got caught by the police). Overwatch's immense popularity has lead to the birth of 3rd party services like <a href="http://overlog.gg/">overlog.gg</a> that provide user statistics and in-depth analyses. This company is also known widely for its LOL stat service <a href="http://www.op.gg/">op.gg</a>.

Riot Games provides an extensive variety of data through their api, but Blizzard isn't doing so at the moment. Instead, overlog.gg offers user logs and various stats. In this post, I've built and tested machine-learning models that classify Master-class users from the rest based on some data samples from overlog.gg.  

![overlog.gg](/assets/overwatch/overlog_homepage_eng.png)

<hr>

## Dataset
The <a href="http://overlog.gg/leaderboards">leaderboard</a> page contains approximately 700,000 pieces of user info. To acquire a sample dataset that represents the whole userbase, I split 700,000 users into 20 subgroups based on their ranks, and cralwed 1,000 users from each subgroup (100 users x 10 pages x 20 subgroups).

<hr>

## Hypothesis
<br>
In Overwatch you can play either quick play or competitive play. Competitive play, which only allows 25+ level users, classifies users with points based on their game win/defeat logs. You get Grand Master from 4,000 pts, Master from 3,500 pts and the rest we are not interested in. Let's see what our sample dataset's rank distribution looks like. Here I put points as `skillRating`.

![Competitive play skillRating distribution](/assets/overwatch/overlog_ladder_eng.png)

The skillRating distribution from the sample dataset shows that there is a chasm between the leader group and the followers. This is caused by the way I crawled the sample dataset. The second best subgroup was from page 700 to 710, and the cut-off of Master-class (skillRating 3500) was around page 65. Although it was not intended at all in the first place, this chasm led me to build and test some machine-learning algorithms to tell the very best users from the crowd, for fun! (It's for fun so it might contain some technical glitches and procedural mistakes. TT) 

<hr>

## EDA & Preprocessing
<br>
Let's see what the dataset looks like first. 

![raw data](/assets/overwatch/overlog_raw.png)

* avgFire: Average time on fire in minutes (when you kill loads of 'em)
* kd: Kill / Death ratio
* level: User level
* mostHeroes: Top 3 most played heroes
* platform: Where you play (Korea, US, Europe)
* playtime: Total play time in hours
* player: Player name
* rank: Rank
* skillRating: Competitive Play Points
* winRatio: Win Ratio(%)

First of all, `avgFire`, `kd` and `winRatio` are not in the right format. Python's lambda function does a wonderful job of changing data types nice and easily. `avgFire` is a string `min:sec` so I just made a new column called `avgFire_sec` in seconds.

{% highlight python %}
df['avgFire_sec'] = df['avgFire'].map(time_converter)
df['kd'] = df['kd'].map(lambda x: float(x))
df['rank'] = df['rank'].map(lambda x: int(x.replace(",", "")))
df['winRatio'] = df['winRatio'].map(lambda x: int(x) / 100)
{% endhighlight%}

And `mostHeroes` column has lists that each contains 3 (sometimes 2) hero strings. All the Overwatch heores fall into 4 big categories (attack, defence, charge, heal), so I added these four features (e.g. `['McCree', 'Reaper', 'Mei']` => `attack: 2`, `defence: 1`)

![why Torbjörn and Widowmaker on ATTACK? // source: gamecrate.com](/assets/overwatch/overlog_selection.png)

{% highlight python %}
def attack(hero_list):
    attack_heroes = ['Genji', 'Reaper', 'McCree', 
                     'Soldier: 76', 'Tracer', 'Pharah']
    intersect = set(hero_list).intersection(attack_heroes)
    return len(intersect)

def defence(hero_list):
    defence_heroes = ['Mei', 'Bastion', 'Widowmaker', 
                     'Junkrat', 'Torbjörn', 'Hanjo']
    intersect = set(hero_list).intersection(defence_heroes)
    return len(intersect)

def charge(hero_list):
    charge_heroes = ['D.va', 'Reinhardt', 'Roadhog',
                     'Winston', 'Zarya']
    intersect = set(hero_list).intersection(charge_heroes)
    return len(intersect)

def heal(hero_list):
    heal_heroes = ['Lúcio', 'Mercy', 'Symmetra', 'Ana', 'Zenyatta']
    intersect = set(hero_list).intersection(heal_heroes)
    return len(intersect)

def gtwh(hero_list):
    gtwh_heroes = ['Genji', 'Tracer', 'Widowmaker', 'Hanjo']
    intersect = set(hero_list).intersection(gtwh_heroes)
    return len(intersect)

df['attack'] = df['mostHeroes'].map(lambda x: attack(x))
df['defence'] = df['mostHeroes'].map(lambda x: defence(x))
df['charge'] = df['mostHeroes'].map(lambda x: charge(x))
df['heal'] = df['mostHeroes'].map(lambda x: heal(x))
df['gtwh'] = df['mostHeroes'].map(lambda x: gtwh(x))

{% endhighlight %}

And I added gtwh (Genji, Tracer, Widowmaker, Hanjo, the most hated quartet in Overwatch (in Korea at least)). They are often picked by selfish and kill-orientated buggers, and it's my hypothesis that the more gtwh heroes you have in your top 3, the less likely that you'll be in the master's level.

Plus, I added individual hero columns too. 

{% highlight python %}
df['Genji'] = df.mostHeroes.map(lambda x: int('Genji' in x))
df['Reaper'] = df.mostHeroes.map(lambda x: int('Reaper' in x))
df['McCree'] = df.mostHeroes.map(lambda x: int('McCree' in x))
df['Soldier'] = df.mostHeroes.map(lambda x: int('Soldier: 76' in x))
df['Tracer'] = df.mostHeroes.map(lambda x: int('Tracer' in x))
df['Pharah'] = df.mostHeroes.map(lambda x: int('Pharah' in x))
{% endhighlight %}

And finally, the following code makes a user_class column that separates Master-class users(skillRating >= 3500) from those are not that good (including myself).

{% highlight python %}

def label_class(skillRating):
    if skillRating >= 3500:
        return 0
    else: 
        return 1
    
df['user_class'] = df['skillRating'].map(lambda x: label_class(x))

{% endhighlight %}

Now we've got the columns ready, let's have a look at some plots!

![user_class ~ kd, level, playTime](/assets/overwatch/overlog_corr1_eng.png)

![user_class ~ winRatio, avgFire_sec](/assets/overwatch/overlog_corr2_eng.png)

Whilst `level` and `playTime` are positively skewed, `kd`, `winRatio` and `avgFire_sec` are more or less normally distributed. The target variable `user_class` is extremely imbalanced, as only one-twentith of them are top players. In the joint plots, blue dots(user_class 0) are the master-level players. Classifying them from the rest is a bit of a tricky challenge as some good yet under 3500 users tend to show stats as good as top players.

Next, the heroes. To see which heros are favoured by which, I calculated the probabilities of heroes being chosen for the top 3 most played heores in master and non-master class user groups. 

![Zarya, McCree, Genji are loved by the top players](/assets/overwatch/overlog_hero_selection_eng.png)

The x-axis is the top players' probabilities of choosing heros for their top3, and y-axis the rest. The heroes below the red line are more favoured by the master players and above by the rest. From my personal experience (of skillRating 1800) Genji and McCree were the most difficult heroes to control as I was forced to move and jump constantly to keep myself alive. So was Zarya as her ultimatum requires a good level of team-spirit and understanding of game-flow. On the other hand, Lucio was a pretty easy-going hero: fast speed, simple music change and powerful touch down(Q).

Would there be more drastic differences between the top and the bottom?

![top vs bottom (under 2100)](/assets/overwatch/overlog_hero_selection_deepsea_eng.png)

This plot doesn't look too different from the previous one, apart from some heroes like Junkrat. Junkrat seems to be in favour of bottom-rank users as Junkrat doesn't usually get involved with dog fights in the battle front thanks to his long-ranged weapon.

Pandas has this amazing visualisation tool `radviz` that makes nice plots like the following. As the hero type columns are integer values I added a bit of noise to make the plots clearer. (integer points overlap each other.)

### Users that are under skillRating 2100
![under-2100 users are rather balanced in choosing hero types](/assets/overwatch/overlog_radviz_deepsea.png)
<br>
### Master-class users
![Masters are not that into defence type heroes](/assets/overwatch/overlog_radviz_top.png)

![Bastion rocks in quick play // 출처: https://i.ytimg.com/vi/m0dVmBmCMJs/maxresdefault.jpg](/assets/overwatch/overlog_bastion.png)

There is much more to discover from this dataset, but it's time to move on and get our hands dirty with some machine-learning algorithms.

<hr>

## Feature Engineering
<br>
In the previous part, I preprocessed the raw data and added some new columns like hero types, and ended up with 32 features. The more features you have in your model, the more likely you'll have `the curse of dimensionality`, so it's better to manually/programmatically select (or extract) features.

### Manual Feature Selection
From the exploratory Data Analysis session above, I saw a glimmer of hope of making a fairly decent classifier with the stat variable and hero categories. I added Genji as the final variable as he was often seen with the top players.

### Automatic Feature Selection (ExtraTreeClassifier)
It's also possible to select features based on their importance to the model. I fed training dataset into sklearn's ExtraTreeClassifier, and filtered the top 12 features by their `feature_importance`.

{% highlight python %}

#feature selection
model = ExtraTreesClassifier(n_estimators=200, min_samples_split=200, random_state=0)
model.fit(X_train, y_train)

{% endhighlight %}

![Feature Selection via ExtraTreeClassifier](/assets/overwatch/overlog_feature_importance.png)

The ExtraTreeClassifier got rid of all the hero categories apart from `gtwh` and put McCree, Tracer, Winston, Zarya, Genji and Ana in their places. 

I have also tried PCA and SVD to extract some latent variables, but the result (e.g. explained variance curve) didn't look convincing for some unknown issues. PCA said the best number of features is 2 for all the individual hero variables.

![Feature Extraction via PCA](/assets/overwatch/overlog_pca.png)

On the PCA plot above, top players(user_class 0) are distributed nearly the same as the rest. Disappointing.

<hr>

## Model Testing
<br>

Before building models, I split the training set and test set (90/10). The training set will be used for model fitting, and the test set for final model evaluation.

There are numerous machine-learning algorithms we can use in classification problems like this one. Here I'm going to test the following 4 options which are my favourite algorithms. (I like the sound of `Random Forest`. How Green.) 

1. MinMaxScaler + SelectKBest + LogisticRegression
2. MinMaxScaler + SelectKBest + RandomForestClassifier
3. MinMaxScaler + SelectKBest + SupportVectorClassifier
4. MinMaxScaler + DNNClassifier

* MinMaxScaler: Convert each data column from 0 to 1
* SelectKBest: select k best features for each classifier
* LogisticRegression: basic yet powerful classification algorithm
* RandomForestClassifier: combine multiple decision trees to improve prediction
* SupportVectorClassifier: find the hyperplane that performs best
* DNNClassifier: complicated neural network using TensorFlow's DNNClassifier

### Pipeline and GridSearchCV
Much of model building, fitting and testing is pretty repetitive and complicated. It gets worse as you have a lot of options to tune the hyper parameters of the algorithms by which you can greatly improve the predition power. Sklearn's PipeLine and GridSearchCV ease the pain amazingly. It's really easy to assign and control the sequential stages by putting them in a pipeline object. GridSearchCV takes parameter sets as an argument, tests them in 3 fold cross validation sets(default) and finds the model with the best performance.

So with my first option..

{% highlight python %}

def minmax_logistic(X_training, y_training):
    ## scaler, selectKbest, logistic 모형 설정
    scaler = MinMaxScaler(feature_range = [0,1])
    select = feature_selection.SelectKBest()
    logistic_fit = LogisticRegression()

    ## putting scaling, feature_selection, model building in a pipeline
    pipeline_object = Pipeline([('scaler', scaler), 
                                ('feature_selection', select), 
                                ('model', logistic_fit)])
    
    ## setting the set of parameter sets to test
    tuned_parameters = [{'feature_selection__k': [3, 5, 7],
                        'model__C': [0.01,0.1,1,10],
                    'model__penalty': ['l1','l2']}]

    ## pass the pipeline and parameter sets to GridSearchCV. 
    ## find the best model by roc_auc score.
    cv = GridSearchCV(pipeline_object, 
                      param_grid=tuned_parameters, 
                      scoring = 'roc_auc')
    
    ## fit GridSearchCV with the training set
    cv.fit(X_training.values, y_training['user_class'].values)
    
    ## print the best AUC score
    print("Best AUC score: ", cv.best_score_)
    
    ## and return the final classifier for the final evaluation
    return cv

cv_log = minmax_logistic(X_training[feature_list], y_training)

{% endhighlight %}

The function `minmax_logistic` returns a trained logistic regression classifier, and we can check its best parameter set via `.best_estimator_`. And it's also possible to access the coefficients of each variables.

{% highlight python %}

fil = cv_log.best_estimator_.named_steps['feature_selection'].get_support()
selected_features = list(compress(feature_list, fil))
logistic_coeff = pd.DataFrame(cv_log.best_estimator_.named_steps['model'].coef_[0],
             selected_features, columns=['coefficient'])
logistic_coeff.sort_values(by='coefficient')

{% endhighlight %}


![Coefficients of the logistic classifier](/assets/overwatch/overlog_logistic.png)

As the top player's classification label is 0, all the coffefficients are minus values. Going by the size of them, winRatio and playTime are the most influential variables. 

Likewise I made `pipeline` and `gridSearchCV` for Random Forest and Support Vector Classifier. I passed `n_estimators` and `min_samples_split` for Random Forest, and `kernel` and `C` for SVM for hyper parameter tuning.

Let's see what variables Random Forest picked.
![Feature Importance of Random Forest classifier](/assets/overwatch/overlog_imp.png)

Random Forest picked winRatio and playTime too. However, it selected 5 variables when logistic classifier picked 7.

(Model 1, 2, 3 performed slightly better with programmatically selected features than with manually selected ones.)

<br>

### TensorFlow's DNNClassifier

Google TensorFlow is the most trending machine-learning library at the minute. I'd never personally tried TensorFlow or fully understood the notion of Deep Learning, but I've given it a go. I made a deep neural network with 4 hidden layers by getting some ideas from its tutorial pages. At the minute I do not know how to combine DNNClassifier with gridSearchCV and cross-validation, so I split the training set into the training set and validation set to fit the parameters of the DNN, before putting it to the final evaluation with the test set.

{% highlight python %}
import tensorflow as tf

classifier = None

def tf_iteration():

    ## acc_list saves validation accuracy after each learning step
    acc_list = []

    ## this nn use manually selected features
    ## 4 hidden layers // 12, 20, 15, 12 nodes in each layer
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=10)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[12, 20, 15, 12],
                                            n_classes=2,
                                            model_dir="/tmp/newd_30")

    ## learn 2000 steps for 50 times
    for i in range(0, 50):
        classifier.fit(x=X_train[feature_list_new], y=y_train['user_class'].values, steps=2000, batch_size=2048)

        y_prediction = classifier.predict(X_validation[feature_list_new]).astype(int)
        report = metrics.classification_report(y_validation['user_class'], y_prediction)
        
        print(report)
        accuracy_score = classifier.evaluate(x=X_validation[feature_list_new], y=y_validation['user_class'].values)

        print ('step:', accuracy_score['global_step'], 
               'auc_score', accuracy_score['auc'])
        acc_list.append(accuracy_score)
    
    return acc_list, classifier
        
{% endhighlight %}

![DNN's AUC score over steps](/assets/overwatch/overlog_dnn_auc.png)

After fitting DNN for 38,000 times, the AUC score curve flattens out. I hardcoded the code above to fit the model 38,000times and evaluated the model performance with the test dataset. 

It was my first time of building a neural network that has 4 hidden layers, but thanks to TensorFlow's powerful yet easy-to-use API, the coding itselt was not too hard. However, as I'm no expert in deep learning I found it difficult to finetune the model by tweaking parameters like the number of hidden layers, the number of nodes in each hidden layer, step size, batch_size, dropout ratio, just to name a few. Furthermore, unlike Iris dataset that was more or less easy to train with low step size, the auc score didn't get up to .5 until a few hundred times of fitting with this overlog dataset. At the end of the day I got what I wanted, but I'm still not sure fitting nn for 38,000 times was the right thing to do. (too much training? but it worked well with the unseen test dataset.)

<hr>

## Model Evaluation
<br>

So, it's time for a show down. I'm going to evaluate 4 models that I've built above with the test dataset. To make things easier, I made the following custom model_tester function.

{% highlight python %}

def model_tester(cv, X_test, y_test):
    y_pred = cv.predict(X_test).astype(int)
    report = metrics.classification_report(y_test['user_class'], y_pred)
    print(report)

{% endhighlight %}

In this classification_report, I'm going to focus on the f1-score of class 0, because A) this dataset is imbalance and B) I'm particularly interested in detecting top players well.

### LogisticRegression
![Logistic_Classifier_Result](/assets/overwatch/overlog_logi_result.png)
<br>
### RandomForestClassifier
![RF_Result](/assets/overwatch/overlog_rf_result.png)
<br>

### SupportVectorClassifier
![SVM_Result](/assets/overwatch/overlog_svm_result.png)
<br>
### DNNClassifier
![DNN_Result](/assets/overwatch/overlog_dnn_result.png)
<br>

Based on the F1-score of class 0, DNN performed best, then RF, SVM and Logistic Regression. DNN was the best performer, but I believe RF and SVM could do better. (I spent most of my time finetuning DNN :P) (DNN with 100,000 steps got test F1-score of 0.92 but the DNN with 38,000 steps was chosen for the best valdiation AUC.)

<hr>

## Prediction

Alright. It's time for a real classification. DNN is my weapon of choice.

First of all, let me test with my own Overwatch log. 0 means top players, 1 means the rest.
![junk3 - Competitive Play](/assets/overwatch/overlog_junk3.png)
<br>
![Yup, the model is well built.](/assets/overwatch/overlog_junk3_result.png)
The DNN says 1. Correct. 

What about some random good and good-ish players from the leaderboard?
![Random Test Result](/assets/overwatch/overlog_random_test.png)
The second player, who is a Master-class player, is mis-classified as a non-Master player. I guess it's due to his low winRatio, k/d and relatively short playTime. 

<hr>

## Outro

In this post, I have crawled some user stat data from overlog.gg, preprocessed features and built some models to classify top players from the crowd. My testing result says that DNN with 4 hidden layers performed the best. This time I have played with the fully connected Deep Learning model, but some experts said it's also possible to use CNN with this dataset. For those who are interested in this overlog dataset and want to try their own classification models, I've uploaded the training and test datasets on <a href="https://github.com/junkwhinger/overlog">github</a>. Make your own model, predict y_pred with X_test and send me(junsik.whang@gmail.com) the prediction result. Then I'll email you the performance evaluation result back :) Thank you for reading. 