---
layout:     post
title:      "OVERWATCH: 천상계 분류기"
date:       2016-10-01 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/overwatch/overwatch_header.jpg
tags: featured
---

## 들어가며
<br>

몇 년간 PC방 점유율 1위를 지켜오던 리그 오브 레전드(LOL)을 밀어내고 블리자드의 오버워치(OverWatch)가 많은 게이머들의 주목을 받고 있습니다. 블리자드 게임이라면 가끔 히오스나 몇 판 하던 저도 최근에는 오버워치로 갈아탔죠! 오버워치가 인기를 끌면서 <a href="http://overlog.gg/">overlog.gg</a>와 같은 유저의 전적을 제공하는 통계 서비스도 등장했습니다. 특히 이 회사같은 경우에는 <a href="http://www.op.gg/">op.gg</a>라는 LOL 전적 제공 서비스로도 유명합니다. 

LOL을 서비스하는 라이엇게임즈에서는 상당히 폭넓은 데이터를 api를 통해 제공하지만, 블리자드에서는 오버워치용 api는 아직 구축하지 않은 모양입니다. 대신 overlog.gg에서 유저들의 전적과 각종 통계치를 확보할 수 있는데요, 이번 분석에서는 overlog.gg에서 일부 크롤링한 데이터를 바탕으로 최상위권 유저를 분류하는 '천상계 분류기' 모델을 만들어봤습니다.

![overlog.gg](/assets/overwatch/overlog_homepage.png)

<hr>

## 데이터셋
overlog.gg에서는 <a href="http://overlog.gg/leaderboards">leaderboard(순위표)</a>를 통해 방대한 양의 유저 전적 정보를 제공하고 있습니다. 전체 유저풀을 표현할 수 있는 샘플링을 하기 위해서 1위(천상계)부터 약 700,000위(심해)사이를 수집하였습니다. 각 페이지는 100명씩 순위가 기록되어 있습니다. 1페이지부터 7천페이지까지 20그룹으로 분류한 후, 각 분류별로 10페이지씩, 총 20,000명의 데이터를 수집하였습니다. (2만명 중 1명은 승률 100% 초과로 제외하여 총 19,999명 데이터 사용)

<hr>

## 분석 방향
<br>
오버워치는 히오스와 동일하게 일반게임인 빠른대전과 래더게임인 경쟁전으로 나눠집니다. 유저레벨 25이상에서만 참여할 수 있는 경쟁전은 실력에 맞는 점수가 주어져서 유저들의 경쟁심을 자극하는데요, 4천점 이상이면 그랜드마스터, 3천 5백점부터는 마스터, 이하는 다이아몬드 등등으로 분류됩니다. 샘플링한 데이터셋을 보면 다음과 같이 유저가 분포하고 있습니다.

![경쟁전 점수대 분포](/assets/overwatch/overlog_ladder.png)

overlog.gg의 순위표에서는 점수대가 완만한 정규분포를 이루는데 반해, 크롤링한 데이터는 페이지 단위로 샘플링을 했기 때문에 최상위와 상위권 사이의 틈이 뭉텅 날아간 것으로 보입니다. 실제로 상위 2번째 그룹의 시작페이지는 700으로 설정했습니다만, 마스터 레벨인 3천 5백점은 약 65페이지에 부근에 위치해있습니다. 

분절된 분포가 조금 아쉽던 순간에 문득 이 두 집단을 분류할 수 있을까? 하는 생각이 들었습니다. 최상위권 유저들과 다이아몬드 이하의 일반 유저들은 전적에서 어떤 차이를 보일까? 전적 데이터만을 기준으로 아주 잘하는, 소위 천상계 유저들을 분류할 수 있지 않을까? 하는 생각이 들어 몇 가지 알고리즘을 활용해 분류모델을 만들어봤습니다.

<hr>

## EDA & Preprocessing
<br>
들어가기에 앞서 데이터셋이 어떻게 구성되어있는지 확인해봅시다. 

![로데이터](/assets/overwatch/overlog_raw.png)

* avgFire: 폭주시간 (연속킬을 하는 등 평균 폭주 시간(단위: 분))
* kd: Kill / Death 비율
* level: 유저 레벨
* mostHeroes: 최장 플레이 영웅 상위 3명
* platform: 활동 플랫폼 (한국, 미국, 유럽)
* playtime: 총 플레이 시간(단위: 시간)
* player: 플레이어명
* rank: 순위
* skillRating: 경쟁전 점수
* winRatio: 승률(단위: %)

일단 현재 실수값으로 되어있지 않은 avgFire, kd와 %단위가 아닌 winRatio를 바꿔줍니다. Python에서 lambda함수를 사용하면 빠르고 편리하게 데이터타입을 바꿀 수 있습니다. avgFire는 분:초 단위로 되어있으므로 avgFire_sec이라는 컬럼을 새로 만들어 초단위 폭주시간을 새로 산출했습니다.

{% highlight python %}
df['avgFire_sec'] = df['avgFire'].map(time_converter)
df['kd'] = df['kd'].map(lambda x: float(x))
df['rank'] = df['rank'].map(lambda x: int(x.replace(",", "")))
df['winRatio'] = df['winRatio'].map(lambda x: int(x) / 100)
{% endhighlight%}

그리고 mostHeroes는 3명의 영웅 String이 리스트안에 들어있습니다. 유저의 숙련도에 따라서 사용하는 영웅도 다를 수 있을 것 같은데요, 모든 영웅을 컬럼으로 바꾸는 것보다는 영웅의 타입(공격형, 방어형, 돌격형, 치유형)별로 영웅 수를 새로운 피쳐로 넣어볼 수 있을 것 같습니다.

![공..토르비욘?에 위도우? // 출처: gamecrate.com](/assets/overwatch/overlog_selection.png)

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

이런 식으로 각 분류별 영웅 수를 새로운 컬럼으로 넣어봤습니다. 또 오버워치에서 지탄을 받는 조합인 겐트위한(겐지/트레이서/위도우메이커/한조)도 추가해봤습니다. 충이 많은 캐릭터를 선택한다면 경쟁전 점수 역시 높지 않을 것이다라는 가설입니다.

![보기만 해도 암이.. // 출처: http://bns.plaync.com/board/free/article/6450221?p=2](/assets/overwatch/overlog_gtwh.png)

추가적으로 개별 영웅 역시 컬럼화시킬 수 있습니다.
{% highlight python %}
df['Genji'] = df.mostHeroes.map(lambda x: int('Genji' in x))
df['Reaper'] = df.mostHeroes.map(lambda x: int('Reaper' in x))
df['McCree'] = df.mostHeroes.map(lambda x: int('McCree' in x))
df['Soldier'] = df.mostHeroes.map(lambda x: int('Soldier: 76' in x))
df['Tracer'] = df.mostHeroes.map(lambda x: int('Tracer' in x))
df['Pharah'] = df.mostHeroes.map(lambda x: int('Pharah' in x))
{% endhighlight %}

마지막으로 경쟁전 점수 3500점을 기준으로 유저 분류 컬럼을 만듭니다. 이 컬럼은 나중에 분류기를 만들때 target 변수로 들어갑니다. 

{% highlight python %}

def label_class(skillRating):
    if skillRating >= 3500:
        return 0
    else: 
        return 1
    
df['user_class'] = df['skillRating'].map(lambda x: label_class(x))

{% endhighlight %}

이제 먼저 유저 분류와 전적 변수간의 상관관계를 봅시다.

![유저 분류 ~ 전적간 상관관계](/assets/overwatch/overlog_corr1.png)

![유저 분류 ~ 전적간 상관관계](/assets/overwatch/overlog_corr2.png)

`level`과 `playtime`은 왼쪽으로 크게 치우친 반면에, `kd`, `winRatio`, `avgFire_sec`은 어느 정도 종형 분포를 이루고 있습니다. 그리고 `user_class`는 전체 유저의 1/20만이 최상위권 유저여서 극히 불균형한 형태를 보입니다. 푸른 점이 최상위유저, 녹색점이 일반 유저인데, 각 변수간 jointplot을 보면 분류가 쉽지 않을 것 같습니다. 최상위권에 미치지는 못하지만 실력이 꽤 좋은 3000점대 유저들과의 분류를 얼마나 잘 하느냐가 핵심일 것 같습니다. 

다음으로 영웅을 살펴봅시다. 영웅 범주나 영웅 선택값은 정수로 이루어져 눈에 띄는 분포가 드러나지 않았습니다. 대신에 최상위와 일반 유저를 분류한 후, 각 영웅이 최다 빈도 3인에 선택될 확률을 비교해봤습니다.

![유저 분류 ~ 영웅 최빈 선택 확률](/assets/overwatch/overlog_hero_selection.png)

x축은 최상위 유저군, y축은 일반 유저의 영웅 최빈 선택 확률입니다. 붉은 선은 선택확률이 같은 벤치마크 선으로, 선보다 아래에 있으면 최상위 유저군이 선호하는, 위에 있으면 일반 유저군이 더 자주 플레이하는 영웅으로 볼 수 있겠습니다. 최상위 유저들은 자리야, 맥크리, 겐지를 선호하고, 일반 유저들은 루시우를 상대적으로 더 플레이한 것으로 볼 수 있겠습니다. 개인적인 경험에서 해석하자면, 피통이 낮은 겐지는 빠르고 유연한 손놀림이, 맥크리는 위치 선정이나 섬광탄 & 구르기 사용이 중요하다는 점에서 초보자들은 하기 어려운 캐릭터가 아닌가 싶습니다. 자리야 역시 탱커이지만 원거리 공격을 하고 방어막을 씌운다는 점에서 플레이하기가 조금 어렵다는 인상을 받았구요. 반대로 루시우는 음악만 바꿔주고 적절히 궁극기만 잘 써주면 되어서 초보자도 하기에 무난했습니다. 

그러면 경쟁점 점수 2100점 이하의 심해와 최상위권을 비교해보면 어떨까요?

![유저 분류 ~ 영웅 최빈 선택 확률(천상계 vs. 심해)](/assets/overwatch/overlog_hero_selection_deepsea.png)

앞선 분포와 크게 차이나는 점은 없어보입니다. 다만, 왼편에 뭉쳐있던 방어형 영웅들이 조금 더 자세히 보입니다. 방어형 유닛 중 특히 정크랫의 심해 픽률이 상대적으로 올라갔는데, 장거리 곡사형 무기를 사용하다보니 후방에서 안전하게 플레이하고 싶은 심해 유저들의 선택을 많이 받은 것으로 보입니다. 

pandas의 radviz를 통해서 영웅 유형에 대한 유저들의 분포를 살펴보면 비슷한 결과가 나옵니다. 영웅 유형의 값이 0, 1, 2, 3으로 정수값이어서 약간의 노이즈를 더해 분포를 뿌려봤습니다.
### 심해 유저의 영웅 유형 선택 경향
![전반적으로 분포해있음](/assets/overwatch/overlog_radviz_deepsea.png)
<br>
### 천상계 유저의 영웅 유형 선택 경향
![방어 유형에 집중된 유저는 거의 없음](/assets/overwatch/overlog_radviz_top.png)


![빠대에선 POTG를 자주 먹는 바스티온 // 출처: https://i.ytimg.com/vi/m0dVmBmCMJs/maxresdefault.jpg](/assets/overwatch/overlog_bastion.png)

더 알아볼 것이 많겠지만, EDA는 여기서 마무리짓고, 지금까지 뽑은 피쳐셋을 가지고 천상계 분류기를 만들어봅시다. 본 분석에서는 경쟁전 점수가 몇 점이냐가 아닌 천상계냐 아니냐 (심해냐?)로 분류를 할 것이므로, 경쟁전 점수 3500점을 기준으로 위면 0, 아니면 1로 분류를 한 'user_class'라는 target 컬럼을 만들어 사용합니다.


<hr>

## Feature Engineering
<br>
앞선 파트에서는 로데이터를 가공하여 영웅 범주와 영웅 최빈 선택 여부를 실수값으로 산출했습니다. 오버워치에는 선택할 수 있는 영웅 수가 많다보니 총 피쳐의 개수가 32개나 되었습니다. 모델에 넣는 피쳐가 많을수록 모델이 복잡해지고 설명력이 떨어지는 '차원의 저주' 문제가 발생하므로, 이 단계에서 Feature를 수동/자동으로 걸러내는 과정을 거쳤습니다.

### 수동 Feature Selection
EDA를 통해서 승률이나 레벨 등 연속형 변수로 깔끔하게 분류하기는 어렵겠지만, 타겟변수와의 대체적인 상관성이 보이므로 이들을 적절히 조합하면 성능이 좋을 것 같다는 생각이 들었습니다. 거기에다 최상위 유저들이 자주 플레이했던 Genji를 추가해서 총 10개의 피쳐를 준비합니다.

### 자동 Feature Selection (ExtraTreeClassifier)
모델에 대한 피쳐의 중요도를 계산하여 선택하는 방법도 있습니다. 여기에서는 sklearn의 ExtraTreeClassifier에 test 셋을 제외한 데이터를 집어넣고 모델을 피팅한 후에 feature_importance를 기준으로 상위 12개의 피쳐를 뽑았습니다.

{% highlight python %}

#feature selection
model = ExtraTreesClassifier(n_estimators=200, min_samples_split=200, random_state=0)
model.fit(X_train, y_train)

{% endhighlight %}

![ExtraTreeClassifier를 이용한 Feature Selection](/assets/overwatch/overlog_feature_importance.png)

수동으로 선택한 피쳐와 매우 유사하지만, gtwh을 제외한 모든 상위 영웅 분류(attack 등)가 날아가고 맥크리, 트레이서, 윈스턴, 자리야, 겐지, 아나 등 일반 유저에 비해 최상위 유저의 픽률이 높았던 영웅들이 대거 포함되었습니다.

최초 변수의 개수가 많고, 어느정도 영웅 선택의 경향이 보여서 PCA나 SVD로 통합적인 변수를 추출하려는 시도를 해보았었습니다. 하지만 데이터셋의 문제인지 학습 오류의 문제인지, 최적의 차원 개수가 2개로 나왔으나 플롯상 의심쩍은 부분이 많아 피쳐셋에 반영하지 않기로 결정했습니다. 

![PCA를 통한 Feature Extraction 결과](/assets/overwatch/overlog_pca.png)

user_class 0, 즉 최상위권의 분포는 일반 유저의 분포와 거의 동일하게 분포하는.. 여튼 차원축소가 잘 안되었습니다.

<hr>

## Model Testing
<br>

먼저 모델을 만들기에 앞서, 전체 19,999명의 데이터를 Training Set과 Test Set으로 9:1로 분류하였습니다. 모델은 Training Set을 사용해서 학습하고, 최종적인 성능 비교는 Test Set으로 하게 됩니다.

천상계 분류기와 같은 분류 문제에는 적용할 수 있는 알고리즘이 굉장히 많습니다. 여기에서는 다음과 같은 옵션을 활용해보았습니다.

1. MinMaxScaler + SelectKBest + LogisticRegression
2. MinMaxScaler + SelectKBest + RandomForestClassifier
3. MinMaxScaler + SelectKBest + SupportVectorClassifier
4. MinMaxScaler + DNNClassifier

* MinMaxScaler: 모든 변수를 0~1 사이의 값으로 변환
* SelectKBest: 각 분류기에 넣는 변수 중 설명력이 높은 K개의 변수를 추림
* LogisticRegression: 타겟 변수가 범주형일때 활용할 수 있는 회귀모델
* RandomForestClassifier: 다수의 의사결정나무모형을 만들고 투표를 통해 모델을 수립
* SupportVectorClassifier: 차원을 늘려나가면서 최적의 분류를 수행하는 초평면을 탐색
* DNNClassifier: TensorFlow의 DNNClassifier로 은닉층이 많은 인공신경망 모델

### Pipeline과 GridSearchCV
이번 분석에서는 Scaler, 변수 선택, 모델을 이어붙이고, 여러 모델을 테스트하다보니 반복되는 과정이 많습니다. 또한 각 단계에서 파라미터를 어떻게 설정하느냐에 따라 성능이 달라질 수 있어 테스팅이 복잡해질 수 있는 여지가 많은데, sklearn에서 제공하는 Pipeline과 GridSearchCV를 사용하면 간결하면서도 오류 발생 가능성을 최소화할 수 있습니다. Pipeline은 여러 과정을 하나의 파이프라인으로 묶어 인풋변수만 바꾸면 바로 테스트셋에도 동일한 기준을 적용할 수 있게 해줍니다. GridSearchCV는 입력하는 변수들의 조합을 만들어 3개의 cross validation set에 테스트한 후, 가장 성능이 좋은 모델을 뽑아줍니다.

예를 들어 1번 프로세스를 보면..

{% highlight python %}

def minmax_logistic(X_training, y_training):
    ## scaler, selectKbest, logistic 모형 설정
    scaler = MinMaxScaler(feature_range = [0,1])
    select = feature_selection.SelectKBest()
    logistic_fit = LogisticRegression()

    ## 설정한 순서를 pipeline_object에 할당
    pipeline_object = Pipeline([('scaler', scaler), 
                                ('feature_selection', select), 
                                ('model', logistic_fit)])
    
    ## 각 순서별로 튜닝하고 싶은 파라미터와 파라미터 집단 설정
    tuned_parameters = [{'feature_selection__k': [3, 5, 7],
                        'model__C': [0.01,0.1,1,10],
                    'model__penalty': ['l1','l2']}]

    ## GridSearchCV에 파이프라인과 파라미터 전달. roc_auc 점수로 최적 모델 선정.
    cv = GridSearchCV(pipeline_object, 
                      param_grid=tuned_parameters, 
                      scoring = 'roc_auc')
    
    ## training set으로 GridSearchCV 학습
    cv.fit(X_training.values, y_training['user_class'].values)
    
    ## 최고 AUC 점수 출력
    print("Best AUC score: ", cv.best_score_)
    return cv

cv_log = minmax_logistic(X_training[feature_list], y_training)

{% endhighlight %}

위의 함수에서 cv라는 이름으로 학습한 로지스틱분류기를 리턴하도록 해두었는데, 여기서 `.best_estimator_`로 학습한 최적 파라미터에 직접 접근할 수 있습니다. 로지스틱회귀는 회귀식에서 사용된 각 변수의 계수값을 알 수 있는데요 다음과 같이 뽑을 수 있습니다.

{% highlight python %}

fil = cv_log.best_estimator_.named_steps['feature_selection'].get_support()
selected_features = list(compress(feature_list, fil))
logistic_coeff = pd.DataFrame(cv_log.best_estimator_.named_steps['model'].coef_[0],
             selected_features, columns=['coefficient'])
logistic_coeff.sort_values(by='coefficient')

{% endhighlight %}


![로지스틱회귀의 계수값](/assets/overwatch/overlog_logistic.png)

최상위권의 분류 레이블이 0으로 되어있기에 계수(coefficient)가 모두 -로 나온 것 같습니다. 계수값의 크기로 미루어보아 승률과 플레이시간이 가장 영향을 많이 미친 것 같습니다. 승률은 그렇다쳐도 플레이 시간이 꽤 중요한 요소로 나온 것은 약간 의외였습니다.

마찬가지로 Random Forest와 Support Vector Classifier도 `pipeline`과 `gridSearchCV`를 포함한 함수를 만들어줍니다. Random Forest는 `n_estimators`와 `min_samples_split`을, SVM에는 `kernel`과 `C`값에 여러 파라미터 옵션을 만들어서 테스트를 해보았습니다.

랜덤포레스트 역시 feature_importance를 뽑아줍니다.
![랜덤포레스트 피쳐중요도](/assets/overwatch/overlog_imp.png)

랜덤포레스트 모델에서도 역시 승률과 플레이시간이 가장 중요한 변수로 나왔습니다. 다만 로지스틱 회귀모델과는 달리, 5개의 변수가 최적의 변수로 도출되었습니다.

(앞서 준비했던 수동/자동 피쳐셋 중 1, 2, 3번 모델에서는 모두 자동 피처셋이 미세하게 성능이 좋았습니다.)

<br>

### TensorFlow를 활용한 DNNClassifier

최근 핫한 TensorFlow도 분류 모델링에 빠질 수 없죠. TensorFlow를 활용해본 적은 없지만 공식홈페이지에 올라온 튜토리얼을 참고해서 간단하게 은닉층 4개짜리 모형을 만들어보았습니다. 아직 gridSearchCV나 cross-validation을 결합하는 방법에 대해서는 이해가 부족하여, 앞서 사용했던 training set을 7:3의 비율로 다시 training set과 validation set으로 나눈 후에 모델을 학습합니다. 그리고 최종 test set을 대상으로 성능을 평가해보겠습니다.

{% highlight python %}
import tensorflow as tf

classifier = None

def tf_iteration():

    ## 학습횟수마다 주기적으로 validation set에 대한 성능을 평가하고 리스트에 저장합니다.
    acc_list = []

    ## 수동피쳐셋 10개를 활용하고, 은닉층은 4층으로 각 층마다 12개, 20개, 15개, 12개의 노드를 배치합니다.
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=10)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[12, 20, 15, 12],
                                            n_classes=2,
                                            model_dir="/tmp/newd_30")

    ## 한번 돌때마다 2000번씩 학습하며 이를 50번 반복하여 auc 점수가 가장 높은 포인트를 찾습니다.
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

![DNN의 AUC 추이](/assets/overwatch/overlog_dnn_auc.png)

총 10만번까지 모델을 학습시켰는데 38000번대에 정점을 기록한 이후에는 소폭 하락하여 특정 수준으로 수렴했습니다. 위의 for-loop을 수정하여 38000번까지만 학습시킨 최종모델을 test셋을 기준으로 평가합니다. 

은닉층을 4층이나 둬서 인공신경망 모형을 만들어본 것은 이번이 처음인데, TensorFlow에서 제공하는 api가 간결하고 정리가 잘 되어있어서 사용하는 것 자체에는 큰 어려움이 없었습니다. 하지만 아직 이론적인 이해나 모델 튜닝에 대한 감이 부족하다보니, 은닉층의 갯수나 각 층의 노드 수, 학습 횟수, dropout 비율을 조정해가는 것에 시간을 많이 할애했습니다. 특히 이 데이터셋 같은 경우에는 iris 데이터셋과는 달리 적은 step 횟수에서는 auc 스코어가 너무 낮게 나와 학습을 굉장히 많이 돌려야했습니다. 그 결과 원하는 수준의 validation auc score를 얻었기는 하지만 그렇게 많이, 오래 학습하는 것이 맞는지에 대해서는 아직 의문점이 많습니다.

<hr>

## Model Evaluation
<br>

앞서 학습에서 제외시켰던 Test Set을 활용하여 4가지 모델을 평가해봅시다. 모델 평가를 위해 다음과 같은 함수를 만들었습니다.

{% highlight python %}

def model_tester(cv, X_test, y_test):
    y_pred = cv.predict(X_test).astype(int)
    report = metrics.classification_report(y_test['user_class'], y_pred)
    print(report)

{% endhighlight %}

classification_report에서 생성해주는 여러 지표 중, 여기서는 최상위군 유저 분류의 f1-score를 기준으로 최적합 모델을 선정합니다.

### LogisticRegression
![Logistic_결과](/assets/overwatch/overlog_logi_result.png)
<br>
### RandomForestClassifier
![RF_결과](/assets/overwatch/overlog_rf_result.png)
<br>

### SupportVectorClassifier
![SVM_결과](/assets/overwatch/overlog_svm_result.png)
<br>
### DNNClassifier (수동피쳐셋 사용)
![DNN_결과](/assets/overwatch/overlog_dnn_result.png)
(DNN에서는 수동피쳐셋의 성능이 더 좋았음)
<br>

최상위 유저 분류의 F1-score를 기준으로 평가하자면 DNN > RF > SVM > Logistic 순으로 성능이 좋게 나왔습니다. DNN이 가장 성능이 좋기는 했지만 RF나 SVM도 파라미터나 피쳐 조정에 따라 더 개선될 여지가 있을 듯 합니다. (100,000번 학습한 DNN은 test f1-score가 0.92가 나왔으나, validation auc를 기준으로 하여 38,000번 학습한 DNN으로 성능 평가를 진행했습니다.)

<hr>

## Prediction

이제 크롤링하지 않은 실제 데이터를 넣어서 분류가 제대로 이루어지는지 확인해봅시다. 앞 단계에서 가장 높은 성능을 보여주었던 DNNClassifier를 사용해보겠습니다.

먼저 제 오버워치 기록을 넣어보겠습니다.
![junk3 경쟁전 기록](/assets/overwatch/overlog_junk3.png)
<br>
![응 넌 심해야](/assets/overwatch/overlog_junk3_result.png)
제 기록은 1, 즉 일반 유저로 분류되었습니다... 

그럼 overlog 순위표에서 랜덤으로 5명을 고른 후 테스트해보겠습니다.
![랜덤 테스트 결과](/assets/overwatch/overlog_random_test.png)
2번째 유저가 실제로는 3773점으로 최상위권 유저이나 분류 결과는 일반유저로 오분류되었습니다. 승률과 킬뎃이 낮고 플레이시간이 짧아 일반 유저로 분류된 것으로 보입니다.

<hr>

## 마치며

이번 포스팅에서는 overlog.gg에서 유저 전적 정보를 수집한 후, 다양한 피쳐를 만들고 선택하여 최상위권 유저를 분류하기 위한 여러 모델을 만들어봤습니다. 여러 모델을 테스팅해본 결과, 제가 설정한 파라미터 환경에서는 은닉레이어를 4층으로 쌓은 딥러닝 분류기가 가장 예측력이 좋았습니다. 이번에는 가장 기본적인 fully connected 딥러닝 모형으로 테스트해보았지만, 이미지처리에 자주 쓰이는 CNN으로도 예측을 해볼 수 있다고 합니다. 혹 관심있으신 분이 있을까하여 training set과 test feature를 <a href="https://github.com/junkwhinger/overlog">github</a>에 올려두었습니다. 예측 모델을 만들고 X_test를 넣어 예측한 결과값 csv를 junsik.whang@gmail.com으로 보내주시면 성능 평가 결과를 보내드리겠습니다! :)