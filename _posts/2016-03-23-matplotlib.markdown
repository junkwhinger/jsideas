---
layout: post
title:  "Matplotlib을 활용한 이세돌 vs. 알파고 데이터 시각화"
date:   2016-03-23 00:34:25
categories: python
image: /assets/alphago/header.png
---

## 들어가며
<br>
그동안 써온 블로그글 중 계속 꾸준히 방문자 수가 이어지는 글은 <a href="http://jsideas.net/python/2015/08/11/histogram_tutorial_part2.html">파이썬 초보 - pandas와 matplotlib을 활용한 간단 분석 part 2</a>입니다. 심지어 이 글은 구글에 'pandas 히스토그램'으로 검색을 하면 검색 상위에 올라올 정도입니다.. 아마 pandas와 히스토그램을 글에서 많이 언급했기 때문이 아닌가 싶습니다.

최근에는 d3.js 쪽 업데이트를 자주 해왔었는데, 이번에는 <a href="http://jsideas.net/python/2016/03/16/Google_DeepMind.html">이세돌 vs. 알파고 분석</a>을 하는데 아주 요긴하게 사용했던 matplotlib과 pandas를 또 간단히 다뤄보도록 하겠습니다. 

<hr>

## 데이터셋
<br>
분석에서 사용할 데이터셋은 제가 직접 수기로 받아적은 알파고 vs. 이세돌 데이터입니다. 로데이터는 <a href="https://github.com/junkwhinger/AlphaGo_raw">GitHub</a>에서 바로 받아가실 수 있습니다.

<hr>

## 분석 방향
<br>
인간과 기계는 어떻게 다르게 생각할 것인가? 그리고 그 차이는 어떤 결과를 만들어낼 것인가? 이번 이세돌 9단과 알파고의 대결을 지켜보며 가장 관심있게 바라본 부분이었습니다. 연산이 아닌 직관의 영역이기에 인간이 우세할 것이다라는 기존의 낙관론은 2국이 끝나있을 때 '인간이 단 한판이라도 이길 수 있을까'하는 비관론으로 변해있었습니다.

첫 대국에서 이세돌 9단의 떨리는 손을 바라보며 표정도, 분위기도 읽을 수 없는 기계를 상대로 게임을 한다는 것이 얼마나 무서울까-싶었습니다. 장고를 거듭하는 이 9단과 달리, 알파고는 거의 일정한 시간 내에 착수를 마쳤습니다. 이 둘의 착수시간 패턴을 시각화할 수 있다면, 인간다움과 기계다움을 대조해서 표현할 수 있지 않을까- 싶었습니다. 그런 생각에서 간단히 손으로 데이터를 모으기 시작했지요.

게임 당 이세돌 9단과 알파고는 각각 2시간을 가지고 시작합니다. 상대방이 착수할 때마다 시간 카운트 다운이 시작되며, 착수하면 시간이 멈춥니다. 2시간이 모두 소진되면 초읽기가 시작됩니다. 이 9단과 알파고 중 처음 2시간을 먼저 사용한 쪽은 어디일까요. 그리고 각 대국 당 소진되는 속도는 어떻게 다를까요? 이번 글에서는 착수시간으로 시각화했던 차트 중에서 'Remaining Time'을 ipython notebook을 사용해서 순서대로 뽑아보겠습니다.

<hr>

## 환경 구축
<br>
본 분석을 ipython notebook을 사용해 진행하며, python 3.5를 사용했습니다. 보다 편리한 분석 환경 구축을 위해 virtualenv 사용을 권해드립니다. 또한 라이브러리는 pandas와 matplotlib을 사용했습니다.

<hr>

## 라이브러리 불러오기
<br>
{% highlight python %}
import pandas as pd #1
import matplotlib.pyplot as plt #2
import numpy as np #3
%matplotlib inline #4
import warnings #5
warnings.filterwarnings('ignore') #5
{% endhighlight %}

1)가장 주요하게 사용할 pandas 패키지를 불러옵니다.<br>
2)시각화에 사용할 matplotlib 패키지를 불러옵니다.<br>
3)x축 레이블 간격에 사용할 numpy 패키지를 불러옵니다. 조금 과하긴 하지만 편하니 불러옵니다.<br>
4)ipython notebook에서 만드는 차트를 별도의 창이 아닌, notebook내에서 보기 위한 설정입니다.<br>
5)간혹 ipython이 실행은 시켜주나 depreciation이 예정된 기능에 대해 경고창을 띄웁니다. 빨간색이 보기 싫거나 한번에 모든 명령을 돌리고 싶은 경우, 경고를 꺼두면 편합니다.<br>

<hr>

## 데이터 경로 정하기
<br>
{% highlight python %}
game1 = '/users/jun/python/alphago/first_game.csv'
game2 = '/users/jun/python/alphago/second_game.csv'
game3 = '/users/jun/python/alphago/third_game.csv'
game4 = '/users/jun/python/alphago/fourth_game.csv'
game5 = '/users/jun/python/alphago/fifth_game.csv'
{% endhighlight %}

바로 명령어를 사용해서 파일을 읽어들이기보다는 경로를 미리 지정해줍니다. 특히 다수의 파일을 사용하는 경우, 경로를 따로 정해주면 좋습니다. 후에 파일명이 변하게 되어도 찾기가 편합니다.

<hr>

## 데이터 전처리
{% highlight python %}
def time_func(x):
    if pd.isnull(x):
        return 0
    else:
        k = x.split(":")
        hour = int(k[0])
        minute = int(k[1])
        second = int(k[2])
        ts = hour * 3600 + minute * 60 + second
        return ts
{% endhighlight %}

먼저 01:41:10과 같은 형태를 취하고 있는 시간을 초 단위로 변환하기 위한 커스텀 함수를 만들어줍니다. 뒤에서 `.apply(lambda x: custom_func(x))` 구문을 사용할 함수로, 시간을 담고 있는 데이터프레임 컬럼의 각 value에 적용합니다. 일단 time_func라는 이름으로 만들어둡니다.

로직은 간단합니다. value가 x라는 인자로 넘어오게 되는데, 이 x가 null값인 경우 0이, 아닌 경우 else: 이후 구문이 실행됩니다. 01:41:10이라는 string이 넘어온 경우, 이를 : 단위로 자른 후 시간, 분, 초에 할당한 후, 초 단위로 통일하여 ts에 저장, 반환합니다.

이 함수는 바로 다음에 이어질 함수안에서 돌아갑니다. 원래는 하나의 함수가 아니라 다 풀어져있었으나, 동일한 방법으로 여러 파일을 처리하게 되다보니 과정을 효율화시키기 위해 하나의 함수로 묶어 정리해주었습니다.

{% highlight python %}
def countdown(a_file):
    data = pd.read_csv(a_file)
    lee_b4_countdown = len(data.Lee_Sedol.dropna())
    al_b4_countdown = len(data.AlphaGo.dropna())
    b4_countdown = max(lee_b4_countdown, al_b4_countdown)
    data['Lee_Sedol_cl'] = data.Lee_Sedol[:b4_countdown]
    data['AlphaGo_cl'] = data.AlphaGo[:b4_countdown]
    ts_df = data[['Lee_Sedol_cl', 'AlphaGo_cl']]
    ts_df['Lee_Sedol_ts'] = data.Lee_Sedol_cl.apply(lambda x: time_func(x))
    ts_df['AlphaGo_ts'] = data.AlphaGo_cl.apply(lambda x: time_func(x))
    temp_df = ts_df[['Lee_Sedol_ts', 'AlphaGo_ts']]
    result_df = temp_df[(temp_df.Lee_Sedol_ts != 0) | (temp_df.AlphaGo_ts != 0)] / 60
    return result_df
{% endhighlight %}

다음은 countdown이라는 커스텀 함수입니다. 여기에서는 인자가 파일 하나로 들어갑니다. 앞에서 밝힌 바와 같이 반복작업의 효율화를 위해 하나의 함수에 몰아넣었습니다.

먼저 `pd.read_csv()`함수를 이용해 a_file이라는 인자로 넘어온 파일 경로를 읽어들입니다.

데이터셋을 살펴보시면 아시겠지만, 초읽기에 들어간 후에는 Lee_Sedol과 AlphaGo 행은 공란으로 기록되어있습니다. 공란은 당연히 시간 변환이 되지 않습니다. 초읽기전의 2시간 데이터만 다루기로 했으므로, 공란 제외를 위해서 `.dropna()` 함수를 사용합니다. 그리고 len()을 덮어씌워 초읽기 전까지의 착수지점(index)를 확보하고, 이를 통해 초읽기 전까지 데이터를 자릅니다.

`ts_df`에 초읽기 전까지 데이터를 집어넣은 후, 앞서 정의한 time_func 함수를 사용합니다. `.apply(lambda x: time_func(x))`를 사용하여 손쉽게 ts를 구합니다. 마지막으로 둘다 초읽기에 들어간 상황, 즉 ts가 0이 된 행을 제외한 result_df를 반환합니다.

여러 스크립트를 몰아넣었지만, def 이후를 따로 떼어다 파일경로를 넣고 실행하면 각 단계별 아웃풋을 확인할 수 있습니다.

이제 앞서 불러온 파일경로에 countdown 함수를 실행해봅니다.

{% highlight python %}
game1_ts = countdown(game1)
game2_ts = countdown(game2)
game3_ts = countdown(game3)
game4_ts = countdown(game4)
game5_ts = countdown(game5)
{% endhighlight %}

잘 처리되었는지 확인해볼까요?
`game1_ts.tail(10)`으로 데이터프레임의 맨 끝 10줄을 봅시다. 
![game1_ts](/assets/alphago/game1_ts.png)

초가 아닌 분 단위로 ts가 잘 정리되어 들어왔습니다. 마지막으로 모든 게임을 하나의 df에 묶기 전에 각 게임의 착수 순번을 기록해두어야 합니다. 현재 착수 순번은 index에 기록되어있습니다. `.reset_index`함수를 사용하면 index를 개별 컬럼으로 손쉽게 빼낼 수 있습니다. 

{% highlight python %}
game1_ts.reset_index(level=0, inplace=True)
game2_ts.reset_index(level=0, inplace=True)
game3_ts.reset_index(level=0, inplace=True)
game4_ts.reset_index(level=0, inplace=True)
game5_ts.reset_index(level=0, inplace=True)
{% endhighlight %}

어디 잘 적용되었는지 테스트해볼까요?

pandas의 특출난 장점 중 하나는 df 뒤에 `.plot()`만 붙여서 바로 차트를 그릴 수 있다는 점입니다. 아주 쓸만하죠. x축을 index로 지정하여 라인플롯을 그려봅시다. 

![game4_ts_plot](/assets/alphago/game4_ts_plot.png)

<hr>

## 전체 남은 시간 차트 그리기 (준비)

자, 이제 5개 대국 데이터를 모두 합쳐서 이세돌 9단과 알파고가 2시간을 어떻게 사용했는지 그려봅시다. 

먼저, 각 대국을 구분할 수 있도록 컬럼명을 바꿔줍니다. 선수 이름 앞에 게임명을 붙여줍니다.

{% highlight python %}
game1_ts.columns = ['index', '[G1]Lee_Sedol', '[G1]AlphaGo']
game2_ts.columns = ['index', '[G2]Lee_Sedol', '[G2]AlphaGo']
game3_ts.columns = ['index', '[G3]Lee_Sedol', '[G3]AlphaGo']
game4_ts.columns = ['index', '[G4]Lee_Sedol', '[G4]AlphaGo']
game5_ts.columns = ['index', '[G5]Lee_Sedol', '[G5]AlphaGo']
{% endhighlight %}

다음은 df 합치기입니다. 모든 df를 이어붙이는 작업인데, append를 사용해서 행 붙이기를 합니다. 
{% highlight python %}
g_df = game1_ts.append(game2_ts).append(game3_ts).append(game4_ts).append(game5_ts)
{% endhighlight %}
이런 식으로 `.append` 함수를 연속적으로 사용하면 됩니다. g_df를 출력하면 null 값이 굉장히 많이 보입니다. 이는 합치는 df의 컬럼이 index를 제외하고는 다 다르기 때문입니다. 다른 열에 대해서 행 붙이기를 하게 되면, 다른 제목을 가진 열에는 붙일 내용이 없으므로 null이 들어가게 됩니다. 
여기서 왜 이런 식으로 df를 합쳤냐라는 의문이 들게 되는데, pandas의 plot에서는 열 이름으로 카테고리를 구분해주기 때문입니다. (분명 직접 지정해주는 방법도 있을 듯 합니다.) 게임명 + 선수명으로 열 이름이 구분되어야 다음에서 깔끔하게 플롯을 그릴 수 있습니다.

마지막으로 x축을 최대 착수 순번에 맞추기 위해서 5국 전체를 통틀어 가장 긴 착수 순번을 뽑습니다. 
`max_length = max(g_df.index)`

<hr>

## 이제 진짜로 그리기
<br>
이제 다왔습니다. 그려봅시다.

{% highlight python %}
fig = g_df.plot(x='index', color=['#f6e8c3','#c7eae5','#dfc27d','#80cdc1','#bf812d','#35978f','#8c510a','#01665e','#543005','#003c30'], marker='o', markersize=6, alpha=0.8, linewidth=3, fontsize=15, figsize=(20, 8))

fig.set_axis_bgcolor('white') 
plt.xticks(np.arange(0, max_length + 2, 5))
plt.grid(b=True, which='major', color='0.8',linestyle='-')

plt.title('Lee Sedol vs. AlphaGo: Remaining Minutes', size=25)
plt.ylabel('Remaining Minutes', size=15)
plt.xlabel('Turn Index', size=15)
plt.legend(prop={'size':15})

plt.show()
{% endhighlight %}
자 먼저, 앞선 게임4 그래프와 동일하게 전체 df인 g_df에 `.plot`을 붙여 차트를 그립니다. 여기에 몇가지 추가 파라미터가 붙습니다. `color` 인자로 컬럼별 색상을 지정해줄 수 있습니다. <a href="http://colorbrewer2.org/">colorbrewer2</a>를 사용하면 아주 효과적인 색상 선택을 할 수 있습니다. 여기서는 색맹인 분들도 차트를 볼 수 있도록 색상을 선택했습니다. 이 외에도 마커타입, 선 굵기, 폰트 크기를 설정할 수 있으며, 특히 figsize로 차트의 크기를 설정해줍니다.

그 다음은 x축의 tick 인터벌 조정입니다. 여기서 numpy를 사용합니다. `np.arange`를 통해 0부터 최대 착수 순번까지 5씩 건너뛰는 수열을 만들고, 이를 x축 인터벌로 지정합니다. 최대 순번에 2를 더하여 끝자락에 여유를 두었습니다.

마지막으로 x, y 레이블과 차트 이름을 붙인 후 `plt.show()`로 완성된 차트를 띄웁니다.

빠밤!

![total_remaining_plot](/assets/alphago/total_remaining.png)
