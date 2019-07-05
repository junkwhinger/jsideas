---
layout: post
title:  "파이썬 초보 - pandas와 matplotlib을 활용한 간단 분석 part 2"
date:   2015-08-10 20:34:25
img: 20150811.png
tags: [python, data analytics]

---

(이미지 출처: <a href="http://stackoverflow.com/questions/27446455/pandas-subplots-in-a-loop">stackoverflow</a>)
<h2>파이썬 초보 - pandas와 matplotlib을 활용한 간단 분석 part 2</h2>
<p><a href="http://jsideas.net/python/2015/08/10/histogram_tutorial.html">part 1</a>에 이은 pandas와 matplotlib을 활용한 간단 분석 두번째 파트입니다.

파트 1에서는 분석을 위한 파이썬 개발환경을 갖추고, ipython에서 pandas 라이브러리를 활용하여 간단하게 데이터를 만져봤습니다. 파트2에서는 앞서 다루지 않았던 몇가지 함수를 더 소개하고, matplotlib 라이브러리를 활용해 히스토그램과 박스플롯을 뽑아보겠습니다.</p>


<h3>파이썬 pandas 만져보기 2</h3>

* `data['Seoul'] = 'in seoul'` - 데이터에 새로운 컬럼을 추가합니다. 앞서 컬럼을 선택할 때 `data.hourly_wage`로 선택을 했는데, `data['hourly_wage']`로도 동일한 기능을 수행할 수 있습니다. 단, 새로운 컬럼을 만드는 경우, `.`이 아닌 `[컬럼명]`으로 할당해주어야 기존 데이터프레임에 추가가 됩니다. 여기서는 `Seoul`이라는 컬럼을 새로 만들고 `in seoul`이라는 문자열을 입력했습니다.

* 커스텀 함수 만들기 - 이번에는 간단한 함수를 정의하고 이를 사용해서 컬럼을 만들어보겠습니다. 먼저 함수명과 그것이 사용할 인자 x를 정의합니다. 함수 `more_than_6000`에 인자 `x`가 들어오면, if문이 발동되는데, 만약 x가 6000보다 클 경우 `1`을 반환하고, 그렇지 않을 경우 `0`을 반환하도록 합니다. 코드로 구현하면 아래와 같이 구성할 수 있습니다. ipython에서는 `:`을 입력하면 자동으로 줄바꿈과 들여쓰기가 되어 그대로 치면 됩니다.
{% highlight python %}
def more_than_6000(x):
  if x > 6000:
  	return 1
  else:
  	return 0
{% endhighlight %}

* `data['more_than_6000'] = data.hourly_wage.map(lambda x: more_than_6000(x))`
앞서 정의한 함수를 사용하여 새로운 컬럼을 뽑았습니다. 약간 식이 복잡한데, 큰 틀은 위에서 문자열을 넣었던 방식과 동일합니다. `map` 함수는 그 안의 시급 데이터를 하나하나 선택하며 원하는 함수를 쓸 수 있도록 해주는데, 여기서는 `lambda x`라는 anonymous function(이름 없는 함수)를 써서 각 시급 데이터에 `more_than_6000` 함수를 적용했습니다. `시급 컬럼의 각 데이터를 돌아가며 함수를 쓸껀데, 그 함수는 6000보다 크면 1이고 아니면 0이다`라는 문장이 있다면 `함수를 쓸껀데, 그 함수는..` 부분이 람다 오퍼레이터가 수행하는 부분입니다. (설명이 야매라 죄송합니다 ㅠ) 여튼 결과가 잘 나왔는지 확인해봅시다. `data[data.more_than_6000 == 1].describe()`

![람다 오퍼레이터 적용 결과](/assets/materials/20150811/6.png)

min값이 6100원인걸 보면 정상적으로 잘 뽑혔네요!

* 마지막으로 `구`과 `6000원 이상` 시급을 새로운 데이터 프레임에 넣어 봅시다. `data2 = data[data.more_than_6000 == 1][['area1', 'hourly_wage']]`를 실행합시다. `data2`는 전체 데이터에서 6000원이 넘는 시급을 가진 행을 추출한 후, 그 행의 구(`area1`)과 시급(`hourly_wage`)를 추린 서브 셋입니다.

![서브셋의 info](/assets/materials/20150811/7.png)

* `data2.to_csv('data2.csv', index=False)`를 실행하여 csv 데이터로 뽑습니다. 여기서 index=False는 pandas DataFrame이 자동으로 부여하는 행번호를 출력하지 않겠다는 의미입니다. 잘 뽑혔는지 확인해봅시다.

![csv로 출력](/assets/materials/20150811/8.png)

정상적으로 잘 뽑혔네요.

<hr />

<h3>스크립트화 및 ipython에서 돌려보기</h3>

지금까지 만들었던 코드 중 필요한 부분을 뽑아 파이썬 스크립트로 만들고 이를 ipython에서 돌려봅니다.

- 필요한 라이브러리 불러오기
- csv 파일 불러와서 pandas 데이터프레임으로 저장하기
- 데이터 프레임 확인하기
- `more_than_6000` 함수 정의하기
- `more_than_6000` 열 추가하기
- 서브셋 만들기
- csv 파일 출력하기

지난 시간에 만들었던 빈 파일인 `test.py`를 열고 코드를 입력해봅시다.

{% highlight python %}
# -*- coding: utf-8 -*-

# pandas 라이브러리 불러오기
import pandas as pd

# csv 파일 불러와서 pandas 데이터프레임으로 저장하기
data = pd.read_csv('convenient_store.csv')

# 데이터 프레임 확인하기
print data.head()
print data.info()
print data.describe()

# `more than 6000` 함수 정의하기
def more_than_6000(x):
	if x > 6000:
		return 1
	else:
		return 0

# 'more_than_6000' 열 추가하기
data['more_than_6000'] = data.hourly_wage.map(lambda x: more_than_6000(x))

# '서브셋 만들기'
data2 = data[data.more_than_6000 == 1][['area1', 'hourly_wage']]

# csv 파일로 출력하기
data2.to_csv('data2.csv', index=False)

{% endhighlight %}

!주의! 위에 코드를 보시면 `# -*- coding: utf-8 -*-`이라는 녀석이 붙어있습니다. 인코딩 기준을 utf-8으로 설정해주는 건데, 이 한줄을 넣지 않고 그냥 코드를 돌리게 되면 `SyntaxError: Non-ASCII character '\xeb' in file test.py on line 1, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details` 이런 괴랄한 오류가 뜹니다. 데이터가 한글로 되어있어서 이런 문제가 발생하는데 저 코드 한줄이면 문제가 일단(!) 해결됩니다.

ipython 환경에서 나오는 방법은 `control + D`를 눌러 y를 치면 됩니다. 다시 가상환경으로 돌아와서 `python test.py`를 실행해 방금 짠 스크립트를 돌려봅시다. 

![터미널에서 파이썬 스크립트 돌리기](/assets/materials/20150811/10.png)

오류 없이 잘 돌아갑니다. 이번에는 같은 코드를 ipython 환경에서 돌려봅시다. ipython 환경에서 돌리는 장점 중 하나는 코드를 돌렸을 때 그 안에서 정의한 변수를 계속 활용할 수 있다는 점입니다. 

* `ipython` - ipython 실행
* `%run test.py` - 터미널과 달리 스크립트 파일을 돌릴때는 %run을 치고 파일명을 쳐줍니다.
* `data.head()`

![ipython에서 파이썬 스크립트 돌리기](/assets/materials/20150811/11.png)

역시 잘 돌아갔습니다. ipython에 데이터를 올려놨으니 이제 시각화를 돌려봅시다.

<h3>파이썬 matplotlib 만져보기 2</h3>

<a href="http://matplotlib.org/">matplotlib</a> 라이브러리는 파이썬을 활용한 데이터 시각화에 자주 쓰이는 좋은 라이브러리입니다. 이 라이브러리를 활용해서 히스토그램, 박스플롯 같은 차트를 쉽고 빠르게 그릴 수 있습니다.

* `import matplotlib.pyplot as plt` - matplotlib 라이브러리 중 pyplot이라는 모듈을 `plt`라는 이름으로 불러옵니다. (이름이 길어서..)
* `data.hourly_wage.hist(bins=10)` - 먼저 히스토그램을 그려봅시다. 전체 편의점 데이터의 시급을 히스토그램으로 그리는데, 구간은 10개 구간으로 정하겠습니다. 이 명령어를 입력하면 `<matplotlib.axes._subplots.AxesSubplot at 0x1111dfc10>` 이런 결과를 뱉는데, matplotlib 플롯을 만들어놨다는 얘깁니다. 
* `plt.show()`를 돌려 그래프를 뽑아봅시다.

![matplotlib 히스토그램](/assets/materials/20150811/12.png)

잘 뽑혔습니다. 위에 명령어에서 bins 값을 조정하거나, 조금 더 나아가서 여기서 다루지 않는 numpy의 linspace 함수를 쓰면 최소값, 최대값, 구간 수를 정해 더 정확한 구간을 가진 히스토그램을 뽑을 수 있습니다. `plot.show`를 치면 새로운 창이 열리면서 그래프가 뜨는데, 이 그래프 창을 꺼야만 ipython에서 다시 입력이 가능해집니다.

* `data.boxplot(column='hourly_wage')` - 이번에는 같은 데이터를 박스플롯으로 표현해봅시다. `plt.show()`를 입력해 시각화된 차트를 띄웁니다.

![matplotlib 박스플롯](/assets/materials/20150811/13.png)

* `data.boxplot(column='hourly_wage', by='name')` - 이번에는 회사별로 박스플롯을 돌려봅시다. 데이터에 `name`이라는 컬럼에 회사명이 표시되어있습니다. 이를 `by='name'`이라는 부분에 집어넣어 옵션으로 지정해주면 간단하게 박스플롯 3개가 한번에 그려집니다.

![회사별 박스플롯](/assets/materials/20150811/14.png)

* `data.boxplot(column='hourly_wage', by='area1')` - 같은 방식으로 구별 히스토그램을 뽑을 수 있습니다.

![회사별 박스플롯](/assets/materials/20150811/15.png)

그런데 ipython에서 한글을 제대로 인식하지 못했네요 ㅠㅠ 내장된 폰트가 한글을 인식하지 못했기 때문인데 이 경우 한글 폰트를 지정해주면 됩니다. `plt.show()`를 치기 전에 `matplotlib.rc('font',family='AppleGothic')`으로 폰트를 지정해주면..

![회사별 박스플롯(한글 수정)](/assets/materials/20150811/16.png)

정상적으로 수정되어 박스플롯이 출력됩니다.

* 이번에는 구별로 회사를 나누어 박스플롯으로 출력하는데, 세로가 아닌 가로로 플롯을 그려봅시다. `data.boxplot(column='hourly_wage', by=['area1', 'name'], vert=False)`를 찍고, 폰트 지정을 해준 후 그래프를 출력합니다. `vert=False`는 박스플롯을 가로로 그리는 옵션입니다.

![구별 회사별 박스플롯](/assets/materials/20150811/17.png)

<hr />
 
<h3>마치며</h3>
파이썬의 pandas와 matplotlib 라이브러리를 활용해 csv 데이터를 불러온 후 기준에 따라 가공하고 히스토그램과 박스플롯으로 시각화해보았습니다. 각 라이브러리의 영문 홈페이지와 개발문서를 보면 본 튜토리얼에서 소개한 몇가지 함수 이외에도 굉장히 많은 유용한 함수가 소개되어있으니, 이를 통해 더 많은 새로운 차트를 그려보세요! 다음번에는 좀 덜 야매스러운 설명으로 튜토리얼을 구성해보겠습니다. 감사합니다!
