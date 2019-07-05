---
layout: post
title:  "파이썬 초보 - pandas와 matplotlib을 활용한 간단 분석 part 1"
date:   2015-08-09 20:34:25
img: 20150811.png
tags: [python, data analytics]

---

(이미지 출처: <a href="http://stackoverflow.com/questions/27446455/pandas-subplots-in-a-loop">stackoverflow</a>)
<h2>파이썬 초보 - pandas와 matplotlib을 활용한 간단 분석 part 1</h2>
얼마전 virtualenv를 활용한 파이썬 가상 환경 구축에 대한 포스팅을 썼습니다. 이번에는 구축한 환경을 바탕으로 파이썬의 pandas과 matplotlib 라이브러리를 활용하여 간단한 히스토그램과 박스플롯을 짜보는 튜토리얼을 소개하고자 합니다.

<h3>엑셀 vs. 파이썬 혹은 R</h3>
최근 데이터 사이언스에 대한 관심이 높아짐에 따라 주변에서도 심심찮게 파이썬과 R 등 프로그래밍 언어를 배우려는 분들을 볼 수 있습니다. 저도 그 중 하나였고요. 프로젝트 수행 중 3만 줄 이상이 넘어가는 엑셀 파일을 여러개 다룬 적이 있었는데, 특정 단어를 포함한 데이터를 필터링하거나 차트를 구성할 때 프로그램이 멈추거나 심하게 느려지는 경우가 있었습니다. 또 전체 데이터셋의 서브 셋을 만들거나 복사본을 만들때 시트를 여러개 만들어야 하는 번거로움도 있었고요. 엑셀에 비해 파이썬과 R은 속도도 빠르고, 원하는대로 함수를 설계하거나 데이터를 유지관리하기 편한 점이 마음에 들었습니다.

한편, 이들 언어를 배우다보면 엑셀의 장점을 더 명확하게 느낍니다. 파이썬과 R은 엑셀처럼 열과 행으로 데이터를 쉽게 보여주거나 실시간으로 바꾸기가 어렵습니다(제가 못하는 걸수도 있지만요). RStudio의 View 기능이나 파이썬의 print를 적절히 활용하면 나름 깔끔하게 볼 수는 있으나, 스크롤를 내리거나 마우스 포인터로 특정 셀을 찍어서 값을 쉽게 바꾸는 건 엑셀만의 특별한 장점입니다. 대부분의 비전공자 분석가가 대규모 데이터 테이블을 다루지는 않는다는 점에서 파이썬과 R은 다루다보면 부딪히는 암초가 한두개가 아닙니다.

하지만 배워놓으면 언젠가 쓸모는 있습니다. 앞서 설명했던 것처럼 파이썬과 R은 엑셀에서 하지 못하거나 엄청나게 돌아가야 하는 복잡한 업무를 쉽게 해결할 수 있도록 해줍니다. 엑셀로도 노가다를 해야하는 일을 몇 분 고민해서 코딩을 하면 몇 초만에 결과를 뽑을 수 있는 매력이 있습니다. 또 머신러닝과 같은 복잡한 연산을 수행할 경우에는 위대한 분들이 만들어놓은 파이썬과 R 라이브러리를 활용할 수 있는 장점이 있지요.

본 튜토리얼에서는 파이썬을 사용해서 어떻게 데이터를 읽어들이는지, 어떤 방식으로 데이터를 관리하고 가공하는지, 마지막으로 어떻게 차트를 뽑는지에 대해 다루고자 합니다. 기록에 틀린 부분이나 추가할 부분이 있다면, 혹은 튜토리얼대로 되지 않는 부분이 있다면 언제든 junsik.whang@gmail.com으로 연락주세요!

<hr />

<h3>개발 환경 구축</h3>
(본 분석은 Mac 개발 환경을 대상으로 하며 Window나 타 운영체제에서는 동일한 명령어가 먹히지 않을 수 있습니다. / 먹히지 않을겁니다 ㅠ)
텍스트 에디터 프로그램을 써서 스크립트를 짜는데, 저는 sublime text3를 사용하고 있습니다. 여기 <a href="http://www.sublimetext.com/3">링크</a>에서 다운받아 실행할 수 있습니다. 또한 본 튜토리얼에서 사용하는 파이썬 프로그램 버전은 2.7.6입니다.

<br>
1. 먼저 가상 환경을 구축할 공간을 만듭니다. 바탕화면에 `pandas_test`라는 폴더를 만들어 봅시다.

* 터미널 실행 (`terminal`) - 화면 우상단 spotlight에서 `terminal`을 찾아 실행시킵니다.
* `cd desktop` - 바탕화면으로 이동합니다. (보통 terminal의 기본설정 상 이 커맨드로 바탕화면으로 이동합니다.)
* `mkdir pandas_test` - mkdir 명령어로 폴더를 만들어줍니다.
* `cd pandas_test` - 새로 만든 폴더로 이동합니다.
* `touch test.py` - test.py라는 파이썬 스크립트 파일을 만듭니다. finder를 통해 바탕화면에 pandas_test 폴더를 열어보세요. 아무것도 쓰여있지 않은 스크립트 파일이 있을 겁니다.
* sublime text3를 실행시키고 비어있는 파일을 불러옵니다.

![terminal, finder, sublime text3 화면](/assets/materials/20150811/1.png)
위에 그림을 보면 terminal 화면에서 `mkdir pandas_test`를 치고 `touch test.py`를 쳤는데, 이러면 이동하기 전 바탕화면에 빈 스크립트 파일을 만들게 됩니다. 즉 잘못친 명령어입니다. 이제 작업 폴더를 만들었으니 가상 개발환경을 만들어봅시다.

2. virtualenv를 사용하여 가상 환경을 만듭시다.

* `virtualenv venv` - 가상 개발환경을 만듭니다. virtualenv가 설치되지 않았다면 <a href="http://jsideas.net/python/2015/07/20/virtualenv.html">여기서</a> 설치 방법을 따라해주세요.
* `source venv/bin/activate` - 가상 환경으로 들어갑니다. 이제부터 앞에 (venv)가 붙기 시작합니다.
* `pip install pandas`- pandas라는 파이썬의 데이터분석 라이브러리를 설치합니다. (인터넷이 연결되어야 다운로드가 이루어집니다.)
* `pip install matplotlib` - matplotlib라는 파이썬 데이터 시각화 라이브러리를 설치합니다.
* `pip install ipython` - ipython이라는 파이썬 개발 프로그램을 설치합니다.
<br>
오케이! 이제 필요한 프로그램은 설치를 마쳤습니다. 아래 링크에서 분석에 사용할 데이터를 다운받아 `pandas_test` 폴더 안에 넣어주세요.
[분석 대상 자료 다운받기](/assets/materials/20150811/convenient_store.csv)

분석에 사용할 자료는 [시급으로 본 서울지역 아르바이트 환경][시급으로 본 서울지역 아르바이트 환경] 분석에서 사용했던 자료 중 편의점에 해당하는 데이터입니다.

<hr />

<h3>파이썬 pandas 만져보기</h3>
자 이제 파이썬을 만져볼 차례입니다. 기본적인 문법 설명이 필요하시면 <a href="https://www.codecademy.com/tracks/python">codecademy</a>에서 문법 튜토리얼을 마치고 오시는 것을 추천드립니다.

* `ipython` - virtualenv가 활성화된 상태에서 이 커맨드를 입력하여 ipython 환경으로 들어갑니다. 
![terminal에서 ipython을 실행시 화면](/assets/materials/20150811/2.png)
* `2+3` - 2 더하기 3을 입력해봅니다. 바로 아래 `Out[1]: 5`로 결과값이 출력됩니다.
* `import pandas as pd` - 바로 pandas 라이브러리를 불러와봅시다. 앞으로 칠 일이 종종 있으니 간단히 pd라는 볆명을 붙여줍니다.
* `data = pd.read_csv('convenient_store.csv')` - 위에서 다운받은 `convenient_store.csv`라는 파일을 불러와서 data라는 변수에 할당합니다. 만약 위의 파일이 최초에 만든 `pandas_test`라는 폴더에 있지 않은 경우 오류 메시지가 뜰테니 꼭 확인하세요!
* `data.head()` - 불러온 데이터의 첫 4개 행을 프린트합니다. 10개 행을 원하면 10을 괄호안에 넣으면 됩니다. `data.head(10)`
* `data.info()` - 불러온 데이터의 특징을 확인해봅니다. 
![불러온 데이터의 특징과 형태](/assets/materials/20150811/3.png)
먼저 `<class 'pandas.core.frame.DataFrame'>`은 변수로 지정한 `data`의 클래스, 즉 형태를 의미합니다. 앞서 `pd.read_csv()`라는 함수를 써서 csv파일을 읽어왔기 때문에 바로 pandas의 DataFrame으로 형태가 변환되었습니다. 그아래를 보면 `177 entries`라는 부분이 있는데 이는 데이터의 레코드 수가 177개라는 것을 의미하고요, `Data columns` 밑에 나오는 7개 컬럼은 데이터의 열을 의미합니다. area는 구+동명, company는 회사명 등등을 의미하고, object와 int64는 해당 컬럼의 데이터 성격을 의미합니다. 어떤 데이터인지 찍어볼까요?

* `data.area` - 전체 데이터에서 area 컬럼에 해당하는 것을 출력해봅니다. 용산구 문배동, 은평구 대조동.. 등 지역명이 나오네요. 마찬가지로 `data.company`, `data.hourly_wage`로 다른 컬럼의 값을 확인할 수 있습니다. 연속형 숫자로 입력된 값은 int64로, 그 이외에는 object로 표현되어 있습니다.
* `data.describe()` - describe() 함수는 전체 혹은 int로 구성된 특정 컬럼 데이터의 빈도, 평균, 편차, 최소값, 25%, 50%, 75%, 최대값을 자동으로 뽑아줍니다. 이 경우 연속형 숫자 컬럼인 `hourly_wage`와 `outlier`의 숫자 분포를 뽑아주네요 (outlier 컬럼의 경우, 박스플롯을 기준으로 outlier인 경우 1을, 아닌 경우 0으로 표시된 범주형 데이터입니다.) `hourly_wage`에 대한 분포를 뽑고 싶은 경우 `data.hourly_wage.describe()`를 입력하면 되겠죠?
* `data[data.hourly_wage > 6000]` - 이번에는 필터를 걸어보겠습니다. 전체 데이터를 대상으로 시급이 6000원 이상인 항목을 프린트해봅니다.
* `data[(data.area1 == '마포구') & (data.hourly_wage > 6000)]` - 이중으로 필터를 넣어봤습니다. `&`는 and, `|`는 or를 의미합니다. 여기서는 마포구에 있으면서 시급이 6천원보다 큰 데이터를 뽑아봤습니다. 마포구 동교동 세븐일레븐 1건이 나옵니다.
![데이터 기준을 활용한 필터링](/assets/materials/20150811/4.png)
* `cu = data[data.company.str.contains('CU')]` - 데이터에서 회사 컬럼에 `CU`라는 명칭이 들어간 경우 (예: CU 한남리첸시아점) 이를 따로 서브셋으로 따로 추려 cu라는 변수에 할당합니다. 위에서의 필터링과 약간 형태가 다른데 결국 같은 논리입니다.
* `cu` - 잘 뽑혔는지 확인해봅시다.
![특정 문자열 포함 기준으로 서브셋 출력](/assets/materials/20150811/5.png)

다음 <a href="http://jsideas.net/python/2015/08/11/histogram_tutorial_part2.html">part2</a>에서는 위의 명령어들을 적절히 활용하여 스크립트를 짜보고 이를 ipython에서 불러와 실행시켜보겠습니다. 또 matplotlib 라이브러리를 활용하여 몇가지 히스토그램과 박스플롯을 짜보겠습니다.

[시급으로 본 서울지역 아르바이트 환경]: http://jsideas.net/python/2015/08/08/albamon_pay.html