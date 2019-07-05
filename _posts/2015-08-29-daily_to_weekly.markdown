---
layout: post
title:  "[python-pandas] 일별 데이터를 주별로 묶기"
date:   2015-08-29 23:34:25
img: 20150830.jpg
tags: [python, data analytics]

---

<h2>[python-pandas] 일별 데이터를 주별로 묶기</h2>
<p>앞서 python의 pandas를 활용하여 간단한 데이터 분석과 시각화를 시도해보았습니다. 이번에는 일별로 기록되어있는 데이터를 주별, 월별로 변환해보는 방법에 대해 살펴보겠습니다.

오늘 활용할 예제는 네이버 뉴스에서 (요즘 핫한 키워드인) `헬조선`으로 검색한 뉴스 기사량 데이터입니다. 아래 링크에서 분석에 사용할 데이터를 다운받아 분석을 수행할 폴더 안에 넣어주세요. 
[분석 대상 자료 다운받기](/assets/materials/20150830/hell_chosen.csv)

데이터를 살펴보면 날짜, 매체, 제목으로 구성되어 있는데요, 이를 pandas를 통해 가공하여 주별, 월별 기사량을 뽑아보록 하겠습니다.

먼저 개발환경을 세팅합시다. 원하는 곳에 폴더를 만들고 virtualenv를 통해 pandas, numpy를 설치해주세요. 어떻게 하는지 기억이 안나면 <a href="http://jsideas.net/python/2015/08/10/histogram_tutorial.html">파이썬 초보 - pandas와 matplotlib을 활용한 간단 분석 part 1</a>의 개발환경 구축 파트에서 다시 보실 수 있습니다.
</p>


<h3>1. 필요한 모듈 import 하기</h3>
파이썬 파일을 만들었다면 이제 코드를 써봅시다. 분석에 필요한 pandas와 numpy를 import 합니다.

* `import pandas as pd`
* `import numpy as np`

<h3>2. csv 파일 읽어와서 DataFrame으로 만들기</h3>

* `df = pd.read_csv(“hell_chosen.csv”)` - hell_chosen.csv를 읽어들여 DataFrame으로 저장합니다.
* `df.columns = [‘date’, ‘press’, ‘title’]` - 저장한 df의 컬럼의 이름을 다음과 같이 바꿔줍니다.

<h3>3. int으로 된 날짜를 datetime object로 바꿔주기</h3>
원본 데이터에서 날짜는 20150101과 같이 되어있으며 파이썬은 이를 int(정수)로 인식하고 있습니다. 분석에 용이하도록 이를 datetime object로 바꿔줍니다. 이번에도 지난번과 같이 임시 함수인 lambda를 써서 바꿔줍니다.

* `df['datetime'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d’))` - `.apply(lambda x: ~`는 내가 x를 다룰건데, 어떻게 할거냐면~ 이라는 뜻입니다. 여기서는 `%Y%m%d` 형식으로 된 x를 pandas의 to_datetime 함수를 통해 datetime object로 변환하는 겁니다. 만약 원본값이 2015-01-01이라면 format을 %Y-%m-%d로 바꿔주면 됩니다.

<h3>4. 임의의 값 1을 넣어 DataFrame 완성하기</h3>
원본 데이터의 각 레코드(행)은 하나의 기사를 의미합니다. 8월에는 같은 날에도 여러개의 기사가 나온 걸 볼 수 있습니다. 기사량을 뒤에서 계산하기 쉽게 각 레코드에 기사량 열을 추가하고 기사 1개를 의미하는 `1`을 넣어줍시다.

* `df['all_news_num'] = 1`

<h3>5. datetime 컬럼을 index로 만들기</h3>
`.index`는 대상 DataFrame의 인덱스를 표시하는 함수입니다. `print df.index`를 실행해보면 

![DataFrame의 인덱스](/assets/materials/20150830/1.png)

다음과 같이 Int64Index가 나오며 0부터 정수가 표시됩니다. 1부터 숫자를 세는 R과는 달리 파이썬은 0부터 숫자를 셉니다. 즉 지금 df라는 DataFrame은 0번째 행, 1번째 행.. 이런 식으로 인덱스 넘버를 가지고 있습니다. daily 데이터를 weekly나 monthly로 변환하기 위해서는 index를 datetime index로 바꿔줘야 됩니다. 

* `df.set_index(df['datetime'], inplace=True)` - 이미 우리는 앞서 df에 `datetime`이라는 컬럼을 만들고 date를 datetime object로 변환해서 넣어두었습니다. `set_index`함수를 써서 datetime 컬럼을 인덱스로 지정합니다.

* `df = df.drop('datetime', 1)` - 그리고 이제는 쓸모없어진 datetime 컬럼을 지워줍니다. `drop`함수를 쓰면 간단하게 지울 수 있으며 저기에 붙는 1은 `axis=1`의 줄임이며 컬럼을 합니다. axis=0은 행을 의미하며, `temp_df = temp_df.drop(‘2’, axis=0)은 temp_df에서 인덱스가 2로 된 행을 지웁니다.

df를 간단히 확인해봅시다.

* `df.head()`

![DataFrame의 첫 5개 행 확인](/assets/materials/20150830/2.png)

* `df.info()`

![DataFrame의 정보 확인](/assets/materials/20150830/3.png)

<h3>6. 주별, 월별 데이터로 기준 재정렬하기</h3>
이 함수를 하나 소개하려고 여기까지 글을 썼습니다. pandas의 resample 함수입니다. 이 함수는 앞서 소개한 것처럼 일별 데이터를 주별, 월별 데이터로 변환하는데 쓰이는 magical한 함수입니다. 주, 월 뿐만 아니라 5분 단위, 250 밀리세컨드 단위 등으로도 얼마든지 변환이 가능합니다. 오늘 여기서 다루는 데이터는 굉장히 간단한 데이터이므로, 주 그리고 월로 바꿔보겠습니다.

* `weekly_df = df.resample('W-Mon', how={'all_news_num':np.sum}).fillna(0)` - resample함수에서 2가지를 지정해줬는데 앞의 `W-Mon`은 `변경하는 기준`으로 주별로 하며 날짜는 해당 주의 `월요일`이라는 뜻입니다. 두번째 `how`는 기준 변경에 대한 행위로, 여기서는 월요일에 시작하는 주별로 데이터를 재정렬하는데, df의 `all_news_num`을 numpy의 sum(합계)함수로 처리한다는 의미입니다. 즉, 지금 일별로 되어있는 1을 주별로 묶어서 합한다는 뜻이지요. 그리고 마지막에 `fillna(0)`는 값이 없는 na 레코드가 있다면 0을 넣겠다는 의미입니다. 데이터를 잘 보시면 4월~6월까지 날짜가 듬성듬성하신 걸 볼 수 있습니다. 이 명령어를 붙이지 않아도 코드는 돌아가겠지만 중간에 기사량이 없는 주는 출력되지 않습니다. 

![resample 함수를 사용한 주별 재정렬](/assets/materials/20150830/4.png)

짜잔! 성공적으로 돌아갔습니다. 2015년 6월 8일을 캘린더에서 찾아보세요! 월요일입니다. 
다음은 월별로 돌려봅시다. 방법은 간단합니다. 위의 명령에서 `W-Mon`을 `M`으로 바꿔주기만 하면됩니다.

* `monthly_df = df.resample('M', how={'all_news_num':np.sum}).fillna(0)`

![resample 함수를 사용한 월별 재정렬](/assets/materials/20150830/5.png)

간단하지요?^^

<h3>더해보기</h3>
요새 `헬조선`이라는 키워드가 뉴스에 등장하기 시작했는데, 사실 이 단어는 인터넷 커뮤니티나 트위터 상에 등장한지 꽤 연식이 된 용어입니다. 네이버 트렌드에서 해당 키워드를 검색하면 1년 전부터 언급이 되오던 걸 볼 수 있지요. 그래서 해당 트렌드 자료와 앞서 주별로 뽑아봤던 데이터를 pandas의 merge를 통해 합치면, 아래와 같은 차트를 만들 수 있답니다!

![resample 함수를 사용한 월별 재정렬](/assets/materials/20150830/6.jpg)

2014년 초에도 검색기록이 남아있지만 시각화를 위해 오늘부터 1년 전인 2014년 9월 경부터 검색 트렌드를 살펴본 결과, 인터넷에서 흥하기 시작한 `헬조선` 키워드는 약 8개월 후에 인터넷 뉴스를 통해 최초로 기사화되었으며, 11개월 정도가 지나야 신문게제기사, 즉 메이저 신문에서 다루어졌네요. 인터넷 신조어가 뉴스에서 다뤄진 다른 케이스들과 비교해봐야겠지만, 지금 당장 인터넷의 한 귀퉁이에서 흥하는 키워드를 1년 후에는 조간신문 지면에서 볼 수 있을지도 모르겠습니다.

읽어주셔서 감사합니다!
