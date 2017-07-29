---
layout: post
title:  "김현수의 레그킥: 데이터 비교 분석"
date:   2016-04-19 10:34:25
categories: python
image: /assets/2016-04-19-kim_hyun_soo_record_files/header.jpg
--- 
# 레그킥은 정말 장타력을 희생하고 정확도를 높였나
(주의: 본 분석은 방법 및 해석에 오류가 있을 수 있습니다. 오류에 대한 피드백은 언제든 환영입니다! junsik.whang@gmail.com으로 연락바랍니다!) 
 
메이저리그에 진출한 한국 타자들의 장타 소식에 연일 스포츠 지면이 활기를 띈다. 아직 킹캉과 추추가 그라운드에 돌아오지 못했지만, 그 빈 자리를
박병호와 이대호가 성공적으로 메우고 있다. 그들의 빛이 강렬해질수록 동시에 진출한 볼티모어 오리올스의 김현수의 그림자는 깊어간다. 잘나가는
메이저리그 데뷔 동기들과 달리 김현수는 아직 시범경기에서부터 시작된 부진을 떨치지 못하고 있다. 
 
오늘 스포츠 기사를 보다보니 발견한 흥미로운 기사 하나. 문화일보 <a href="http://www.munhwa.com/news/view.html?no=2016041901072739176002" target="_blank">"레그킥 또 논란... '마이 스타일'로 정면 돌파하라"</a>. 레그킥이란 타자가 타격을
할 때 앞다리를 들었다가 내려놓는 자세를 의미한다. 한국인 타자들이 부진할때마다 메이저리그 쪽에서 단골로 지적하는 부분인듯. <a href="http://sports.khan.co.kr/news/sk_in
dex.html?cat=view&art_id=201604151411003&sec_id=510301" target="_blank">박병호가 시즌
초반에 부진했을때 미국 블리처리포트에서 토텝과 레그킥를 문제로 지적했다 한다.</a> KBO보다 구속이 빠른 메이저리그에서 타격
타이밍을 놓치는 원인으로 레그킥을 지목했다. 

![강정호의 레그킥!! 출처: spotvnews](/assets/2016-04-19-kim_hyun_soo_record_files/kang_leg.jpg) 
 
재밌는 점은 박병호와 이대호는 레그킥을 여전히 쓰고 있으며 최근 타격감이 좋다. 특히 박병호는 벌써 시즌 3호 홈런까지 기록했다. 반면 2015
시즌부터 메이저리그 조언을 받아들여 레그킥을 자제해온 김현수는 벤치워머 신세다. 기사에서는 "김현수는... 바뀐 타격 자세에 적응 기간을
거쳤기에 심리적 안정을 찾고 많은 기회를 얻으면 정교한 타격을 되찾겠지만 장타력 감소는 불가피할 전망이다"라고 내다봤다. 정말 그럴까? 뭔가
의심이 들어 데이터를 조금 찾아봤다. 
 
### 가설: 레그킥을 사용하지 않은 2015 김현수는 사용한 2014 김현수보다 장타력은 낮고 정교한 타격을 했을 것이다. 
 
2015년과 2014년의 김현수 타격 데이터를 비교해서 기록상으로 어떤 차이가 있었는지 보자. 데이터는 <a href="http:
//www.koreabaseball.com/Record/Player/HitterDetail/Basic.aspx?playerId=76290" target="_blank">KBO</a> 사이트에서 구해왔다.
메이저리그는 csv로 쉽게 받을 수 있던데 KBO는 못 찾아서 손으로 옮겨적음 ㅠ) 

**In [1]:**

{% highlight python %}
import pandas as pd
{% endhighlight %}

**In [2]:**

{% highlight python %}
raw_df = pd.read_csv("kim_record.csv")
{% endhighlight %}

**In [3]:**

{% highlight python %}
raw_df
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>타율</th>
      <th>경기</th>
      <th>타수</th>
      <th>득점</th>
      <th>안타</th>
      <th>2루타</th>
      <th>3루타</th>
      <th>홈런</th>
      <th>루타</th>
      <th>타점</th>
      <th>도루</th>
      <th>도루실패</th>
      <th>볼넷</th>
      <th>사구</th>
      <th>SO</th>
      <th>병살타</th>
      <th>장타율</th>
      <th>출루율</th>
      <th>실책</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>0.322</td>
      <td>125</td>
      <td>463</td>
      <td>75</td>
      <td>149</td>
      <td>26</td>
      <td>0</td>
      <td>17</td>
      <td>226</td>
      <td>90</td>
      <td>2</td>
      <td>0</td>
      <td>53</td>
      <td>7</td>
      <td>45</td>
      <td>10</td>
      <td>0.488</td>
      <td>0.396</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>0.326</td>
      <td>141</td>
      <td>512</td>
      <td>103</td>
      <td>167</td>
      <td>26</td>
      <td>0</td>
      <td>28</td>
      <td>277</td>
      <td>121</td>
      <td>11</td>
      <td>5</td>
      <td>101</td>
      <td>8</td>
      <td>63</td>
      <td>13</td>
      <td>0.541</td>
      <td>0.438</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


 
김현수의 통산 기록은 2006년도부터 시작하지만, 가급적 최근의 데이터로 비교하기위해 14년도와 15년도 데이터만 들고 왔다. 

**In [4]:**

{% highlight python %}
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%pylab inline

sns.set()
{% endhighlight %}

    Populating the interactive namespace from numpy and matplotlib


**In [5]:**

{% highlight python %}
kim_df = raw_df
{% endhighlight %}

**In [6]:**

{% highlight python %}
kim_df
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>타율</th>
      <th>경기</th>
      <th>타수</th>
      <th>득점</th>
      <th>안타</th>
      <th>2루타</th>
      <th>3루타</th>
      <th>홈런</th>
      <th>루타</th>
      <th>타점</th>
      <th>도루</th>
      <th>도루실패</th>
      <th>볼넷</th>
      <th>사구</th>
      <th>SO</th>
      <th>병살타</th>
      <th>장타율</th>
      <th>출루율</th>
      <th>실책</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>0.322</td>
      <td>125</td>
      <td>463</td>
      <td>75</td>
      <td>149</td>
      <td>26</td>
      <td>0</td>
      <td>17</td>
      <td>226</td>
      <td>90</td>
      <td>2</td>
      <td>0</td>
      <td>53</td>
      <td>7</td>
      <td>45</td>
      <td>10</td>
      <td>0.488</td>
      <td>0.396</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>0.326</td>
      <td>141</td>
      <td>512</td>
      <td>103</td>
      <td>167</td>
      <td>26</td>
      <td>0</td>
      <td>28</td>
      <td>277</td>
      <td>121</td>
      <td>11</td>
      <td>5</td>
      <td>101</td>
      <td>8</td>
      <td>63</td>
      <td>13</td>
      <td>0.541</td>
      <td>0.438</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**In [7]:**

{% highlight python %}
kim_df.index = kim_df.연도
{% endhighlight %}

**In [8]:**

{% highlight python %}
kim_df.drop('연도', axis=1, inplace=True)
{% endhighlight %}

**In [9]:**

{% highlight python %}
matplotlib.rc('font',family='AppleGothic')
ax = kim_df.T.plot(kind='bar', cmap='coolwarm')
plt.show()
{% endhighlight %}

 
![김현수 데이터 비교](/assets/2016-04-19-kim_hyun_soo_record_files/2016-04-19-kim_hyun_soo_record_15_0.png) 

 
대충 트렌드는 보이지만, 각 기록의 기준점이 달라 잘 안보인다. 비교를 위해 각 항목별로 높은 값을 1로 표준화하여 비교해보자 

**In [10]:**

{% highlight python %}
re_kim_df = kim_df.T
{% endhighlight %}

**In [11]:**

{% highlight python %}
def return_max(a, b):
    return max(a, b)
{% endhighlight %}

**In [12]:**

{% highlight python %}
re_kim_df['max'] = re_kim_df.max(axis=1)
{% endhighlight %}

**In [13]:**

{% highlight python %}
re_kim_df['2014_norm'] = re_kim_df[2014] / re_kim_df['max']
re_kim_df['2015_norm'] = re_kim_df[2015] / re_kim_df['max']
{% endhighlight %}

**In [14]:**

{% highlight python %}
ax = re_kim_df[['2014_norm','2015_norm']].plot(kind='bar', cmap='coolwarm')
ax.legend_.remove()
plt.show()
{% endhighlight %}

 
![김현수 데이터 비교2](/assets/2016-04-19-kim_hyun_soo_record_files/2016-04-19-kim_hyun_soo_record_21_0.png) 

 
해놓고 나니 별로 이쁘지는 않다. 겹쳐서 레전드를 지웠는데, 여튼 파란색이 14시즌, 15년이 붉은색이다. 장타율과 출류율을 보면 모두 15년도
성적이 높다. 타율도 15년 성적이 14년보다 근소하게 높은 수준이다. 원래 가설대로라면 레그킥을 자제한 15년도 성적은 출류율과 타율이 높은
대신 장타율이 낮아야하는데, 오히려 장타율 역시 높은 수준을 기록했다. 
 
### 혹시 15년에는 단타를 더 많이 치지 않았을까? 14년과 안타 유형의 분포를 비교해보자! 

**In [15]:**

{% highlight python %}
kim_hit_df = kim_df[['안타','2루타','3루타','홈런']]
kim_hit_df
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>안타</th>
      <th>2루타</th>
      <th>3루타</th>
      <th>홈런</th>
    </tr>
    <tr>
      <th>연도</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014</th>
      <td>149</td>
      <td>26</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>167</td>
      <td>26</td>
      <td>0</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>


 
1루타는 별도로 데이터에 없다. 2루타 이상을 제외한 안타가 1루타이므로 1루타 컬럼을 새로 하나 만들어주자. 

**In [16]:**

{% highlight python %}
def hit_one(df):
    hit_one_series = df.안타 - df['2루타'] - df['3루타'] - df.홈런
    return hit_one_series
{% endhighlight %}

**In [17]:**

{% highlight python %}
kim_hit_df['1루타'] = hit_one(kim_hit_df)
{% endhighlight %}


**In [18]:**

{% highlight python %}
kim_hit_df
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>안타</th>
      <th>2루타</th>
      <th>3루타</th>
      <th>홈런</th>
      <th>1루타</th>
    </tr>
    <tr>
      <th>연도</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014</th>
      <td>149</td>
      <td>26</td>
      <td>0</td>
      <td>17</td>
      <td>106</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>167</td>
      <td>26</td>
      <td>0</td>
      <td>28</td>
      <td>113</td>
    </tr>
  </tbody>
</table>
</div>


 
자 이제 각 루타의 연도별 비율을 뽑아보자. 

**In [19]:**

{% highlight python %}
def normalise(a_series):
    return a_series / kim_hit_df.안타
{% endhighlight %}

**In [20]:**

{% highlight python %}
kim_hit_df_norm = kim_hit_df.apply(normalise)
kim_hit_df_norm
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>안타</th>
      <th>2루타</th>
      <th>3루타</th>
      <th>홈런</th>
      <th>1루타</th>
    </tr>
    <tr>
      <th>연도</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014</th>
      <td>1</td>
      <td>0.174497</td>
      <td>0</td>
      <td>0.114094</td>
      <td>0.711409</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>1</td>
      <td>0.155689</td>
      <td>0</td>
      <td>0.167665</td>
      <td>0.676647</td>
    </tr>
  </tbody>
</table>
</div>



**In [21]:**

{% highlight python %}
# 안타는 당연히 1이 되므로 날려주고, 3루타도 둘다 0이므로 날린다.
kim_hit_df_norm.drop(['안타', '3루타'], axis=1, inplace=True)
{% endhighlight %}

**In [22]:**

{% highlight python %}
kim_hit_df_norm
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2루타</th>
      <th>홈런</th>
      <th>1루타</th>
    </tr>
    <tr>
      <th>연도</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014</th>
      <td>0.174497</td>
      <td>0.114094</td>
      <td>0.711409</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>0.155689</td>
      <td>0.167665</td>
      <td>0.676647</td>
    </tr>
  </tbody>
</table>
</div>


 
2개의 비율이 주어졌을 때 이들간에 통계적으로 유의미한 차이가 있는지 검증해보자. 사용된 variable이 categorical(범주형)이고,
앞서 각 안타유형의 발생 빈도가 5 이상이었으므로 chi-square goodness of fit test를 사용하기로 한다. 물론 샘플링
메소드로 simple random sampling을 사용하지는 않았지만, 카이스퀘어 분석을 해보고 싶었으므로 이는 넘어가도록 한다. 

 
## 가설 (state the hypothesis)

* H0: 2014년도와 2015년도의 안타 유형 비율은 consistent할 것이다.
* Ha: 2014년도와 2015년도의 안타 유형 비율은 consistent하지 않을 것이다. 

 
## 설계 (formulate an analysis plan)

* Significant Level - 0.05
* test method - chi-square goodness of fit test 


##분석 (analyse sample data) 

**In [23]:**

{% highlight python %}
import scipy, scipy.stats
{% endhighlight %}

**In [24]:**

{% highlight python %}
def chisquare(observed_values,expected_values):
    test_statistic=0
    for observed, expected in zip(observed_values, expected_values):
        test_statistic+=(float(observed)-float(expected))**2/float(expected)
    return test_statistic
{% endhighlight %}

**In [25]:**

{% highlight python %}
obs = scipy.array(kim_hit_df_norm.iloc[0])
exp = scipy.array(kim_hit_df_norm.iloc[1])
{% endhighlight %}

**In [26]:**

{% highlight python %}
scipy.stats.chisquare(obs, exp)
{% endhighlight %}




    Power_divergenceResult(statistic=0.021174474151259448, pvalue=0.98946861045470613)


 
* degree of freedom = 2
* test statistic = 0.02117
* pvalue = 0.9894

pvalue가 일단 0.98가 나왔다. 신뢰수준 95%가 아니라 1%로 해야 검정을 통과할 수준. 
 
##해석 (interpret results)

pvalue가 0.98이므로 신뢰수준 95%를 만족시키지 못하므로 null hypothesis를 기각하지 못함. 즉 레그킥을 한 2014년과
레그킥을 자제한 2015년의 김현수 간에 유의미한 안타 비율 차이는 존재하지 않음. (레그킥 하나 안하나 단타 치고 장타 쳤음) 
 
결론적으로 김현수 데이터로는 레그킥 자제로 인한 효과가 기사에서 곁들인 추측을 뒷받침하지는 못했다. 물론 이를 KBO 데이터로 더 검증하기
위해서는 김현수처럼 레그킥 유무를 독립변수로 둔 샘플이 더 많아야 한다. 그렇다하더라도 기사의 추측에서 김현수 데이터를 활용하여 보다 풍성한
정보 전달을 하지 못한 점은 아쉽다. 

김현수 화이팅!
![김현수 출처: mlbpark](/assets/2016-04-19-kim_hyun_soo_record_files/kim_smile.jpg) 

