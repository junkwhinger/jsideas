---
layout:     post
title:      "연관규칙 분석사례 - 대하드라마 정도전"
date:       2015-05-02 23:50:00
author:     "Jun"
img: 20150503.jpg
tags: [r, data analytics]

---

<h2 class="section-heading">연관규칙 분석사례 - 대하드라마 정도전</h2>
<p>association analysis case - Jeong Do Jeon, a Korean history drama</p>
<p>그림 출처: http://starplanet.tistory.com/1872 </p>

<p>요즘 패스트캠퍼스에서 진행하는 머신러닝 강의를 듣고 있는데, 지난주 수요일에 들었던 강의는 R의 ‘arules’ 라이브러리를 활용한 ‘연관규칙 분석’입니다. 여러가지 변수간의 공통출현빈도를 계산해서 상관관계의 rule을 도출해내는 분석 방법인데, 수업을 듣던 도중 작년 여름에 했던 <a href="http://www.slideshare.net/jdjmania/jdj-network-analysisvf">대하드라마 정도전 네트워크 분석</a>이 생각났습니다.</p>

<p>당시 드라마를 보면서 같은 장면에 등장한 인물들의 목록을 데이터화했고 python의 ‘networkx’라이브러리와 gephi를 통해 분석을 했습니다. 연관규칙분석을 몰랐던 당시에는 3인 이상의 대화 장면을 각 인물간의 1:1 교류(all possible unique pair)로 변환한 후에, 최다 대화빈도, 최다 교류빈도 등과 같은 차트를 만들었었습니다.</p>

<p>R을 사용하여 연관규칙분석을 돌리면 첫번째는 교류로 변환하는 과정이 필요없고, 두번째는 support, confidence, lift를 활용한 ‘규칙’ 도출이 가능합니다. 그래서 이번에는 같은 데이터를 R을 통해 연관규칙 분석해보았습니다.</p>

<h2 class="section-heading">0. 분석 방법</h2>
<p>각 회별로 정리된 인물데이터 csv파일을 불러온 후 모두 합쳐 50회 분량의 데이터를 하나의 리스트에 저장했습니다. 저장한 리스트를 R의 transactions 데이터 타입으로 변환하고 중간에 실수로 들어갔던 중복(duplicates)을 처리했습니다. 그리고 앞서 소개한 ‘arules’라이브러리와 시각화를 위한 ‘arulesViz’, ‘wordcloud’를 사용하여 워드클라우드와 빈도플롯을 그려보았습니다. 마지막으로는 support와 confidence값을 조정하여 규칙 리스트를 뽑아낸 후, lift가 높은 순으로 규칙 5개를 뽑았습니다.</p>

<h2 class="section-heading">1. 워드클라우드</h2>
![pic1](/assets/materials/20150503/pic1.png)
<span class="caption text-muted">fig 1. wordcloud</span>

<p>워드클라우드는 등장 빈도와 같은 값에 비례하여 해당 단어나 변수의 크기와 색을 글자뭉치형태로 보여주는 그림입니다. 여기서는 각 인물들의 등장 빈도를 기반으로 그림을 구성하였고, 등장빈도가 유사하면 같은 색으로 처리되었습니다. mac에서 RStudio로 작업을 하였는데, 한글을 AppleGothic으로 처리하였더니 그래픽이 예쁘지는 않습니다. (폰트 지정을 아예 안해주면 네모칸으로 표시됩니다ㅠㅠ) 당연하게 정도전이 가장 크게 나오지만, 정도전이 보좌하는 이성계도 만만치 않은 등장빈도를 자랑합니다. 정도전의 숙적으로 나왔던 정몽주, 이방원, 이인임도 보이네요. 극 초반 포스를 보여줬던 공민왕은 저 멀리 구석에서 발견됩니다.
</p>

<h2 class="section-heading">2. 등장빈도 분석</h2>
![pic2](/assets/materials/20150503/pic2.png)
<span class="caption text-muted">fig 2. itemFrequencyPlot</span>

<p>두번째는 인물들의 등장빈도 차트입니다. 여기서는 support를 0.05로 잡고 차트를 뽑았습니다. 결과는 앞선 워드클라우드를 바차트로 표현한 것과 동일합니다. 다만 support로 threshold를 주었기에 보수주인같은 비주요 등장인물들은 나오지 않습니다.</p>

<h2 class="section-heading">3. 연관규칙 도출</h2>
<p>이제 연관규칙 분석의 핵심인 ‘규칙’을 뽑을 차례입니다. 규칙은 3가지 지표를 통해 도출하는데 이들은 앞서 언급했던 support, cofidence, 그리고 lift입니다. support는 조건절이 일어날 확률, confidence는 조건절이 일어났을 때 결과절이 일어날 확률, 그리고 lift는 조건절과 결과절의 발생이 서로 독립이 아닐 확률(이 경우 독립 확률인 1보다 커야함)입니다. </p>
<p>apriori 함수를 사용할 때 support를 최소 0.025 이상, confidence를 최소 0.5 이상으로 잡고 규칙을 뽑았습니다. 총 25개 규칙이 나왔네요. (support와 confidence를 설정하는데는 정해진 답은 없다고 합니다. 데이터의 성격에 맞게 돌리는 것인데, 대략 20~40개 정도가 나오면 적정하다고 합니다.)</p>
![pic3](/assets/materials/20150503/pic3.png)

<p>자 그러면 여기서 lift (조건절과 결과절이 독립이 아닐 확률, 즉 서로 강한 영향을 주고 받을 확률. 1보다 높아야 함)를 기준으로 상위 5개를 뽑아보겠습니다.</p>
![pic4](/assets/materials/20150503/pic4.png)
<span class="caption text-muted">fig 3. rules</span>

<p>규칙은 5개가 나왔지만, 정리하면 크게 3가지로 나뉘어지네요.</br>
1. 이숭인, 권근</br>
2. 배극렴, 이성계, 변안열</br>
3. 염흥방, 임견미</br>
</p>

<p>이 3커플(?)은 극에서 가장 중요한 커플들은 아니었습니다만, 생각해보면 서로 함께 많이 나오던 사람들이기는 합니다. 이숭인과 권근은 신진사대부 세력으로 초반에 정몽주를 도와 이인임과 싸우다가 나중에는 정도전과 싸웁니다. 배극렴, 이성계, 변안열은 드라마 중반 침략한 왜구들을 맞아 함께 싸우는 장면이 많이 등장합니다. 마지막으로 염흥방과 임견미는 이인임의 수족으로 함께 전횡을 일삼는 탐관오리 커플로 그려집니다.</p>
<p>드라마의 대표 커플은 당연히 정도전과 이성계일 것입니다. 그런데 이 규칙에서 그들이 보이지 않는 이유는.. 아마 두 등장인물간 대화도 많았겠지만, 타 인물들과의 관계도 못지 않았기 때문이지 않을까요? 예를 들어 정도전은 가족들, 이성계 일가들, 측근들, 정적들 등 거의 대부분의 인물들과도 함께 포착되었기에 support는 높을 수 있어도 lift는 낮을 수 있습니다. 결과절이 ‘정도전’인 것으로 규칙을 다시 뽑아봤습니다. </p>
![pic5](/assets/materials/20150503/pic5.png)
<span class="caption text-muted">fig 4. rules with 정도전 on rhs</span>
<p>역시 리프트를 보면 앞선 규칙들과 달리 3을 넘지 못합니다. 즉 앞선 규칙들에 비해 힘이 약하다고 볼 수 있겠네요.</p>
<p>결과적으로 앞서 도출된 3쌍에 의미를 부여하자면, 직관적으로나 빈도상으로 파악했을때 가장 중요한 커플들은 아니었으나, 서로 가장 끈끈하게 관계를 유지했던 인물들로 이해할 수 있을 것 같습니다. 특히 이숭인과 권근, 임견미와 염흥방은 말이지요.</p>

{% highlight r %}

library(arules)
library(arulesViz)
library(wordcloud)
file_list <- list()

data_list <- list()
for (i in 1:50) {
  filename <- paste("jdj", i, ".csv", sep="")
  file<-read.csv(filename, header=F, stringsAsFactors=F)
  file<-file[, colSums(is.na(file)) != nrow(file)]
  
  
  a_list <- list()
  for (row in 1:nrow(file)) {
    temp <- unlist(file[row,])
    temp <- as.vector(temp)
    temp2 <- vector()
    for (item in 1:length(temp)) {
      if (temp[item] != "") {
        temp2 <- c(temp2, temp[item])
      }
    }
    a_list[[row]] <- temp2
    
  }
  data_list <- c(data_list, a_list)
  
}

names(data_list) <- paste("scene", c(1:length(data_list)), sep="")

trans <- as(data_list, "transactions")

par(family="AppleGothic")
itemName <- itemLabels(trans)
itemCount <-itemFrequency(trans) *1766
col <- brewer.pal(8, "Dark2")

wordcloud(words = itemName, freq=itemCount, min.freq=1, scale=c(7,0.1), col=col, random.order=FALSE)
itemFrequencyPlot(trans, support=0.05, cex.names=0.8)

rules <- apriori(trans, parameter=list(support=0.01, confidence=0.5))
inspect(head(sort(rules,by="lift"),5))
rules <- apriori(trans, parameter=list(support=0.01, confidence=0.5), appearance=list(rhs="정도전", default="lhs"))

plot(rules)
plot(rules, method="grouped")


{% endhighlight %}