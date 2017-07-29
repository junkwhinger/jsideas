---
layout:     post
title:      "세월호 대통령 담화문 비교분석 및 시각화"
date:       2015-06-09 23:50:00
author:     "Jun"
categories: "r"
header-img: "img/post-bg-05.jpg"
---

<h2 class="section-heading">세월호 대통령 담화문 비교분석 및 시각화</h2>

<p>머신러닝 중 텍스트 마이닝에 쓰이는 R코드로 세월호 대통령 담화문을 비교 분석해보았습니다. 아래 네트워크에서 연결된 키워드는 함께 등장할 확률이 높은 키워드들입니다. 예를 들어 '대한','민국'처럼 텍스트 전처리가 완벽하지 않아 당연히 함께 걸리는 키워드도 보이지만, 사건발생 34일째는 {'민관', '유착'}, {'국가안전','안전','기능'}, {'재난','대응'} 처럼 또다른 세월호 사건을 막기 위한 갖가지 대응책이 제시됩니다. 이에 반해 세월호 1주기 대통령 담화문에서는 {'세월호', '희생자'}, {'국민', '추모'} 등의 단어 언급이 주를 이룹니다. 키워드의 크기는 해당 키워드의 사이중심성(betweenness centrality)이 높을수록 큽니다.</p>

<iframe width="100%" height="900" src="//jsfiddle.net/junkwhinger/3j6k5out/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<iframe width="100%" height="900" src="//jsfiddle.net/junkwhinger/x7j7ukhz/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>