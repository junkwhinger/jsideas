---
layout:     post
title:      "메르스 - 기록들(업데이트 중)"
date:       2015-06-30 21:30:00
author:     "Jun"
categories: "r"
header-img: "img/post-bg-05.jpg"
---

<h2 class="section-heading">MERS Spread and Media Coverage</h2>

<p>메르스가 확산되기 시작한 2015년 5월 20일부터 완연한 진정세에 들어선 지금까지 메르스에 대한 언론의 대응은 어땠을까요? 네이버 뉴스의 '메르스' 관련 기사 중 '신문게재기사'의 기사 건수와 메르스 신규 확진자 수를 비교해보았습니다.</p>

<p>Key Findings</p>
<ul>
	<li>메르스 신규 확진자 수와 기사 생성 건수는 어느정도 비슷한 양상을 보임</li>
	<li>R의 cov함수('pearson' 방식)로 상관계수를 계산하면 0.303으로 높지 않은 상관계수가 도출됨</li>
	<li>신규 확진자가 급증한 6월 6일~7일에 기사량은 오히려 감소한 것으로 드러나는 것으로 보아, 기사량의 변화를 설명하는데 다른 강력한 변수가 있을 것으로 추정됨</li>
	<li>언론에서는 사태 최초기에는 확산세를, 초중기에는 정부대응 및 정보공개 비판을, 중기 이후로는 시스템 개편과 경제적 손실 추산 등을 주요하게 다룸</li>	
</ul>

<iframe width="100%" height="600" src="//jsfiddle.net/junkwhinger/1g50dda6/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<p>해당 데이터 포인트에 마우스를 올리면, 그날의 신규 확진자와 주요 기사가 툴팁으로 제공됩니다.<br/>
(주요 기사의 경우, 객관적 사실 제공을 위해 네이버 검색시 최상단 3개 기사를 선정하였습니다)</p>	

<h2 class="section-heading"># of News Article per Media</h2>

<p>Key Findings</p>
<ul>
	<li>언론매체들은 상당부분 유사한 신문게제기사량 패턴을 보임</li>
	<li>소위 보수매체(조선일보, 중앙일보, 동아일보)과 진보매체(한겨레, 경향)는 1일 간격으로 차이를 보이고 있음</li>
	<li>기사의 내용에서는 큰 유의미한 차이를 보이지는 않으나, 보수매체의 보도 경향이 조금 더 신중한 것으로 추측함</li>
</ul>

<p>메르스 관련 네이버 신문기사게제량을 각 신문사별로 쪼개보았습니다. 우측의 신문사 네모박스를 클릭하여 라인을 활성화시킬 수 있습니다.</p>
<iframe width="100%" height="600" src="//jsfiddle.net/junkwhinger/pe1toart/105/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>