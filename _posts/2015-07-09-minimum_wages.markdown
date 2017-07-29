---
layout:     post
title:      "Minimum Wage and Big Mac"
date:       2015-07-09 20:30:00
author:     "Jun"
categories: "python"
header-img: "img/post-bg-05.jpg"
---

<h2 class="section-heading">Minimum Wage and Big Mac</h2>

<strong>모바일 환경에서는 차트가 보이지 않습니다.</strong>
<br>

<h3>최저임금과 빅맥</h3>
<p>근로자 최저임금이 2015년에는 시급 5,580원으로 작년에 비해 370원 인상되었습니다. 그리고 내년에는 8.1% 인상되어 6,030원이 된다고 합니다. 한시간 일하면 빅맥 세트는 무난하게 먹을 수 있는 수준까지는 올라온 셈입니다. 말 나온김에 국가별 최저임금(시급)과 빅맥지수를 Scatter Plot에 뿌려봤습니다.</p>

<p>자료 출처</p>
<ul>
	<li>국가별 최저임금: <a href="https://en.wikipedia.org/wiki/List_of_minimum_wages_by_country">위키피디아</a></li>
	<li>국가별 빅맥지수: <a href="https://en.wikipedia.org/wiki/List_of_minimum_wages_by_country">Statista</a></li>
</ul>

<p>최저 시급과 빅맥지수 데이터가 주어진 국가만을 추려 시각화했습니다. 우리나라는 어디쯤 있을까요? 마우스를 올려 찾아보세요. (붉은선은 빅맥지수와 최저시급이 1:1로 만나는 지점입니다. 즉, 붉은 선 왼편의 국가에서는 1시간 일해서 빅맥을 사지 못합니다. 선 왼편에는 주로 동남아시아와 남미 국가들이, 오른편에는 서유럽, 북미 그리고 일부 동북아 국가들이 위치해있네요.</p>

<iframe width="100%" height="550" src="//jsfiddle.net/junkwhinger/oswf63gp/9/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>