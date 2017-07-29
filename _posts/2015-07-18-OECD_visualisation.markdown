---
layout: post
title:  "OECD Statistics Visualisation"
date:   2015-07-18 19:34:25
categories: d3.js
image: /assets/oecd_header.jpg
---

<h2>OECD Statistics Visualisation</h2>

<p>이미지 출처: <a href="http://compass.ptvgroup.com/2014/04/summit-of-transport-ministers/?lang=en">PTV Group</a></p>

<hr />

<h3>Services Trade Restrictiveness Index(STRI)</h3>

OECD 국가의 서비스 교역 장벽을 국가별로, 분야별로 시각화해보았습니다. [OECD][OECD] 자료에 따르면, 서비스 교역은 지식과 아이디어의 교류를 촉진하고 기업의 생산성을 높이며 낮은 가격과 폭넓은 선택의 다양성을 소비자에게 안겨준다고 합니다. 그러나 국가간 교역 및 투자 장벽 및 규제가 서비스 교역을 어렵게 하고 있습니다. 이러한 상황에서 OECD는 이 지표(Services Trade Restrictivess Index)를 통해 정책입안자들과 협상가들에게 정량적으로 현상을 판단할 수 있는 인사이트를 제공하고 있습니다. STRI는 Accounting부터 Construction까지 18가지 세부 서비스 분야로 나뉘어져있습니다. 0은 해당 서비스 산업의 `완전개방`을, 1은 `완전비개방'을 의미합니다. 

<h5>[STRI: visualisation]</h5>

<iframe src="http://bl.ocks.org/junkwhinger/raw/53896a1c6cc37c0dbd3e/47ec390a84d71ea6175645bb7984931ea82c2271/" width="100%" height="600px" marginwidth="0" marginheight="0" scrolling="no" frameBorder="0"></iframe>



<h5>사용방법</h5>
우측의 범례를 눌러 해당 나라의 그래프를 활성화 혹은 비활성화시킬 수 있습니다. 맨 위의 'All' 버튼을 누르면 전체 선택이 가능합니다.

<h5>Key Insight</h5>
1. OECD국가의 전반적인 서비스 시장 개방지수의 전체평균은 0.2로 상당부분 개방되어 있습니다.
2. 분야별 평균은 `Air transport`, `Accounting`, `Legal` 순으로 가장 높았습니다. 분야별 평균 최하위 3개 분야는 `Distribution`, `Sound recording`, `Motion pictures`였습니다. 비행기 사고, 회계 부정, 법적 분쟁 등 리스크가 큰 서비스 분야는 개방도가 낮고, 상대적으로 리스크가 적거나, 철도 규격 통일이나 영화 기술 등 국제적 협업이 필요한 분야에서는 장벽이 낮은 것으로 보입니다.
3. 특이국가: 전체 OECD국가의 철도 화물 운송 서비스는 0.17로 낮은 반면, 이스라엘은 1로 완전비개방 상태입니다. 주변 아랍국가와의 비우호적 관계나, 철도가 놓이기에는 비교적 작은 국토 면적이 원인인 듯 싶습니다.
4. 대한민국과 유사한 국가: 이 부분은 추후에 계층적 군집화를 통해 확인해보겠습니다.

<h5>[STRI: hierarchical clustering]</h5>

머신러닝의 군집화 기법 중 상향식 계층적 군집화(hierarchical clustering)를 사용하여 앞서 소개된 18가지 서비스 산업군에서의 개방정도를 바탕으로 비슷한 국가끼리 모았습니다. 같은 색상으로 칠해진 국가는 같은 범주에 속하며, 유사 색상간에 관계는 없습니다.

<iframe src="http://bl.ocks.org/junkwhinger/raw/392dd4830b1f762712c0/" width="100%" height="650px" marginwidth="0" marginheight="0" scrolling="no" frameBorder="0"></iframe>

<h5>사용방법</h5>
지도상에서 해당 국가위에 마우스 커서를 올리면 지도 아래 범주에 유사 국가의 목록이 표시됩니다.

<h5>Key Insight</h5>
1. Better Life Index와는 달리 STRI에서는 대륙별로 국가들이 군집화되지 않고, 비교적 다양하게 섞였습니다.
2. 우리나라의 경우 독일, 터키, 일본, 룩셈부르크, 네덜란드, 스위스, 벨기에, 그리스와 같은 군집에 속해있는 것으로 나타납니다.
3. 특이국가: `Iceland`는 군집 개수를 줄이면 캐나다, 핀란드, 스웨덴과 묶일 수 있으나, 군집 6에서는 함께 묶이는 국가가 없는 것으로 나타났습니다.

<hr />

<strong>계속 업데이트됩니다.</strong>

[OECD]:		   http://www.oecd.org/tad/services-trade/services-trade-restrictiveness-index.htm
