---
layout:     post
title:      "냉장고를 부탁해 - 네트워크 분석(업데이트 중)"
date:       2015-06-30 21:30:00
author:     "Jun"
categories: "python"
header-img: "img/post-bg-05.jpg"
---

<h2 class="section-heading">냉장고를 부탁해!</h2>

<p>요즘 재밌게 보고 있는 JTBC의 냉장고를 부탁해-! 7월 2일 기준으로 33회까지 등장한 셰프들과 냉장고의 데이터를 손으로 긁어(ㅠㅠ) 네트워크 시각화를 해보았습니다.</p>

<h2 class="section-heading">셰프 대결 네트워크</h2>

<p>김풍은 과연 샘킴의 천적인가? 홍석천과 정창욱 중 누가 진짜 냉부의 깡패인가..그동안의 대결 데이터를 한판에 뿌려봤습니다.</p>

<h3>네트워크 해설</h3>
<ul>
	<li>노드 크기: 라디오 버튼으로 선택할 수 있는 지표에 따라 달라지며, 지표값이 클수록 노드 크기가 커집니다.</li>
	<li>노드 색상: 같은 색의 노드는 같은 그룹에 속합니다. (*directed graph를 undirected로 변형하여 계산함)</li>
	<li>링크 화살표: 승리자 -> 패배자이며, 화살표가 클수록 많이 승리하거나 많이 패배한 것입니다.</li>

</ul>

<h3>해보세요!</h3>
<ul>
	<li>노드 더블클릭 - 해당 셰프의 직접적인 네트워크를 보여줍니다.</li>
	<li>노드 마우스오버 - 해당 셰프의 전적 요약을 보여줍니다.</li>
	<li>노드 드래그 - 노드를 이리저리 끌어당길 수 있습니다.</li>	
	<li>라디오 버튼 - 노드 크기를 셰프의 '출연 횟수', '승리 횟수', '패배 횟수', '승리 확률'로 변형시킬 수 있습니다.</li>
</ul>

<iframe width="100%" height="700" src="//jsfiddle.net/junkwhinger/3ea5u9vb/25/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<h2 class="section-heading">냉장고 재료 분석</h2>
<p>온갖 식재료가 가득했던 션의 냉장고에서부터 생각보다 빈약했던 호지롷의 냉장고까지! 연예인들의 냉장고에는 무슨 재료가 있을까요? r의 군집화 패키지를 활용하여 워드클라우드를 뽑아봤습니다.</p>

<h3>냉장고 전체 재료 워드클라우드</h3>
![fridge wordcloud](/assets/fp_all_wc.png)

<h3>셰프 선택 재료 워드클라우드</h3>
![fridge wordcloud](/assets/fp_used_wc.png)

<p>글씨가 클 수록 재료의 출연빈도가 높습니다. 달걀과 우유는 왠만하면 선발되는 것 같고..냉동만두는 냉장고에는 들어있지만 셰프들의 선택을 자주 받지는 못한 모양입니다. 냉부 보면서 몰랐던 재룐데 페페론치노가 냉장고에도 자주 등장하고, 셰프들의 선택도 자주 받은 모양이네요.</p>

<h3>주요 재료 상대 빈도 추이</h3>
<p>우측 범례를 눌러 재료의 그래프를 활성화시켜보세요.</p>	
<iframe width="100%" height="550" src="//jsfiddle.net/junkwhinger/1tv9eyvz/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<p>위의 워드클라우드 중 주요 재료를 선별하여 냉장고와 셰프선택 상대 빈도(relative frequency)를 비교해보았습니다. 우측 범례에서 '밀싹'을 기준으로 위가 셰프 선택 빈도 최다 재료, 밀삭을 포함한 아래가 빈도 차가 가장 적은 재료, 즉 나오기만 하면 왠만하면 쓰였던 재료라고 할 수 있겠네요. 김풍셰프가 흥.칩.풍에서 사용했던 라이스페이퍼가 눈에 띕니다. 달걀은 냉장고 빈도가 60%로 10개 냉장고 중에 6개 냉장고에 들어있는데, 전체 요리 10개 중 3개에 사용됩니다.(달걀없는 냉장고는 대체 뭐지..)</p>

<h3>냉장고 재료 연관규칙 분석</h3>
<p>r의 Arules 패키지를 사용해서 재료들의 공통출연빈도(co-occurence)를 통해 연관규칙을 도출해봤습니다. 즉, 어떤 재료 A가 있다하면 꼭 어떤 재료 B가 있더라..라는 식의 규칙을 도출해보는 겁니다. <a href="http://jsideas.net/r/2015/05/03/jdj-association-analysis/">드라마 '정도전'의 등장인물 연관규칙분석</a>에서도 사용했던 방식입니다.</p>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-s6z2{text-align:center}
</style>
<table class="tg">
  <tr>
    <th class="tg-031e">#</th>
    <th class="tg-031e">lhs</th>
    <th class="tg-031e">rhs</th>
    <th class="tg-031e">support</th>
    <th class="tg-031e">confidence</th>
    <th class="tg-031e">lift</th>
  </tr>
  <tr>
    <td class="tg-031e">1</td>
    <td class="tg-031e">아이스크림=&gt;</td>
    <td class="tg-031e">초콜릿</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">1.0</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">2</td>
    <td class="tg-031e">초콜릿=&gt;</td>
    <td class="tg-031e">아이스크림</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">0.5</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">3</td>
    <td class="tg-031e">블랙올리브=&gt;</td>
    <td class="tg-031e">파슬리</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">1.0</td>
    <td class="tg-s6z2">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">4</td>
    <td class="tg-031e">파슬리=&gt;</td>
    <td class="tg-031e">블랙올리브</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">0.5</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">5</td>
    <td class="tg-031e">플레인요거트=&gt;</td>
    <td class="tg-031e">딸기</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">1.0</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">6</td>
    <td class="tg-031e">딸기=&gt;</td>
    <td class="tg-031e">플레인요거트</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">0.5</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">7</td>
    <td class="tg-031e">사과,양파=&gt;</td>
    <td class="tg-031e">깻잎</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">1.0</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">8</td>
    <td class="tg-031e">달걀,애호박=&gt;</td>
    <td class="tg-031e">페페론치노</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">1.0</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">9</td>
    <td class="tg-031e">사과,양파,파프리카=&gt;</td>
    <td class="tg-031e">깻잎</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">1.0</td>
    <td class="tg-031e">7.1667</td>
  </tr>
  <tr>
    <td class="tg-031e">10</td>
    <td class="tg-031e">달걀,사과,양파=&gt;</td>
    <td class="tg-031e">깻잎</td>
    <td class="tg-031e">0.0697</td>
    <td class="tg-031e">1.0</td>
    <td class="tg-031e">7.1667</td>
  </tr>
</table>

<p>개략적인 해석은 다음과 같습니다.</p>
<ul>
	<li>lhs는 조건절, rhs는 결과절로, '조건절이 있으면 결과절이 있다'라고 해석할 수 있습니다만, 상관관계일 뿐 인과관계를 의미하지는 않습니다.</li>
	<li>하지만 대충 '아이스크림'이 있으면 '초콜릿'이 있겠구나.. '파슬리'가 있으면 '블랙올리브'가 있겠구나 하고 생각하면 됩니다.</li>
	<li>support는 lhs가 있을 확률입니다. support를 0.5로 잡아서 규칙을 뽑으면 달걀밖에 안나옵니다. 앞서 얘기했지만, 달걀이 나올 확률은 거의 60%에 수렴하며, 이 정도 상대 빈도를 보이는 다른 재료는 없기 때문입니다. 이 분석에서는 특이한 재료를 잡아보기 위해 support를 0.05로 잡았습니다.</li>
	<li>confidence는 lhs가 있을 때 rhs가 있을 확률입니다. 여기서는 confidence cut-off를 0.5로 설정했습니다.</li>
	<li>마지막으로 lift는 lhs와 rhs가 독립이 아닐 확률로, 독립일 경우 1이 됩니다. 위에서 뽑은 10가지 규칙이 모두 7.1667인 것으로 보아, 꽤 신빙성있는 규칙이 도출되었다고 볼 수 있습니다.</li>
	<li>결론적으로 냉부에 출연한 연예인같은 냉장고를 가지고 싶다면 마트에 가서 파슬리를 살때 블랙올리브를 함께 집으면 됩니다..</li>
</ul>

<h3>셰프 선택 재료 연관규칙 분석</h3>
<p>앞서 돌려봤던 연관규칙은 셰프가 선택한 재료를 대상으로 할 때 더 의미가 있을 것 같습니다. 즉, 어떤 재료를 사용했을 때 또 어떤 재료를 선택했을 것인가? 라는 질문에 대답을 할 수 있습니다.</p>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
</style>
<table class="tg">
  <tr>
    <th class="tg-031e">#</th>
    <th class="tg-031e">lhs</th>
    <th class="tg-031e">rhs</th>
    <th class="tg-031e">support</th>
    <th class="tg-031e">confidence</th>
    <th class="tg-031e">lift</th>
  </tr>
  <tr>
    <td class="tg-031e">1</td>
    <td class="tg-031e">라조장=&gt;</td>
    <td class="tg-031e">족발</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">2</td>
    <td class="tg-031e">족발=&gt;</td>
    <td class="tg-031e">라조장</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">3</td>
    <td class="tg-031e">트러플오일=&gt;</td>
    <td class="tg-031e">치킨스톡</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">4</td>
    <td class="tg-031e">치킨스톡=&gt;</td>
    <td class="tg-031e">트러플오일</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">5</td>
    <td class="tg-031e">프로슈토=&gt;</td>
    <td class="tg-031e">바게트</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">6</td>
    <td class="tg-031e">바게트=&gt;</td>
    <td class="tg-031e">프로슈토</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">7</td>
    <td class="tg-031e">파르메산치즈=&gt;</td>
    <td class="tg-031e">설렁탕</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">8</td>
    <td class="tg-031e">설렁탕=&gt;</td>
    <td class="tg-031e">파르메산치즈</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">9</td>
    <td class="tg-031e">사워크림=&gt;</td>
    <td class="tg-031e">타코시즈닝</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
  <tr>
    <td class="tg-031e">10</td>
    <td class="tg-031e">타코시즈닝=&gt;</td>
    <td class="tg-031e">사워크림</td>
    <td class="tg-031e">0.011</td>
    <td class="tg-031e">1</td>
    <td class="tg-031e">84.5</td>
  </tr>
</table>
<p>해석은 아래와 같습니다.</p>
<ul>
	<li>이번에는 support cut-off를 0.01로 대폭 낮췄습니다. 기존의 0.05로 돌려봤더니 달걀과 우유(당연한 조합)만 검출이 되었습니다. 냉장고와는 달리 셰프의 레시피에는 더 적은 재료가 다양하게 들어가서 그런 모양입니다. support를 0.01로 낮추고 돌려본 결과, 재밌는 조합이 나옵니다. 이 중에 그냥 써먹을만한 규칙도 보입니다.</li>
	<li>족발은 라조장에 찍어먹자</li>
	<li>프로슈토는 바게트에 넣어먹자 (몰랐는데 프로슈토는 햄이라고 합니다.)</li>
	<li>타코먹을때는 사워크림도 함께..</li>
	<li>마지막으로 설렁탕 먹을땐 파르메산 치즈도 함께 주문하자..(응?)</li>
</ul>

<h2>냉장고 재료 기준 연예인 군집화</h2>
<p>냉장고에 쟁여놓은 재료를 기준으로 비슷한 사람끼리 묶는다면, 누구와 누가 엮일 수 있을까요? 머신러닝의 알고리즘 중 하나인 자기조직화지도(Self-Organising Map(SOM)을 사용하여 군집화를 돌려보았습니다.</p>

<h3>자기조직화지도(SOM)</h3>
![fridge wordcloud](/assets/fp_som.png)

<p>'냉장고를 부탁해'에 소개된 36개의 냉장고에는 459개의 식재료가 들어있습니다. 자기조직화지도를 만드는데 쓰인 데이터베이스에는 각 냉장고의 자료가 범주형으로 저장되어있습니다. 예를 들어.. 강균성 냉장고 - 오이, 사과, 달걀.. 이 레코드를 1,0의 이산형으로 변형시킵니다. 강균성 냉장고 - 1, 0, 1, 1, 0.. 그렇게 되면 36개의 냉장고에 대해 459차원이 생기고, 459차원 상의 냉장고간 유사도를 대폭 줄여 2차원 공간상에 뿌린 것이 자기조직화지도입니다.</p>
<p>r의 Kohonen 패키지를 써서 지도를 만드는데까지는 성공을 했습니다만.. 459차원 상의 유사성을 하나하나 검증할 수 없기에 사실 올바르게 분류되었는지는 알 턱이 없습니다. 원안에 같이 들어있는 인물들끼리는 같은 유형이라고 파악학 수 있으며, 굵은 선은 원들간의 경계를 의미합니다. 격리선이 일부 원들을 감싸고 있는 것으로 보아, 대부분의 냉장고는 상호간 어느정도 유사성을 띄고 있는 것으로 보입니다.</p>	

<h3>계층적 군집화 - 냉장고</h3>
<p>결과물이 썩 맘에 들지 않았던 자기조직화지도 대신에 이번에는 조금 더 명확하게 이해할 수 있는 계층적 군집화 방식을 써보았습니다. 군집 개수를 정해주고 시작해야 하는 k-means clustering과는 달리, 계층적 군집화는 미리 정의를 해줄 필요가 없습니다. 36개의 냉장고가 주어졌을 때 계층적 군집화 알고리즘은 가장 가까운 것 2개를 합쳐서 묶고, 그다음엔 2개씩 묶은 집합을 또 묶어 최종적으로 모든 것을 다 묶을때까지 군집화가 진행됩니다. 사전 군집 개수를 정하지 않아도 된다는 장점 이외에도 체계적으로 군집이 나뉘어 상대적으로 이해하기 쉽습니다. 본 분석에서는 아래에서 위로 올라가는 상향식 군집화를 썼습니다. 어떤 능력자 외국분이 r에서 바로 d3로 익스포트하는 코드를 만들어놔서 한방에 결과물이 나왔습니다. <a href="http://www.coppelia.io/2014/07/converting-an-r-hclust-object-into-a-d3-js-dendrogram/">관련 링크</a></p>

<iframe width="100%" height="850" src="//jsfiddle.net/junkwhinger/tcr8u94a/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<p>위에 자기조직화지도 결과물과 비교해보면 차이점이 보입니다. SOM에서는 김기방과 문희준을 묶어놨지만, 계층적 군집화 결과에서는 2번째 상위노드인 node33에 가서야 겨우 하나로 묶입니다. "강남,사유리"는 맞췄네요. 자기조직화지도는 개념은 굉장히 멋지지만, 개인적으로는 계층적 군집화에 더 믿음이 갑니다.</p>

<h3>계층적 군집화 - 셰프의 레시피</h3>

<p>냉장고를 부탁해에 등장한 132종의 레시피를 재료를 기준으로 계층적 군집화를 돌려보았습니다.</p>
<ul>
  <li>레시피에 쓰인 재료를 기준으로 군집화를 돌렸기 때문에 맨 우측에 같은 회에 출연한 (동일한 냉장고를 사용한) 레시피가 많이 뭉쳐있습니다.</li>
  <li>같은 회에 출연한 레시피가 뭉친 경우, 셰프들이 비슷한 재료를 사용했다고 볼 수 있습니다. 하지만 홍석천의 '렛잇컵'(8회)와 미카엘의 '따라미소'(29회)는 시기가 다르지만 재료 구성에서는 가장 유사하다고 합니다.</li>
  <li>좋아하는 레시피를 찾아보세요. 그 레시피와 함께 묶인 레시피도 좋아할 가능성이 높습니다.</li>
</ul>    
<iframe width="100%" height="2350" src="//jsfiddle.net/junkwhinger/06oLbqyL/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<h3>계층적 군집화 - 셰프</h3>
<iframe width="100%" height="350" src="//jsfiddle.net/junkwhinger/2ppme8hn/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
