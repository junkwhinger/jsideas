---
layout: post
title:  "The Hacking Team - 'SKA' emails"
date:   2015-07-21 00:34:25
categories: python
image: /assets/nsa_header.jpg
---

<h2>'The Hacking Team' Crisis in South Korea</h2>

이탈리아 해킹팀 `The Hacking Team`이 해킹을 당하면서 풀린 수많은 고객 리스트. 그 중에 국정원이 속해 있어 많은 파장을 불러일으키고 있습니다. 언론매체 [시사인][시사인]에서 [위키리크스][위키리크스]에 올라온 `The Hacking Team`의 이메일 유출 자료를 소개했는데요, 직접 들어가서 `SKA`(South Korea Army), '5163', 'nanatech'가 포함된 2,197개의 이메일 자료를 내려받아 `python`과 `d3.js` 사용하여 간단히 분석을 해보았습니다. 

본 이슈가 내포하고 있는 정치적, 사회적 파장을 고려하여 어떤 명확한 분석 결과를 내기보다는 `메일 발송량`, `메일 회신 및 회람 수`, `메일 송수신자` 등 위키리크스에서 `공개된 사실`만을 기반으로 인터랙티브 차트를 만들어보았습니다.

<hr />

<h3>자료 수집 및 분석 방법</h3>

* 수집 대상 자료: 위키리크스 `The Hacking Team` 유출 자료 중 `SKA`, `5163`, `nanatech` 포함 이메일 (중복 제거)
* 크롤링, DB화 및 네트워크: python (beautifulsoup, pandas, networkx)
* 시각화: d3.js

유출된 메일 중 해킹팀 소속의 이메일은 'hackingteam.com'으로 통일하였으며, 'naga@hackingteam.com'처럼 가명이나 익명으로 된 주소는 메일 본문 및 외신기사를 참조하여 해당 인물에 매칭시켰습니다.

<hr />

<h3>언제 메일 커뮤니케이션이 이루어졌나</h3>
메일 상의 `SKA`는 최근 공개되는 정황 상 국정원이 거의 확실해 보입니다. 안타깝게도 어제 해킹 소프트웨어와 관련된 국정원 진원이 목숨을 끊으면서 이제 핵심은 국정원이 이 해킹 소프트웨어를 어디에 무슨 용도로 사용했나로 집중되고 있습니다. 5163부대에 관한 이메일의 발송 시점을 보면 아래의 차트와 같습니다. 해당 시점의 바(bar)위에 마우스를 올려보세요.

<iframe src="http://bl.ocks.org/junkwhinger/raw/2d867a14be51c876fb84/" width="100%" height="600px" marginwidth="0" marginheight="0" scrolling="no" frameBorder="0"></iframe>

공개된 메일 중 검색 키워드가 포함된, 즉 5163부대와 관련된 메일은 2010년 8월부터 나오기 시작합니다. 해당 시점의 바에 마우스를 올려보면 해당월에 가장 많이 대화가 진행되었던 (Re:, Fwd: 등이 가장 많은) 이메일을 찾아볼 수 있는데, `nanatech`가 제목과 수신인에 등장합니다. 메일 본문을 추적해보면 2010년 8월부터 프로그램 도입에 대한 논의가 오간 모양입니다. 실제로 2010년 8월에 표시된 <a href="https://wikileaks.org/hackingteam/emails/emailid/440959#searchresult">440959번</a> 메일을 보면 아래와 같은 답변 메일이 달려있습니다.

![email-440959](/assets/ska_leaks/sales.png)

내용 말미에 보면 `스카이프와 같은 암호화된 컨텐츠에 대한 접근`을 위해 `다른 이탈리안 회사`를 소개해주는데 그게 바로 `The Hacking Team`입니다. 그리고 이 메일이 해킹팀에도 전해져 5163부대로 `Remote Control System` 관련 문서를 전송하는 등 향후 계약을 위한 논의가 준비되는 것으로 보입니다.

최초 메일 교환 이후로 여러번의 피크가 있었는데 메일의 제목과 본문으로 간략히 추론해본 이슈는 다음과 같습니다.

* 2011-2012 연말연초: 한국측 파트너(나나테크 추정)와 엔드 유저(5163부대 추정)의 해킹팀 밀라노 사무실 방문. 계약 논의, 계약금 지급
* 2012 중순: 추가계약 논의, 필드 어플리케이션 엔지니어와 어카운트 매니저의 한국 방문
* 2013 연초: 유지보수, 트레이닝, 나나테크 커미션 논의
* 2014 연초: 해킹팀 소프트웨어에 대한 언론 노출 <a href="https://wikileaks.org/hackingteam/emails/emailid/691508#searchresult">[691508]</a>
* 2014 연말: 유지보수
* 2015 중순: 유지보수 및 트레이닝

국내 주요 정치 이슈를 살펴보면 계약이 이루어진 후인 2012년 말에 제 18대 대통령 선거가 있었으며, 2014년 6월에는 제 6회 지방선거가 있었습니다. 모든 메일을 읽어보지는 못했지만 구체적인 타겟에 대한 논의보다는 기술 유지보수에 대한 내용이 주로 발견되었으며, 엔드 유저가 원하는 키워드 중 하나는 `bomb`(폭탄)이라는 내용도 있었습니다. 

이외에도 흥미로운 대화와 문의가 많이 있습니다. 제가 미처 발견하지 못한 내용은 범례에 `[longest email conversation - ]` 옆에 있는 메일 id를 `위키리크스`에서 찾아보세요.

<hr />

<h3>어떤 대화가 중요한 대화였나?</h3>
`메일 회신 및 회람 수`의 분포는 어떻게 될까요?
<iframe src="http://bl.ocks.org/junkwhinger/raw/6d2e86d240405195a02b/70e8c23d3378238cd5f4e08cf014730305b693c9/" width="100%" height="600px" marginwidth="0" marginheight="0" scrolling="no" frameBorder="0"></iframe>
공개된 메일 중 약 90%가 0~3번 회신되었으나, 10번(0.05%), 9번(0,43%)으로 굉장히 많은 대화가 오갔던 내용들도 있었습니다. 회신과 회람이 다수 발생한 메일일수록 중요 내용을 담고 있다고 가정하고, 아래와 같이 월별로 가장 많은 회신 및 회람 수를 기록한 메일 리스트를 뽑았습니다. (위 바차트의 범례에 등장하는 메일과 동일합니다.)

<br>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
</style>
<table class="tg">
  <tr>
    <th class="tg-031e">date_x</th>
    <th class="tg-031e">id</th>
    <th class="tg-031e">from</th>
    <th class="tg-031e">to</th>
    <th class="tg-031e">subject</th>
  </tr>
  <tr>
    <td class="tg-031e">2010-08</td>
    <td class="tg-031e">440959</td>
    <td class="tg-031e">m.bettini@hackingteam.it</td>
    <td class="tg-031e">['emanuele.marcozzi@area.it', 'paolo.mandelli@area.it', 'nazareno.saguato@area.it', 'm.luppi@hackingteam.it']</td>
    <td class="tg-031e">R: RE: Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2010-09</td>
    <td class="tg-031e">440767</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['nanatechp@paran.com']</td>
    <td class="tg-031e">R: Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2010-10</td>
    <td class="tg-031e">440886</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['nanatechp@paran.com', 'rsales@hackingteam.it']</td>
    <td class="tg-031e">R: Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2010-11</td>
    <td class="tg-031e">440940</td>
    <td class="tg-031e">nanatechp@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it']</td>
    <td class="tg-031e">Re: R: Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2010-12</td>
    <td class="tg-031e">441020</td>
    <td class="tg-031e">nanatechp@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it', 'nanatech@paran.com', 'nanatechhan@paran.com']</td>
    <td class="tg-031e">Re: R: Meeting in Seoul</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-01</td>
    <td class="tg-031e">441112</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['nanatechp@paran.com']</td>
    <td class="tg-031e">R: BULK  Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-02</td>
    <td class="tg-031e">441159</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['nanatechp@paran.com']</td>
    <td class="tg-031e">R: Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-03</td>
    <td class="tg-031e">441372</td>
    <td class="tg-031e">nanatech@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it', 'nanatechhan@paran.com', 'nanatechp@paran.com']</td>
    <td class="tg-031e">Re: R: BULK Re: Fwd: R: Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-05</td>
    <td class="tg-031e">441390</td>
    <td class="tg-031e">nanatech@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it', 'nanatechco@paran.com']</td>
    <td class="tg-031e">Re: R: Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-06</td>
    <td class="tg-031e">440946</td>
    <td class="tg-031e">nanatechco@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it']</td>
    <td class="tg-031e">Re: R: R: Hello, it' from Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-07</td>
    <td class="tg-031e">440860</td>
    <td class="tg-031e">nanatechco@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it']</td>
    <td class="tg-031e">Re: R: R: R: R: Hello, it' from Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-08</td>
    <td class="tg-031e">440897</td>
    <td class="tg-031e">m.bettini@hackingteam.it</td>
    <td class="tg-031e">['m.luppi@hackingteam.it']</td>
    <td class="tg-031e">R: Re: R: R: it's from Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-09</td>
    <td class="tg-031e">440970</td>
    <td class="tg-031e">nanatechco@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it']</td>
    <td class="tg-031e">Re: R: Reply about visit</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-10</td>
    <td class="tg-031e">441077</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['m.valleri@hackingteam.it', 'naga@hackingteam.it', 'rsales@hackingteam.it']</td>
    <td class="tg-031e">R: R: R: Hello, it's from Nanatech.</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-11</td>
    <td class="tg-031e">440636</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['nanatechco@paran.com', 'rsales@hackingteam.it']</td>
    <td class="tg-031e">R: BULK  Re: R: BULK Hello, It's NANATECH</td>
  </tr>
  <tr>
    <td class="tg-031e">2011-12</td>
    <td class="tg-031e">441035</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['nanatech@paran.com']</td>
    <td class="tg-031e">R: R: R: R: R: BULK Re: Re: BULK Re: Re: contract</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-01</td>
    <td class="tg-031e">588949</td>
    <td class="tg-031e">f.busatto@hackingteam.it</td>
    <td class="tg-031e">['alberto@hackingteam.it', 'g.russo@hackingteam.it', 'delivery@hackingteam.it', 'm.luppi@hackingteam.it']</td>
    <td class="tg-031e">R: Re: R: Re: R: Re: R: Re: R: Certificato Symbian per SKA</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-02</td>
    <td class="tg-031e">440870</td>
    <td class="tg-031e">g.russo@hackingteam.it</td>
    <td class="tg-031e">['nanatech@paran.com', 'm.luppi@hackingteam.it', 'm.bettini@hackingteam.it']</td>
    <td class="tg-031e">Re: Fwd: Re: R: 2nd payment</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-03</td>
    <td class="tg-031e">441407</td>
    <td class="tg-031e">nanatech@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it']</td>
    <td class="tg-031e">Re: R: R: R: quote maintenance</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-04</td>
    <td class="tg-031e">960672</td>
    <td class="tg-031e">nanatech@paran.com</td>
    <td class="tg-031e">['f.busatto@hackingteam.it']</td>
    <td class="tg-031e">Re: Re: Very Urgent</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-05</td>
    <td class="tg-031e">440954</td>
    <td class="tg-031e">nanatech@paran.com</td>
    <td class="tg-031e">['m.luppi@hackingteam.it']</td>
    <td class="tg-031e">Re: R: Re: Information</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-06</td>
    <td class="tg-031e">761917</td>
    <td class="tg-031e">a.scarafile@hackingteam.it</td>
    <td class="tg-031e">['nanatech@paran.com']</td>
    <td class="tg-031e">Re: R: Seoul 9-11 July</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-07</td>
    <td class="tg-031e">441311</td>
    <td class="tg-031e">m.luppi@hackingteam.it</td>
    <td class="tg-031e">['nanatech@paran.com', 'm.bettini@hackingteam.it']</td>
    <td class="tg-031e">R: Re: R: New</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-08</td>
    <td class="tg-031e">440944</td>
    <td class="tg-031e">bruno@hackingteam.it</td>
    <td class="tg-031e">['m.luppi@hackingteam.it', 'delivery@hackingteam.it']</td>
    <td class="tg-031e">Re: I: Re: Re: Contract (Urgent)</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-09</td>
    <td class="tg-031e">829116</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['a.scarafile@hackingteam.com']</td>
    <td class="tg-031e">RE: Re: R: Re: Additional Order</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-11</td>
    <td class="tg-031e">440953</td>
    <td class="tg-031e">bruno@hackingteam.it</td>
    <td class="tg-031e">['zeno@hackingteam.it', 'a.pelliccione@hackingteam.it', 'rcs-support@hackingteam.com']</td>
    <td class="tg-031e">Richiesta di SKA - Dispositivi Android che supportano la registrazione della chiamata</td>
  </tr>
  <tr>
    <td class="tg-031e">2012-12</td>
    <td class="tg-031e">440734</td>
    <td class="tg-031e">g.russo@hackingteam.it</td>
    <td class="tg-031e">['m.luppi@hackingteam.it', 'm.bettini@hackingteam.it']</td>
    <td class="tg-031e">Re: I: R: R: New Order (URGENT)</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-01</td>
    <td class="tg-031e">727025</td>
    <td class="tg-031e">s.woon@hackingteam.com</td>
    <td class="tg-031e">['daniel']</td>
    <td class="tg-031e">Re: Fw: RE: RE: Maintenance Contract(URGENT)</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-02</td>
    <td class="tg-031e">441401</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com', 'm.luppi@hackingteam.it']</td>
    <td class="tg-031e">RE: RE: RE: RE: Re: Re: Connector</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-03</td>
    <td class="tg-031e">606424</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['nanatechheo@daum.net', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">RE: RE: RE: RE: Training</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-04</td>
    <td class="tg-031e">449488</td>
    <td class="tg-031e">g.russo@hackingteam.it</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com', 'a.capaldo@hackingteam.it', 'm.bettini@hackingteam.com', 'amministrazione@hackingteam.it', 'rsales@hackingteam.it']</td>
    <td class="tg-031e">Commissions to our agent/broker was Re: Ordine x commissioni Nanatech</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-05</td>
    <td class="tg-031e">440791</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com', 'm.luppi@hackingteam.it']</td>
    <td class="tg-031e">RE: RE: Re: Training (Urgent)</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-07</td>
    <td class="tg-031e">729366</td>
    <td class="tg-031e">s.woon@hackingteam.com</td>
    <td class="tg-031e">['nanatech', 'daniel']</td>
    <td class="tg-031e">Re: Help</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-08</td>
    <td class="tg-031e">346024</td>
    <td class="tg-031e">daniel@hackingteam.com</td>
    <td class="tg-031e">['fulvio@hackingteam.it', 'd.maglietta@hackingteam.com', 'russo@hackingteam.it', 'm.catino@hackingteam.com', 'f.degiovanni@hackingteam.com', 'marco.bettini@hackingteam.it', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">Re: 2nd Payment</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-09</td>
    <td class="tg-031e">18783</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['nanatechheo@daum.net', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">Re: RE: RE: RE: Further items</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-10</td>
    <td class="tg-031e">18895</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['nanatechheo@daum.net', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">Re: RE: Further items</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-11</td>
    <td class="tg-031e">392626</td>
    <td class="tg-031e">g.russo@hackingteam.com</td>
    <td class="tg-031e">['s.gallucci@hackingteam.com']</td>
    <td class="tg-031e">Re: R: Fwd: FW: RE: Re: Invoice</td>
  </tr>
  <tr>
    <td class="tg-031e">2013-12</td>
    <td class="tg-031e">441339</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com', 'm.luppi@hackingteam.it']</td>
    <td class="tg-031e">RE: RE: RE: RE: RE: RE: RE: RE: RE: Re: Offer</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-01</td>
    <td class="tg-031e">440825</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com', 'm.luppi@hackingteam.it']</td>
    <td class="tg-031e">RE: Re: RE: RE: RE: RE: RE: RE: Maintenance</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-02</td>
    <td class="tg-031e">18934</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['nanatechheo@daum.net', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">RE: RE: Re: RE: RE: RE: RE: RE: RE: Maintenance</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-03</td>
    <td class="tg-031e">691508</td>
    <td class="tg-031e">d.maglietta@hackingteam.it</td>
    <td class="tg-031e">['serge@hackingteam.com']</td>
    <td class="tg-031e">FW: RE: RE: RE: Top Urgent</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-04</td>
    <td class="tg-031e">692654</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['d.maglietta@hackingteam.it', 's.woon@hackingteam.it']</td>
    <td class="tg-031e">RE: Re: RE: RE: Tactical\xa0Network\xa0Injector</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-05</td>
    <td class="tg-031e">710963</td>
    <td class="tg-031e">f.cornelli@hackingteam.it</td>
    <td class="tg-031e">['s.woon@hackingteam.it', 'bug@hackingteam.com']</td>
    <td class="tg-031e">Re: Samsung Knox and local root</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-06</td>
    <td class="tg-031e">470961</td>
    <td class="tg-031e">m.valleri@hackingteam.it</td>
    <td class="tg-031e">['a.mazzeo@hackingteam.it', 'd.milan@hackingteam.it', 'a.ornaghi@hackingteam.it', 'naga@hackingteam.it', 'cod@hackingteam.it']</td>
    <td class="tg-031e">R: Re: sample su VT</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-07</td>
    <td class="tg-031e">959731</td>
    <td class="tg-031e">b.muschitiello@hackingteam.com</td>
    <td class="tg-031e">['marco', 'daniel']</td>
    <td class="tg-031e">Fwd: Re: Fwd: TNI</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-08</td>
    <td class="tg-031e">203540</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['amministrazione@hackingteam.it', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">Fw: Invoice</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-09</td>
    <td class="tg-031e">1001854</td>
    <td class="tg-031e">f.busatto@hackingteam.com</td>
    <td class="tg-031e">['daniel', 'serge', 'marco']</td>
    <td class="tg-031e">Re: SKA opportunity</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-10</td>
    <td class="tg-031e">46211</td>
    <td class="tg-031e">g.russo@hackingteam.com</td>
    <td class="tg-031e">['anita', 'david']</td>
    <td class="tg-031e">Fwd: Re: Next meeting alternative dates</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-11</td>
    <td class="tg-031e">145064</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['nanatechheo@daum.net', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">RE: RE:\xa0\xa0RE:\xa0RE:\xa0Re:\xa0Re:\xa0Answer</td>
  </tr>
  <tr>
    <td class="tg-031e">2014-12</td>
    <td class="tg-031e">987044</td>
    <td class="tg-031e">a.dipasquale@hackingteam.com</td>
    <td class="tg-031e">['f.busatto@hackingteam.com']</td>
    <td class="tg-031e">R: Re: R: Re: R: LVM</td>
  </tr>
  <tr>
    <td class="tg-031e">2015-01</td>
    <td class="tg-031e">18854</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['nanatechheo@daum.net', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">RE: RE: RE: RE: RE: RE: Maintenance</td>
  </tr>
  <tr>
    <td class="tg-031e">2015-02</td>
    <td class="tg-031e">18931</td>
    <td class="tg-031e">d.maglietta@hackingteam.com</td>
    <td class="tg-031e">['nanatechheo@daum.net', 'simonetta@hackingteam.com', 'rsales@hackingteam.com']</td>
    <td class="tg-031e">RE: RE: RE: RE: Question</td>
  </tr>
  <tr>
    <td class="tg-031e">2015-03</td>
    <td class="tg-031e">26529</td>
    <td class="tg-031e">s.gallucci@hackingteam.com</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com', 'g.russo@hackingteam.com']</td>
    <td class="tg-031e">R: RE: FW: RE: RE: RE: RE: RE: RE: Question</td>
  </tr>
  <tr>
    <td class="tg-031e">2015-04</td>
    <td class="tg-031e">22576</td>
    <td class="tg-031e">m.bettini@hackingteam.com</td>
    <td class="tg-031e">['f.busatto@hackingteam.com', 'g.russo@hackingteam.com', 'm.bettini@hackingteam.it']</td>
    <td class="tg-031e">R: Re: R: Fwd: !SIX-648-45157: Support portal available time</td>
  </tr>
  <tr>
    <td class="tg-031e">2015-05</td>
    <td class="tg-031e">642183</td>
    <td class="tg-031e">l.guerra@hackingteam.com</td>
    <td class="tg-031e">['c.vardaro@hackingteam.com', 'b.muschitiello@hackingteam.com', 'f.busatto@hackingteam.com']</td>
    <td class="tg-031e">Re: SKA: Servers change for Proxy System</td>
  </tr>
  <tr>
    <td class="tg-031e">2015-06</td>
    <td class="tg-031e">1052404</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com']</td>
    <td class="tg-031e">RE: RE: Re: RE: Maintenance</td>
  </tr>
  <tr>
    <td class="tg-031e">2015-07</td>
    <td class="tg-031e">1135900</td>
    <td class="tg-031e">nanatechheo@daum.net</td>
    <td class="tg-031e">['d.maglietta@hackingteam.com']</td>
    <td class="tg-031e">RE: RE: RE: Re: RE: Maintenance</td>
  </tr>
</table>


<hr />

<h3>누가 가장 많이 보내고 누가 가장 많이 받았나</h3>
5163부대의 계약과 관련된 메일의 최다 송신 및 수신자는 누구일까요?

<iframe src="http://bl.ocks.org/junkwhinger/raw/f3afe90bdf0789f20386/3c2cf10ca33f25c8b0c35be7f1d128b4f65cefb5/" width="100%" height="600px" marginwidth="0" marginheight="0" scrolling="no" frameBorder="0"></iframe>

가장 많은 메일을 발송한 사람은 `David Maglietta`로, 이 사람은 해킹팀의 싱가포르 지점 대표입니다. 메일 내용으로 보아 주로 밀라노 본사와 나나테크 사이를 중개한 것으로 추정됩니다. 이어 동아시아 담당 Key Account Manager인 `Massimiliano Luppi`, Senior Security Consultant라는 `S.Woon`이 상당한 양의 메일을 발송했습니다. `S.Woon`의 풀네임은 `Serge Woon`으로, 성(last name)으로 미루어보아 동아시아인일 가능성이 있어 보입니다.

<br>
다음은 최다 수신자 차트입니다.

<iframe src="http://bl.ocks.org/junkwhinger/raw/0bf01eb6bad614dc7857/" width="100%" height="600px" marginwidth="0" marginheight="0" scrolling="no" frameBorder="0"></iframe>

최다 수신자 상위 20인에서는 싱가포르 지점대표는 없어졌지만, Key Account Manager인 `Massimiliano Luppi`와 Sales Manager인 `Marco Bettini`가 눈에 니다. 발신자와 수신자 리스트에서 CEO, CIO 등 C-level 고위 임원들이 빠지지 않는 것으로 보아, 5163 부대와의 계약에 해킹팀이 상당한 신경을 쓰고 있었을거라 추측됩니다. 단순한 발신/수신량보다 누가 누구와 대화를 많이 나누었는지가 더 중요할 것으로 보입니다. 이는 다음 파트에서 네트워크를 시각화하여 살펴보겠습니다.

<hr />

<h3>누가 누구와 메일을 주고 받았나</h3>

아래의 네트워크 차트는 `메일 발신자`(source) -> `메일 수신자`(target)의 관계를 시각화한 것으로, 10회 이상의 관계만 필터링하여 시각화 한 것입니다. 각 노드의 색깔은 직책별로 분류하였으며, 노드의 크기는 네트워크 상의 매개중심성(`betweeness centrality`)을 의미합니다. 즉 네트워크의 중앙에 위치할수록 노드의 크기가 큽니다. 노드간 링크의 굵기가 굵을수록 메일이 많이 오고갔습니다.

* 마우스 오버: 해당 인물 직책 툴팁 활성
* 노드 더블클릭: 직접적으로 연결된 포인트 하이라이트
* 마우스 휠: 줌 인 & 줌 아웃
 
<iframe src="http://bl.ocks.org/junkwhinger/raw/81d9432e97fa3c92b42f/5eb035290bd7bb85c222d670d04595f4180e4fa0" width="100%" height="990px" marginwidth="0" marginheight="0" scrolling="no" frameBorder="0"></iframe>

위 차트에서 다음과 같은 사실을 발견할 수 있습니다.

* 최중심에 위치한 Key Account Manager: 전체 네트워크에서 가장 중심적인 역할(브로커)을 수행한 자는 `Massimiliano Luppi`로, 5163부대 계약과 관련하여 nanatech 측과 Hacking Team 사이를 연결한 것으로 보입니다. 주요 연결선을 보면 `nanatech@paran.com`으로부터 받은 내용을 COO인 `g.russo`, Sales Manager인 `m.bettini`, 그리고 필드 엔지니어인 `a.scarafile`과 공유한 것으로 보입니다.

* nanatech 이메일: nanatech로 검색된 이메일은 총 4개입니다. 이 중 `nanatechp@paran.com`과 'nanatechco@paran.com'은 주요 연결선이 `m.luppi`에만 닿아있는데 반해 허순구 대표의 이메일인 `nanatechheo@daum.net`이나 `nanatech@paran.com`은 COO인 `g.russo`와 연결되는 것으로 보아 더 중요한 메시지가 오간 것으로 추측됩니다.
 
* The Hacking Team의 조직구조: 이메일 연결 및 본문을 통해 살펴본 결과, The Hacking Team의 조직은 CEO, COO, CIO, CTO의 `C-level`, Software Developer, Senior Security Engineer의 `Dev Team`, Sales Manager, Key Account Manager, Field Application Engineer 등 고객의 접점에서 활동하는 `Field Team`으로 크게 분류할 수 있습니다. 또한 The Hacking Team의 조직구조는 매우 유연해보였습니다. Client와의 주요한 이메일 내용은 회람을 통해 C-Level 의사결정자에게 바로 전해지거나 `sharepoint@hackingteam.com`과 같은 공유 메일계정을 통해 CIO 등 유관 담당자에게도 정보가 적극적으로 공유되는 것으로 보입니다.

<hr />

<h3>마치며</h3>

5163부대가 누구를 타겟으로 RCS 프로그램을 사용했는지는 아직 아무도 모릅니다. 다만 분석 결과, 2011년 말~2012년 초의 어느 시점부터 5163부대에서 RCS 프로그램을 운용해왔고, 지속적인 추가계약, 유지보수, 트레이닝을 받아왔으며, 이 과정에서 양측이 한국과 밀라노에서 지속적으로 접선해왔다는 것이 드러났습니다. 5163부대는 어떤 목적으로 RCS 프로그램을 사용했던 걸까요? 현 국정원의 주장처럼 내국인 사찰은 없었던 것일까요? 최근에 안타깝게 자살을 선택한 국정원 직원은 The Hacking Team이나 Nanatech와 어떤 관계가 있었을까요? 우리나라의 사이버 보안을 해치지 않으면서도 국정원의 잘못된 점이 있었다면 확실히 개선하여 국민들이 가진 의구심을 해소할 수 있도록 현 국정원 사태가 공명정대한 결말로 이어졌으면 좋겠습니다.  

[시사인]:		   http://m.sisainlive.com/news/articleView.html?idxno=23895
[위키리크스]:		   https://wikileaks.org/hackingteam/emails
