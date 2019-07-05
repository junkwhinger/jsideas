---
layout:     post
title:      "Avengers 2012/2015 Comparison"
date:       2015-06-04 02:00:00
author:     "Jun"
author:     "Jun"
tags: [d3.js, data visualisation]
---

<h2 class="section-heading">Background</h2>

<p>Marvel's lastest film, <a href="http://www.imdb.com/title/tt2395427/">Avengers: Age of Ultron</a> reaped a huge success over here in South Korea. I myself am a huge fan of them and went to the cinema to check it out. One thing that really surprised me was the secret existance of Hawk_Eye's family. It didn't appear officiallly, but I'd always thought Black_Widow was kinda seeing Hawk_Eye (mentioning stuff of Budapest in the first film). And it turned out that Black_Widow had a thing for Hulk! This joyful revelation inspired me to make some visualisations that would show how the main characters' relationships changed in the original Avengers and its sequal.</p>

<h2 class="section-heading">Methodology</h2>
<p>The data collection method was exactly the same as my previous analysis on Korean version of Game of Throne (but based on real history). I wrote down the names of characters who appeared together. To narrow down the focus, I only took the following characters into account:</p>

<ul>
	<li>good characters: Quick_Silver, Scarlet Witch (when they were on Ultron's side), Loki, Ultron</li>
	<li>real characters: Peggy and Heimdall that appeared in hallucination</li>
</ul>

<p>For example, the data collected through the process introduced above look like this:</p>
<ul>
	<li>Hawk_Eye, Black_Widow</li>
	<li>Black_Widow, Captain_America, Thor, Iron_Man</li>
	<li>Iron_Man, Hulk, Captain_Amercia</li>
</ul>

<p>Then, before putting these data into a network, I transformed scenes into all unique pairs, and gave weight to them according to the number of characters of each scene. A 2-people scene must be different from a 3-people scene. For instance, in a 2-people scene, the number of pairs will be 1, thus the weight of a pair is 1. In a 3-people scene, there will be 3 unique pairs with a weight of 1/3 each. A 4-people scene would have 6 pairs with 1/6 weight each.</p>

<p>When the ingredient was fully ready, I created graphs out of it with Python's networkx library that did the hard works calculating centrality measures like Eigenvector Centrality. And the following charts are created and uploaded on jsfiddle using d3.js.</p>

<h2 class="section-heading">Character Network Graph</h2>

<iframe width="100%" height="550" src="//jsfiddle.net/junkwhinger/d8gz64zf/28/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<p>Try...</p>
<ol>
  <li>Double click a node to check out a character's immediate neighbours.</li>
  <li>Mouse on a node to check out a character's detailed info.</li>
  <li>Click the radio buttons above the charts to see who's how powerful based on different network cetralities. </li>
</ol>

<p>The graphs above show the relationships between good guys and girls in two Avengers films. The bigger the node is, the bigger the selected centrality measure of that character. A character with a high frequency means that he or she appeared a lot in a film. A character with a high degree centrality means that he or she is connected to a lot of other characters. For example, Thor in Avengers 2012 has a lower frequency than Captain_America, but in terms of Degree Centrality He is as big as the Captain as he's connected to 8 other characters. A character with a high betweenness centrality means that the character is placed in the middle area of the whole network. And I believe in this particular example, if that character sits in the middle of a friendly network, he or she is likely to be a leader among others. In the original film, the betweenness leader is Iron_Man, but in the sequal, it seems that Thor took the lead by awakening Vision after a mysterious dive into that black well. And finally, a character with a high eigenvector centrality indicates the level of influence of that person in the network. The more powerful neighbors you have, the more eigenvector value you'd have. And of course, the thicker a link is, the more intimate relationship it means. It's fun to see how Black_Widow's relationship has changed in two films. Click <a href="http://en.wikipedia.org/wiki/Centrality">here</a> to check out more about network centrality.</p>

<div class="bumper"></div>

<h2>Network Centrality Bar Chart</h2>

<iframe width="100%" height="700" src="//jsfiddle.net/junkwhinger/v9t4f58r/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<p>This chart above shows how each characters' network centrality measures changed over two films. It practically shows the same information but in a different format that clearly shows the increase and decrese of measures.</p>

<div class="bumper"></div>

<h2>Character Intimacy Network</h2>
<iframe width="100%" height="700" src="//jsfiddle.net/junkwhinger/yn8e2czt/21/embedded/result" allowfullscreen="allowfullscreen" frameborder="0"></iframe>

<p>To make the relationships of the main 6 guys more clear, I extracted each main character's network and visualised as above. And now it becomes more obvious that Black_Widow had a stronger link with Hulk than with Hawk_Eye in the Age of Ultron. As two network graphs are merged to create this visualisation, all node size is fixed.</p>

<p>Thank you for reading!</p>


<link href="/d3_css/avengers_comparison.css" rel="stylesheet">

