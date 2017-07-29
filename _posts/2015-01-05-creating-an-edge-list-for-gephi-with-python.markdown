---
layout:     post
title:      "Creating an edge list for Gephi with Python"
date:       2015-01-05 22:10:00
author:     "Jun"
categories: "Python"
header-img: "img/post-bg-05.jpg"
---
<h2 class="section-heading">Creating an edge list for Gephi with Python</h2>

<p>For various projects including my personal ones I frequently use Gephi to make some interesting network graphs. Last summer in 2014, I did <a href="http://www.slideshare.net/jdjmania/jdj-network-analysisvf">a brief analysis(it's done in Korean)</a> on the story flow of a Korean History Drama (kinda like The Game of Thrones, but based on history) by drawing graphs with <a href="https://networkx.github.io/">NetworkX(a python library for network analysis)</a> and <a href="http://gephi.github.io/">Gephi</a>. In doing that I was in need of making edge lists of characters in csv so that Gephi could chew it nicely.</p>

<p>And I've just tweeked a bit of the code so that it changes this</p>

<p>Banana, Apple, Pear<br>Pear, Apple</p>

<p>to this, which is a bunch of all possible unique pairs.</p>
<p>Banana, Apple<br>Apple, Pear<br>Banana, Pear<br>Pear, Apple</p>

<p>The code and use are available on my <a href="https://github.com/junkwhinger/edge_list_creator">github</a>!</p>

<p>For data in Korean, you have to make sure that your csv is in UTF-8 not in ANSI.</p>

<p>[EDITED] After uploading the code, I found a minor error that Gephi could not read the type of edges this code had generated. I found a plugin called "Convert Excel and csv files to networks (including dynamic!)" on Gephi that allows you to directly import csv and excel files to Gephi. And to make my code work with the plugin, I deleted the header bit(Source,Target,Type). Please find the detailed instruction on my <a href="https://github.com/junkwhinger/edge_list_creator">github</a>. Thank you.</p>



{% highlight python %}

import csv

def pairing(source):
  result = []
  for item1 in range(len(source)):
    for item2 in range(item1+1, len(source)):
      result.append([source[item1], source[item2]])
  return result

filename = raw_input('file name? ')

f = open (filename, 'rb')

csv_f = csv.reader(f, quoting=csv.QUOTE_ALL)

temp_list = []
for row in csv_f:
  row = [var for var in row if var]
  temp_list.append(pairing(row))

final_list = []
for itemlist in temp_list:
  for item in itemlist:
    final_list.append(item)

new_filename = "done_" + filename

with open(new_filename, 'wb') as csvfile:
    rowwriter = csv.writer(csvfile, delimiter=',', quotechar=' ')
    for row in final_list:
      rowwriter.writerow(row)

{% endhighlight %}

{% include google_analytics.html %}
