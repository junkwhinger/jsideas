---
layout:     post
title:      "Sound of Mind"
date:       2015-03-29 18:50:00
author:     "Jun"
categories: "Python"
header-img: "img/post-bg-05.jpg"
---

<h2 class="section-heading">Sound of Mind - rating change in time</h2>

<p>It’s been a long time since I last wrote a posting.
To brush up on python and R, I worked on a tiny project that was to crawl episode and ratings data of a famous webtoon, <a href="http://comic.naver.com/webtoon/list.nhn?titleId=20853" target="_blank">Sound of Mind</a>. Using the default ‘plot’ in R, I drew the following chart which shows two darkest times in which the ratings significantly dropped.</p>

![ms_image](/assets/ms_Rplot.png)

{% highlight python %}

#-*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from time import sleep
import urllib
from bs4 import BeautifulSoup
from xlwt import Workbook


ms_url = 'http://comic.naver.com/webtoon/list.nhn?titleId=20853&page=1'
ms_data = urllib.urlopen(ms_url)
ms_soup = BeautifulSoup(ms_data)

l_episode = ms_soup.findAll('td', attrs={'class':'title'})

latest_episode = l_episode[0].find('a').contents[0]
latest_episode_num = int(latest_episode.split(".")[0])
end_page_index = int(latest_episode_num / 10) + 1


episode_num_list = list()
episode_date_list = list()
episode_ratings_list = list()
episode_link_list = list()

for page in range(1,end_page_index+1):
	page_url = 'http://comic.naver.com/webtoon/list.nhn?titleId=20853&page=' + str(page)
	page_data = urllib.urlopen(page_url)
	page_soup = BeautifulSoup(page_data)
	total_episode = page_soup.findAll('td', attrs={'class':'title'})

	for num in range(0, len(total_episode)):
		episode_num = total_episode[num].find('a').contents[0]
		episode_num_list.append(episode_num)

		date = total_episode[num].next_sibling.next_sibling.next_sibling.next_sibling.contents[0]
		episode_date_list.append(date)

		ratings = total_episode[num].next_sibling.next_sibling.find('strong').contents[0]
		episode_ratings_list.append(ratings)

		link = total_episode[num].find('a').get('href')
		link = 'http://comic.naver.com' + link
		episode_link_list.append(link)

	print "%i is done" %page

	sleep(0.5)



book = Workbook()

sheet = book.add_sheet("ms")

sheet.write(0,0,'num')
sheet.write(0,1,'episode_title')
sheet.write(0,2,'date')
sheet.write(0,3,'ratings')
sheet.write(0,4,'link')

for x in range(0,len(episode_num_list)):
	sheet.write(x+1,0,latest_episode_num-x)
	sheet.write(x+1,1,episode_num_list[x])
	sheet.write(x+1,2,episode_date_list[x])
	sheet.write(x+1,3,float(episode_ratings_list[x]))
	sheet.write(x+1,4,episode_link_list[x])

book.save('ms.xls')





#episode_num // title // date // ratings // rating_num // link

{% endhighlight %}