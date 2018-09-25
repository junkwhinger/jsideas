---
layout:     post
title:      "How to crawl Kakao Story using R"
date:       2015-01-06 22:10:00
author:     "Jun"
header-img: "img/post-bg-05.jpg"
tags: [r, web crawling]

---
<h2 class="section-heading">How to crawl Kakao Story using R</h2>

<p><a href="https://story.kakao.com">Kakao Story</a> is a social networking service provided by DaumKakao, and it's been loved by many young Korean people. It has its own dev centre and open APIs, but as I'm not skilled enough to use them, I used R and an html file to crawl data from it. It's really simple and primitive, so if you are already fully familiar with crawling data from Kakao Story, this posting wouldn't be much of a help.</p>

<p>The basic idea is to download a html file and parse it with R. It's really important to have a closer look at the html tag structure and check where your data lies.</p>

<p> Here's how I did.<br> 
  1. Ready the file<br>
Go to the page you want to crawl. And copy and save the div you want to crawl, and save it as an html file.<br>
<br>

2. Read in the file on RStudio.<br>
> library("XML") // importing XML library to use "htmlParse".
> doc <- readLines(“html_file.html")<br>
You’ll see an Warning message like ‘incomplete final line found..” but it’s okay.<br>
<br>

3. parse the doc. If in Korean, encode with UTF-8<br>
> doc_2 <- htmlParse(doc, encoding="UTF-8")<br>
<br>

4. find the div class or id names that you want to crawl.<br>
For example, likeCount is in the div class called “_likeCount”.<br>
<br>

5. crawl using CSS library.<br>
> install.packages(“CSS”)<br>
> library(“CSS”)<br>
> likeCounts <- cssApply(doc_2, “._likeCount”, cssCharacter)<br>
You have to put a period(“.”) before the div name. If it’s a string of characters you are going to crawl then the third parameter is “cssCharacter”. Find more about this <a href="http://cran.r-project.org/web/packages/CSS/vignettes/CSS.pdf">here</a>. This CSS library is awesome to say the least.<br>
Crawl other info by using the same method.<br>
like.. > urls <- cssApply(doc, ".player>a", cssLink)<br>
<br>

6. if the data you’re crawling is missing at some points, then it’s better to use cssApplyInNodeSet to put NAs to make a complete set with other data.<br>
<br>

7. use cbind command to make a matrix<br>
> my_matrix <- cbind(dates, likeCounts, contents)<br>
<br>

8. export to Excel<br>
To do so, download “xlsx” library and load it.<br>
I tried “write.csv” command, but for some encoding issues occurred and Excel couldn’t load the file properly.<br>
> install.packages(“xlsx”)<br>
> library(“xlsx”)<br>
> write.xlsx(my_matrix, “name_of_your_file.xlsx”)<br>
<br>

9. Go to Excel and open it!<br>
Voila! It’s done!<br></p>




