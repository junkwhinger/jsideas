---
layout:     post
title:      "Parsing Kakao Story using R"
date:       2015-01-13 12:50:00
author:     "Jun"
tags: [r, web crawling]
---
<h2 class="section-heading">Parsing Kakao Story using R</h2>

<p> This blog post is a follow up of my previous post about crawling data from Kakao Story. As I mentioned previously, I extracted data from html sources from Kakao Story, not using its API. By using R's XML and CSS libraries, I made this simple R script by which you can extract published time, content, likes, comments, shares and urls. </p>

<h2 class="section-heading">How to use</h2>
<p> 1. Set the right R environment. Go and download R on your PC or MAC, and install XML and CSS libraries. <br />
	> install.packages("XML")<br />
	> install.packages("CSS")</p>

<p> 2. Ready the data. Go to Kakao Story and find the account you want to crawl. Scroll down until you reach the extent you want to crawl. Right click the screen and select "inspect element"(on Chrome). For a demonstation, I chose SK Telecom's offical Kakao Story account. </p>

![SKT official account](/assets/materials/20150113/kakao_skt.png)
<span class="caption text-muted">SKT's offical Kakao Story account</span>

![SKT official account](/assets/materials/20150113/kakao_element.png)
<span class="caption text-muted">Scroll down until you get to the right time period.</span>

<p> I scrolled down to the first of November, 2014, and hit "inspect element". You'll see loads of "<div class="section _activity"...>"s. That is a container for each post. Go up until you find a div with a class called "_listContainer". Right click and copy the div, and paste it on an empty text editor. Save it as an html file. This is what you'll get. </p>

![SKT official account](/assets/materials/20150113/kakao_html.png)
<span class="caption text-muted">Scroll down until you get to the right time period.</span>

<p> 3. Download the R script file(KSParser.R). You can download it from my <a href="https://github.com/junkwhinger/kakaoStoryParser">github</a> page. </p>

<p> 4. Place the script file in the same directory as the html file. </p>

<p> 5. Open R or RStudio, and run the script. <br />
	> source("KSParser.R")</p>

![SKT official account](/assets/materials/20150113/kakao_rstudio.png)
<span class="caption text-muted">KSParser.R on RStudio</span>

<p>type in the name of the file, which in my case, 'skt' from skt.html.</p>

<p> 6. In about a few seconds, a new file will be created with a prefix "done_". Open the file on Excel and see if everything's okay</p>

![SKT official account](/assets/materials/20150113/kakao_result.png)
<span class="caption text-muted">Voila! It's done and dusted.</span>

<p> Extraction completed! 130 post and their attributes are all completely donwloaded. </p>



<p> Here's my R script code </p>
{% highlight r %}

library("XML")
library("xlsx")
library("CSS")

name_of_file <- readline("type in the name of html file without .html: ")

file_name <- paste(name_of_file, ".html", sep="")

k_doc <- htmlParse(file_name, encoding="UTF-8")

k_root <- xmlRoot(k_doc)

time <- xpathSApply(k_doc, "//a[@class='time _linkPost']", xmlValue)

content <- cssApplyInNodeSet(k_doc, ".fd_cont", ".txt_wrap", cssCharacter)

likes <-cssApply(k_doc, "._likeCount", cssNumeric)

comments <-cssApply(k_doc, "._commentCount", cssNumeric)

shares <-cssApply(k_doc, "._shareCount", cssNumeric)

link <- xpathSApply(k_root, "//a[@class='time _linkPost']", xmlGetAttr, "href")

link_pasted <- paste("https://story.kakao.com", link, sep="")

final <- cbind(time, likes, comments, shares, link_pasted, content)

renamed_file <- paste("done", name_of_file, sep="_")

xlsx_file <- paste(renamed_file, ".xlsx", sep="")

write.xlsx(final, xlsx_file)

{% endhighlight %}
