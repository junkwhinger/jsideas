---
layout:     post
title:      "Parsing Instagram using R"
date:       2015-01-17 14:50:00
author:     "Jun"
categories: "R"
header-img: "img/post-bg-05.jpg"
---
<h2 class="section-heading">Parsing Instagram using R</h2>

<p> <a href="http://jsideas.net/r/2015/01/13/kakao-story-parser/">My previous blogpost</a> was about how to extract data from Kakaostory. I manipluated its R code a little bit to parse html files from Instagram.</p>

<h2 class="section-heading">How to use</h2>
<p> 1. Set the right R environment. Go and download R on your PC or MAC, and install XML and CSS libraries. <br />
	> install.packages("XML")<br />
	> install.packages("CSS")</p>

<p> 2. Ready the data. This bit is pretty much the same as Kakao Story. Go to Instagram and find the account you want to crawl. Scroll down until you reach the extent you want to crawl. Right click the screen and select "inspect element"(on Chrome). For a demonstation, I picked Cara Delevinge's instagram, @caradelevingne. </p>

![@cara account](/assets/insta_cara.png)
<span class="caption text-muted">Cara Delevinge, a famous model</span>

<p> I scrolled down to the first of December, 2014, and hit "inspect element". You'll see loads of "div data-reacted="..."s. That is a container for each post. Go up until you find a div with a class called "PhotoGrid". Right click and copy the div, and paste it on an empty text editor. Save it as an html file. This is what you'll get. </p>

![@cara account](/assets/insta_element.png)
<span class="caption text-muted">Scroll down until you get to the right time period.</span>

<p> 3. Download the R script file(KSParser.R). You can download it from my <a href="https://github.com/junkwhinger/Instagram-Parser">github</a> page. </p>

<p> 4. Place the script file in the same directory as the html file. </p>

<p> 5. Open R or RStudio, and run the script. <br />
	> source("ISParser.R")</p>

<p>type in the name of the file, which in my case, 'cara' from cara.html.</p>

<p> 6. In about a few seconds, a new file will be created with a prefix "done_". Open the file on Excel and see if everything's okay</p>

![@cara account](/assets/insta_result.png)
<span class="caption text-muted">Voila! It's done and dusted.</span>

<p> Extraction completed! 160 post and their attributes are all completely donwloaded. </p>



<p> Here's my R script code </p>
{% highlight r %}

library("XML")
library("xlsx")
library("CSS")

name_of_file <- readline("type in the name of html file without .html: ")

file_name <- paste(name_of_file, ".html", sep="")

i_doc <- htmlParse(file_name, encoding="UTF-8")

i_root <- xmlRoot(i_doc)

time <- cssApply(i_doc, ".pgmiDateHeader", cssCharacter)

url <- cssApply(i_doc, ".pgmiImageLink", cssLink)

re_url <- paste("http://www.instagram.com", url, sep="")

like_and_comment <- xpathSApply(i_root, "//div[@class='PhotoGridMediaItem']", xmlGetAttr, "aria-label")

like_and_comment_table <- read.table(textConnection(like_and_comment))

likes <- like_and_comment_table[,1]

comments <- like_and_comment_table[,3]

final <- cbind(time, likes, comments, re_url)

renamed_file <- paste("done", name_of_file, sep="_")

xlsx_file <- paste(renamed_file, ".xlsx", sep="")

write.xlsx(final, xlsx_file)

{% endhighlight %}
