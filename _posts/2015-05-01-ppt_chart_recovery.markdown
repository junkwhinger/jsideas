---
layout:     post
title:      "How to recover data from disconnected ppt charts"
date:       2015-05-01 23:50:00
author:     "Jun"
categories: "R"
header-img: "img/post-bg-05.jpg"
tags: [r]
---

<h2 class="section-heading">How to recover data from disconnected ppt charts</h2>

<p>For those who handle lots of powerpoint files at work, few things are more annoying than ppt charts that have lost links with their original data set. Luckily you can still figure out the original data by hovering over the points, but what would you do with a chart that has hundreds, if not thousands of data points?</p>

<p>A few days ago I was in that situation where I had a disconnected grouped bar chart. It had 4 series times 7 categories, so 28 data points. This was enough to irritate me, and I found <a href="http://blog.magicbeanlab.com/data-viz/retrieve-data-from-powerpoint-charts-when-linked-file-not-available/">this link</a>.</p> 

<p>This guy’s blog explains the solution well, but I thought I could do something with R. So here’s what I’ve come up with. If you have R installed in your mac, you can easily do it.</p>

<p>1. let’s get the ingredients ready.</p>
![pic1](/assets/materials/20150502/pic1.png)
<p>Here I have a ppt file with 4 different types of charts; a line chart, a pie chart, a bar chart and a group bar chart.</p>

<p>2. save the ppt file, and change it file type from pptx to zip.</p>
![pic2](/assets/materials/20150502/pic2.png)
<p>It sounds pretty weird, but your computer will allow that to happen.</p>

<p>3. unzip the newly modified zip file.</p>
![pic3](/assets/materials/20150502/pic3.png)
<p>In the folder, you’ll see a bunch of folders and files.. and among them, you’ll find [charts] folder inside [ppt] folder. That’s where our charts are. 4 xml files there!</p>

<p>4. put them in a folder with ppt_chart_recovery_f.R (again, you can download it from <a href="https://github.com/junkwhinger/ppt_chart_recovery">here</a>). Make sure you have no other xml files in the folder, as my R script is designed to parse all the xml files it can find in its folder. Like this.</p>
![pic4](/assets/materials/20150502/pic4.png)

<p>5. open terminal and go to the test folder. as my [test] folder is in the [desktop] folder, I just need to type “cd desktop/test"</p>

<p>6. run the r code to get the result. type “r” to activate r, and type “source(“ppt_chart_recovery_f.R”). Within a few seconds, the job is done.</p>
![pic5](/assets/materials/20150502/pic5.png)
<p>It shows some messages that go “Namespace prefix c…” but you can ignore it.</p>

<p>7. check out the results.</p>
![pic6](/assets/materials/20150502/pic6.png)
<p>Voila! four new csv files! click and hit space to check out what they look like.</p>
![pic7](/assets/materials/20150502/pic7.png)
<p>Job done!</p>

{% highlight r %}

library("XML")

#loading files
current_files <- dir()
file_vec <- vector()
for (i in 1:length(current_files)) {
  file <- current_files[i]
  if (grepl(".xml", file)) {
    if (!(file %in% file_vec)) {
      file_vec <- c(file_vec, file)
    }
  }
}
ser_list <- list()
for (file in 1:length(file_vec)) {
  doc <- xmlTreeParse(file_vec[file])
  root <- xmlRoot(doc)
  #chart type
  chart_type <- names(doc$doc$children$chartSpace["chart"]$chart["plotArea"]$plotArea[2])
  
  #fetching series
  ser = xpathApply(root, "//c:ser/c:tx/c:strRef/c:strCache/c:pt/c:v")
  
  ser_vec = vector()
  for (i in 1:length(ser)) {
    ser_item <- toString.XMLNode(ser[[i]])
    ser_item <- gsub("<.*?>","", ser_item)
    ser_item <- gsub(" $","", ser_item, perl=T)
    ser_vec <- c(ser_vec, ser_item)
    
  }
  
  #fetching categories
  cat0 <- xpathApply(root, "//c:cat")
  cat_val <- names(cat0[[1]])
  if (cat_val == "c:strRef") {
    cat <- xpathApply(root, "//c:cat/c:strRef/c:strCache/c:pt/c:v")
  } else {
    cat <- xpathApply(root, "//c:cat/c:numRef/c:numCache/c:pt/c:v") 
  }
  
  
  
  cat_vec = vector()
  for (i in 1:length(cat)) {
    cat_item <- toString.XMLNode(cat[[i]])
    cat_item <- gsub("<.*?>","", cat_item)
    cat_item <- gsub(" $","", cat_item, perl=T)
    if (!(cat_item %in% cat_vec)) {
      cat_vec <- c(cat_vec, cat_item)
    } 
  }
  
  #fetching values
  value = xpathApply(root, "//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v")
  
  val_vec = vector()
  for (i in 1:length(value)) {
    value_item <- toString.XMLNode(value[[i]])
    value_item <- gsub("<.*?>","", value_item)
    value_item <- gsub(" $","", value_item, perl=T)
    value_item <- as.numeric(value_item)
    val_vec <- c(val_vec, value_item)
  }
  
  #creating matrix
  final <- matrix(val_vec,nrow=length(cat_vec),ncol=length(ser_vec))
  
  rownames(final) <- cat_vec
  colnames(final) <- ser_vec
  
  #export as a csv file
  filename <- paste("done_", "chart", file, sep="")
  filename <- paste(filename, ".csv", sep="")
  write.csv(final, filename)
  
}

{% endhighlight %}