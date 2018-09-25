---
layout:     post
title:      "Exporting a csv file to Excel using R"
date:       2015-01-10 12:50:00
author:     "Jun"
img: "excel_right.png"
tags: [r]

---
## .csv to .xlsx using R

<p>In many occasions, you don't really have to use R to open .csv files on Excel. You can just open a csv file on Excel, and it works!</p>

<p> You can do the same by using several commands of R, too. And it magically solves some encoding issues. I was trying to open a csv file that has Korean texts and encoded in UTF-8 on Excel(ANSI) in Windows. Notepad ++ did change the encoding type, but the file didn't look neat enough on Excel... like this</p>

![this doesn't look alright](/assets/materials/20150110/excel_wrong.png)
<span class="caption text-muted">My twitter activities from analytics.twitter.com</span>

<h2 class="section-heading">xlsx library on R</h2>

<p> By using "xlsx" library on R, you can easily turn a .csv file to .xlsx file. <br />

1. open R and import "xlsx" library. <br />
(If not installed) > install.packages("xlsx")<br />
> library("xlsx")<br />
<br />
2. read in the csv file and designate it to "doc" variable.<br />
> doc <- read.csv("the_csv_file_you_want.csv", encoding="UTF-8")<br />
<br />
3. write an xlsx file.<br />
> write.xlsx(doc, "new_name_of_the_file.xlsx")<br />
<br />
4. open the file on Excel!<br />
it works! <br />
</p>

![this look alright](/assets/materials/20150110/excel_right.png)
<span class="caption text-muted">Now it looks alright!</span>



