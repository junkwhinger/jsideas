---
layout:     post
title:      "PDF Spliter"
date:       2015-05-16 00:30:00
author:     "Jun"
tags: [python]
---

<h2 class="section-heading">PDf Spliter</h2>
<p>There are many ways you can chop bulky PDF files into chewable pieces, and python is not an exception. </p>
<p>Run the python code presented below and type the name of the pdf file you wish to chop. Enter the start and end page number. Within a few seconds it's done and dusted. </p>

{% highlight python %}

from PyPDF2 import PdfFileWriter, PdfFileReader 
filename = str(raw_input("which file do you want to split? "))
file_to_parse = filename + ".pdf"
infile = PdfFileReader(open(file_to_parse, 'rb'))

startPage = int(raw_input("start page? ")) -1
endPage = int(raw_input("end page "))
outfile = PdfFileWriter()
for page in range(startPage, endPage):
    p = infile.getPage(page)
    outfile.addPage(p)
with open(filename+'-%03d-%03d.pdf' % (startPage+1, endPage), 'wb') as f:
    outfile.write(f)

{% endhighlight %}