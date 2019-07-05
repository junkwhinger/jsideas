---
layout:     post
title:      "Vigenere Cypher"
date:       2015-05-05 22:30:00
author:     "Jun"
tags: [python]
---

<h2 class="section-heading">Vigenere Cypher</h2>
<p>Found a random python tutorial website about encryption. Thought I could make a python program that uses vigenere cypher, so I tried as below. It seemed to me at first that it was nearly impossible to decrypt it, but it is indeed possible!</p>
<iframe width="560" height="315" src="https://www.youtube.com/embed/P4z3jAOzT9I" frameborder="0" allowfullscreen></iframe>


{% highlight python %}

import string

uppercase_string_list = list(string.ascii_uppercase)

def vigenere_cypher(text, keyword):
  
  result_list = list()

  #getting the text and keyword length
  text_length = len(text)
  keyword_length = len(keyword)

  #setting the new keyword which is as long as the text
  text_list = list(text)
  keyword_list = list(keyword)
  new_keyword_list = list()
  for i in range(0,text_length):
    i = i % keyword_length
    char = keyword_list[i]
    new_keyword_list.append(char)
  

  #row & col
  for i in range(0, len(new_keyword_list)):
    #row
    row_char = new_keyword_list[i]
    print row_char
    rowidx = uppercase_string_list.index(row_char)
    print rowidx
    new_row = uppercase_string_list[rowidx:] + uppercase_string_list[:rowidx]
    print new_row

    #col
    col_char = text_list[i]
    colidx = uppercase_string_list.index(col_char)
    print colidx
    #answer
    answer = new_row[colidx]
    print answer

    result_list.append(answer)

  result = "".join(result_list)
  return result



print vigenere_cypher("YOURSECRETISSAFE", "MYSTERY")

{% endhighlight %}