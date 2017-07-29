---
layout:     post
title:      "AlphaGo vs. Lee Sedol: Levenshtein Distance"
date:       2016-07-07 00:00:00
author:     "Jun"
categories: "Python"
image: /assets/alphago/header.jpg
tags: featured
---
# AlphaGo vs. Lee Sedol: time spent pattern comparison 
 
In my latest blogposts on AlphaGo vs. Lee Sedol, I uploaded some graphs that
clearly showed how Lee Sedol and AlphaGo used their time differently in the
Google DeepMind Challenge. Recently, I have come across some really interesting
articles and lecture videos on distance measures including LevenShtein distance,
and thought it would be interesting to see the patterns between human and
machine! (Even though we already know that AlphaGo is a man-made machine). 
 
### Thinking Time Remaining
![Man vs. Machine!](/assets/alphago/total_remaining_edited.png)

# Data Preprocessing 
 
The following custom function 'preprocessing' 1) reads the raw csv file, 2)
calculates time spent between each turn index. It's basically the same code (yet
more concise) from my previous ipython notebook uploaded <a href="https://github.com/junkwhinger/AlphaGo_raw">here</a>. 

**In [115]:**

{% highlight python %}
#loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
{% endhighlight %}

**In [117]:**

{% highlight python %}
#custom function to calculte timespent in seconds
def time_func(x):
    k = x.split(":")
    hour = int(k[0])
    minute = int(k[1])
    second = int(k[2])
    ts = hour * 3600 + minute * 60 + second
    return ts
{% endhighlight %}

**In [119]:**

{% highlight python %}
#custom function to read and preprocess raw data files
def preprocessing(raw_data):
    data = pd.read_csv(raw_data, index_col='turn_index')
    df_alpha = data.AlphaGo.dropna().reset_index()
    df_lee = data.Lee_Sedol.dropna().reset_index()
    
    
    df_alpha['AlphaGo_ts'] = df_alpha.AlphaGo.apply(lambda x: time_func(x))
    df_lee['Lee_Sedol_ts'] = df_lee.Lee_Sedol.apply(lambda x: time_func(x))

    df_lee['Lee_Sedol_lag'] = df_lee.Lee_Sedol_ts.shift(1)
    df_alpha['AlphaGo_lag'] = df_alpha.AlphaGo_ts.shift(1)
    
    df_lee['Lee_Sedol_tt'] = df_lee.Lee_Sedol_lag - df_lee.Lee_Sedol_ts
    df_alpha['AlphaGo_tt'] = df_alpha.AlphaGo_lag - df_alpha.AlphaGo_ts
    
    df_lee_result = df_lee[['turn_index','Lee_Sedol_tt']][1:]
    df_alpha_result = df_alpha[['turn_index','AlphaGo_tt']][1:]
    
    df_lee_ott = data.Lee_Sedol_ott.dropna().reset_index()
    df_lee_ott['Lee_Sedol_tt'] = df_lee_ott.Lee_Sedol_ott.apply(lambda x: time_func(x))
    df_lee_ott_result = df_lee_ott[['turn_index', 'Lee_Sedol_tt']]
    
    df_alpha_ott = data.AlphaGo_ott.dropna().reset_index()
    df_alpha_ott['AlphaGo_tt'] = df_alpha_ott.AlphaGo_ott.apply(lambda x: time_func(x))
    df_alpha_ott_result = df_alpha_ott[['turn_index', 'AlphaGo_tt']]
    
    df_alpha_final = df_alpha_result.append(df_alpha_ott_result).reset_index(drop=True)
    df_lee_final = df_lee_result.append(df_lee_ott_result).reset_index(drop=True)
    
    result_df = df_alpha_final.merge(df_lee_final, on='turn_index')
    result_df.index = result_df.turn_index
    result_df.drop('turn_index', axis=1, inplace=True)

    return result_df
{% endhighlight %}

**In [120]:**

{% highlight python %}
#file paths
file1 = '/users/jun/python/alphago/first_game.csv'
file2 = '/users/jun/python/alphago/second_game.csv'
file3 = '/users/jun/python/alphago/third_game.csv'
file4 = '/users/jun/python/alphago/fourth_game.csv'
file5 = '/users/jun/python/alphago/fifth_game.csv'
{% endhighlight %}

**In [121]:**

{% highlight python %}
#preprocessing in one go!
g1 = preprocessing(file1)
g2 = preprocessing(file2)
g3 = preprocessing(file3)
g4 = preprocessing(file4)
g5 = preprocessing(file5)
{% endhighlight %}
 
# Stringify(?) the thinking time 
 
To make it easier for my laptop to guess how different AlphaGo is in Game 1 from
Game 2, I'm going to transform the float data to strings according to their time
length like the following custom function. 

**In [122]:**

{% highlight python %}
def string_classifier(time_spent):
    if time_spent <= 10.0:
        return 'A'
    elif time_spent <= 40.0:
        return 'B'
    elif time_spent <= 120.0:
        return 'C'
    elif time_spent <= 300.0:
        return 'D'
    else:
        return 'E'
{% endhighlight %}
 
It really depends on how you want to design your stringified thinking time. I
designed it to cluster hasty moves, normal moves and moves with prolonged
thoughts. Pandas 'map' function makes it super easy to apply this to every value
in the DataFrames! 

**In [125]:**

{% highlight python %}
g1_A = g1.AlphaGo_tt.map(string_classifier)
g2_A = g2.AlphaGo_tt.map(string_classifier)
g3_A = g3.AlphaGo_tt.map(string_classifier)
g4_A = g4.AlphaGo_tt.map(string_classifier)
g5_A = g5.AlphaGo_tt.map(string_classifier)
{% endhighlight %}

**In [126]:**

{% highlight python %}
g1_L = g1.Lee_Sedol_tt.map(string_classifier)
g2_L = g2.Lee_Sedol_tt.map(string_classifier)
g3_L = g3.Lee_Sedol_tt.map(string_classifier)
g4_L = g4.Lee_Sedol_tt.map(string_classifier)
g5_L = g5.Lee_Sedol_tt.map(string_classifier)
{% endhighlight %}
 
# Levenshtein Distance 
 
Alright, now that the data is good to go, let's move on to the distance measure.
Levenshtein Distance measure is one of the distance algorithms we can use to
tell how different two given strings are. Let's say you are given 'Dorian' and
'Durians'. Dorian is my pet cat, and Durians is the plural form of my least
favourite fruit. Anyway, LevenShtein Distance measure gets higher as you delete,
insert or replace to make one to be exactly the same as the other. So in our
example, we need to change o->u (1 point), delete s at the end (1 point), so
their Levenshtein distance is 2. 


Levenshtein distance algorithm from <a href="https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance">wikibooks</a>

**In [156]:**
{% highlight python %}
def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]
{% endhighlight %}

**In [157]:**

{% highlight python %}
first_word = 'Dorian'
second_word = 'Durians'
l = levenshtein(first_word, second_word)
print ("Levenshtein distance between {} and {} is {}.".format(first_word, second_word, l))
{% endhighlight %}

    Levenshtein distance between Dorian and Durians is 2.

 
# Levenshtein Distance on AlphaGo and Lee Sedol 
 
Okay! We already know from the my previous visualisations that AlphaGo time
spending habit was significantly different from Lee Sedol. So it's not code
worthy to calculate the distance between the two. What about themselves?
AlphaGo's thinking time didn't vary much but how about Lee Sedol? In the second
game he took a highly defensive position, whereas in the third round he started
off with offensive position. Did his time spending habit change as Google
DeepMind Challenge proceeded? (It's interesting that the same logic is widely
used in bot-detection practices in gaming industry and fraud detection
algorithms in finance trades.) 
 
I've made the following custom function to 1) prepare 5 time logs of AlphaGo and
Lee Sedol, 2) calcaulate Levenshtein distance between their games 3) and present
pretty heatmaps side by side. The third parameter 'length' means the threshold
to slice the time log in order to make more comparisons. It's a wide-spread
practice to slice time frames in order to make the difference in distance more
dramatic. The longer the logs are, the more likely they become heterogeneous. 

**In [246]:**

{% highlight python %}
def levheatmap(alist, blist, length):
    al = []
    bl = []
    if length != 'full':
        
        for a in alist:
            a = a[:length]
            al.append(a)
        
        for b in blist:
            b = b[:length]
            bl.append(b)
    else:
        al = alist
        bl = blist
            
    array_A = np.zeros([len(al),len(al)])
    for i in range(0, len(al)):
        for j in range(0, len(al)):
            array_A[i][j] = levenshtein(al[i], al[j])
            
    array_B = np.zeros([len(bl),len(bl)])
    for i in range(0, len(bl)):
        for j in range(0, len(bl)):
            array_B[i][j] = levenshtein(bl[i], bl[j])
            
    maxval = max(array_A.max(), array_B.max())
            
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1_title = 'AlphaGo: leven distance / string length: {}'.format(length)
    ax2_title = 'Lee Sedol: leven distance / string length: {}'.format(length)

    sns.heatmap(array_A, alpha=0.75, vmax=maxval, cmap='RdBu', linewidths=.5, cbar=False, annot=True, xticklabels=['Game 1','Game 2','Game 3','Game 4','Game 5'], yticklabels=['Game 1','Game 2','Game 3','Game 4','Game 5'], ax=ax1)
    
    sns.heatmap(array_B, alpha=0.75, vmax=maxval, cmap='RdBu', linewidths=.5, cbar=False, annot=True, xticklabels=['Game 1','Game 2','Game 3','Game 4','Game 5'], yticklabels=['Game 1','Game 2','Game 3','Game 4','Game 5'], ax=ax2)
    ax1.set_title(ax1_title, y=1.08)
    ax2.set_title(ax2_title, y=1.08)
       
    ax1.xaxis.tick_top()
    ax2.xaxis.tick_top()

    plt.show()
{% endhighlight %}

**In [247]:**

{% highlight python %}
list_A = [g1_A, g2_A, g3_A, g4_A, g5_A]
list_L = [g1_L, g2_L, g3_L, g4_L, g5_L]
{% endhighlight %}
 
# Heatmaps! 
 
It turns out (as expected) that Lee Sedol's games were much more diverse than
AlphaGo's games. The homogeneity of AlphaGo moves is not surprising, but its
Game 5 was quite different from all the other. Lee Sedol's Levenshtein distances
are all at least higher than 55. Like AlphaGo, the odd one out in his games was
game number 5. 

**In [251]:**

{% highlight python %}
#Here we go. Full string comparison. The bluer the cell is, the higher the distance is.
levheatmap(list_A, list_L, 'full')
{% endhighlight %}

 
![string length: full]({{ BASE_PATH }}/assets/distance_analysis_files/distance_analysis_26_0.png) 

 
Well, to my surprise, AlphaGo's LevenShtein distance was a little bit higher
than I expected. If so, how can we detect if it's AlphaGo playing with a
disguised identity on an online Go match? To bring justice, we need to slice the
time log. Let's focus on the very first 10 strings. 

**In [256]:**

{% highlight python %}
#Let's focus on the very first 10 strings. Lee Sedol's style still varies, while AlphaGo keeps its style.
levheatmap(list_A, list_L, 10)
{% endhighlight %}

 
![string length: 10]({{ BASE_PATH }}/assets/distance_analysis_files/distance_analysis_28_0.png) 


**In [253]:**

{% highlight python %}
#string length 30. 
levheatmap(list_A, list_L, 30)
{% endhighlight %}

 
![string length: 30]({{ BASE_PATH }}/assets/distance_analysis_files/distance_analysis_29_0.png) 


**In [254]:**

{% highlight python %}
#string length 60.
levheatmap(list_A, list_L, 60)
{% endhighlight %}

 
![string length: 60]({{ BASE_PATH }}/assets/distance_analysis_files/distance_analysis_30_0.png) 


**In [255]:**

{% highlight python %}
#string length 90.
levheatmap(list_A, list_L, 90)
{% endhighlight %}

 
![string length: 90]({{ BASE_PATH }}/assets/distance_analysis_files/distance_analysis_31_0.png) 

 
Now we got it. Just by looking at the first 10 time logs, we can successfully
tell AlphaGo from Lee Sedol! There are more distance measures like hamming
distance, Jaro–Winkler distance and so on! I'll cover more of these in my future
blogposts. 

**In [None]:**

{% highlight python %}

{% endhighlight %}
