---
layout:     post
title:      "PDF Extractor in python"
date:       2015-05-16 11:30:00
author:     "Jun"
tags: [python]
---

<h2 class="section-heading">PDf Extractor in python</h2>
<p>Reading a backlog of articles in English could be mind-boggling at times. Instead of spending a lengthy amount of hours, I came up with a brilliant idea, which is to extract the most important-looking sentences from academic gobbledygook.</p>

<p>The idea behind the code below is simple. In well-written academic papers, authors tend to put “Thus, “, “Therefore, “, “In sum, “ to summarise their arguments. “Yet, “, “However, “ are the common phrases to introduce counter-intuitive or contradicting facts and opinions. This code is designed to find the sentences that begin with these keywords. (I decided not to include keywords in lowercase for now as the result gets messier due to the uncontrollable variation of English language.)</p>

<p>After extracting the list of important sentences and removing invalid unicode characters in them, this code accesses my Evernote account, open the designated notebook, creates a note with the following information:</p>

<ul>
	<li>note title: “Summary from” + file name</li>
	<li>note content: an unordered list of important sentences</li>
	<li>note tag: the top 5 most frequently used words excluding “a, the, in, and, etc"</li>
</ul>

<p>The code I wrote below contains a code that runs <a href="http://www.pdf2txt.com/">“pdf2txt.py”</a>. (I couldn’t get my head around parsing lines of PDF files yet, so this could be a lazy yet highly amateurish approach.) The code below WON't work unless you have pdf2txt.py installed on your machine.</p>

<p>Here’s a simple example using pdf_extractor.py.</p>

<p>1) place a pdf file in the same folder as pdf_extractor.py.</br>
The sample pdf file I used for this blogpost is “Impact_of_the_social_sciences.pdf” which was available on the internet (hope this is not a case of copyright infringement). </p>

<p>2) get your own Evernote developer token and put it in the code.</p>
<p>Follow the <a href="https://dev.evernote.com/doc/start/python.php">instructions provided by Evernote</a>.</p>

<p>3) run the python code and provide the name of the pdf file without “.pdf”.</p>
<p>Like this.</p>
![pic1](/assets/materials/20150516/pic1.png)

<p>4) done. check your folder in Evernote.</br>
Voila! Does this make sense to you? It can’t be perfect but I think it’s a good skimming of the article. </p>
![pic2](/assets/materials/20150516/pic2.png)

<p>Here are the thankful blogposts and websites that helped me a lot.</p>
<ul>
	<li><a href="https://maxharp3r.wordpress.com/2008/05/15/pythons-minidom-xml-and-illegal-unicode-characters/">removing invalid unicode chars</a></li>
	<li><a href="http://www.uk.sagepub.com/upm-data/59598_Bastow__Impact_of_the_social_sciences.pdf">sample pdf file</a></li>
	<li><a href="http://www.pdf2txt.com/">pdf2txt.com</a></li>
</ul>

<p>Thanks for reading!</p>
{% highlight python %}

import os
import re
import evernote.edam.userstore.constants as UserStoreConstants
import evernote.edam.type.ttypes as Types
from collections import defaultdict
import operator
import xml.dom.minidom
from evernote.api.client import EvernoteClient

# invalid unicode chars
RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                 u'|' + \
                 u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                  (unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                   unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                   unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff))


# load the file to parse
file_to_extract = str(raw_input("Which file? "))

# command to execute pdf2txt.py
command = "pdf2txt.py -O summary -o summary/"+file_to_extract+".text  -t text "+file_to_extract+".pdf"

# execute the command to retrieve the text from the pdf file
os.system(command)

# read the text file
file_path = 'summary/' + file_to_extract + ".text"
file = open(file_path)

t = file.read()

# keywords to extract important sentences
key_set = ['Thus, ',
	'Hence, ', 
	'Therefore, ', 
	'In conclusion ', 
	'To summarize, ', 
	'Yet, ', 
	'First, ', 
	'Secondly, ', 
	'Thirdly, ', 
	'Lastly, ', 
	'However, ', 
	'As a result, ', 
	'Furthermore, ', 
	'Moreover, ', 
	'In sum', 
	'Finally, ',
	'As a result ',
	'So,']

# extract sentences and clean them
sentence_list = list()
for key in key_set:
	for m in re.finditer(key, t):
		sentence = list()
		counter = 0
		text = list()
		while t[m.start() + counter] != ".":
			char = t[m.start() + counter]
			text.append(char)
			counter = counter + 1
		text.append(".")
		text = "".join(text)
		text = text.replace('\n', " ")
		text = text.replace('- ',"")
		text = re.sub(RE_XML_ILLEGAL, "?", text)
		sentence_list.append(text)

print 'there are %i sentences extracted.' % len(sentence_list)

# top 5 most frequent word list to use as tags
d = defaultdict(int)

for word in t.split():
	d[word] += 1

del_key_list = ['I','i','o', 'you', 'your', 'We', 'we', 'You', 'the', 'The', 'of', 'Of', 'and', 'or', 'in', 'on', 'On', 'In', 'to', 'To', 'is', 'are', 'a', 'that', 'That', 'Those', 'those', 'by', 'until', 'has', 'from', 'it', 'be', 'at', 'This', 'this', 'under', 'own', 'as', 'As', 'have', 'between', 'an', 'with', 'not', 'its', 'over', 'also', 'more','for', 'their', 'Their' ]

# remove common keywords from the dictionary
for key in del_key_list:
	if key in d:
		del d[key]

# sort dictionary items from the most to the least frequent words
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

# create a tag list of top 5 most frequent keywords
tag_list = list()
for i in range(0,5):
	tag_list.append(sorted_d[i][0])

# access evernote and create a note
if len(sentence_list) > 0:

	auth_token = "Enter_your_auth_token_here"

	client = EvernoteClient(token=auth_token, sandbox=False)

	user_store = client.get_user_store()

	version_ok = user_store.checkVersion(
	    "Evernote EDAMTest (Python)",
	    UserStoreConstants.EDAM_VERSION_MAJOR,
	    UserStoreConstants.EDAM_VERSION_MINOR
	)
	print "Is my Evernote API version up to date? ", str(version_ok)
	print ""
	if not version_ok:
	    exit(1)

	note_store = client.get_note_store()

	# List all of the notebooks in the user's account
	notebook_list = list()
	notebooks = note_store.listNotebooks()
	print "Found ", len(notebooks), " notebooks:"
	for notebook in notebooks:
	    notebook_list.append(notebook.name)

	# Create a summary folder if it doesn't exist
	if "PDF_Summary" not in notebook_list:
		notebook = Types.Notebook()
		notebook.name = "PDF_Summary"
		notebook = note_store.createNotebook(notebook)

	# Fetch 'PDF_Summary's guid'
	notebook_guid = 'dummy'
	for notebook in notebooks:
		if notebook.name == "PDF_Summary":
			notebook_guid = notebook.guid
			print notebook.guid

	# create a new note
	note = Types.Note()
	note.notebookGuid = notebook_guid
	note.title = "Summary from " + file_to_extract

	note.content = '<?xml version="1.0" encoding="UTF-8"?>'
	note.content += '<!DOCTYPE en-note SYSTEM ' \
	    '"http://xml.evernote.com/pub/enml2.dtd">'
	note.content += '<en-note><ul>'
	for sentence in sentence_list:
	    pasted_sentence = '<li>'+sentence + '</li>'
	    note.content += pasted_sentence
	note.content += '</ul></en-note>'
	note.tagNames = tag_list

	created_note = note_store.createNote(note)

	print "Successfully created a new note with GUID: ", created_note.guid

{% endhighlight %}