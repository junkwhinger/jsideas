---
layout:     post
title:      "[UPDATED]PDF Extractor in python"
date:       2015-05-23 10:30:00
author:     "Jun"
categories: "Python"
header-img: "img/post-bg-05.jpg"
---

<h2 class="section-heading">[Updated]PDf Extractor in python</h2>
<p><a href="http://jsideas.net/python/2015/05/16/pdf_extractor/">PDF extractor I designed the other day</a> worked find, but had some performance issues. Instead of extracting sentences in order of appearance, the code iterated over the key words like ‘Thus, ‘, ‘Therefore, ‘ and appended the results. As a result, I got a bunch of ‘Secondly, ‘ before ‘Thirdly ‘. Plus, it failed to print some long sentences that were written between two pages. To address this issue, I redesigned the code with the following steps.</p>

<ol>
	<li>read the pdf file</li>
	<li>split the file into paragraphs using spaces</li>
	<li>remove things like page numbers, words from figures</li>
	<li>if the paragraph starts with a lowercase letter, paste it with the previous paragraph</li>
	<li>using the keyword set, extract their index number in the whole text</li>
	<li>sort the index list in ascending order</li>
	<li>iterate through the index list and extract characters until the loop hits the period mark</li>
	<li>put them in a list and upload the list on my Evernote account</li>
</ol>

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

#spliting paragraphs
paragraph_text = re.split('\s{2,}',t)

#cleaning each paragraph
paragraph_list = list()
for paragraph in paragraph_text:
	adjusted_paragraph = paragraph.replace('\n'," ")
	adjusted_paragraph = adjusted_paragraph.replace('-', "")
	adjusted_paragraph = re.sub(RE_XML_ILLEGAL, "?", adjusted_paragraph)
	
	if len(adjusted_paragraph) > 3: #removing page numbers
		dot_found = "." in adjusted_paragraph
		semicolon_found = ";" in adjusted_paragraph
		colon_found = ":" in adjusted_paragraph
		exclam_found = "!" in adjusted_paragraph
		char_found = any(char.isalpha() for char in adjusted_paragraph)
		if (dot_found or semicolon_found or colon_found or exclam_found) and char_found == True:
			paragraph_list.append(adjusted_paragraph)

result = list()
for paragraph in paragraph_list:
	upper = paragraph[0].isupper()
	if upper:
		result.append(paragraph)
	else:
		last_appended_paragraph = result[len(result)-1]
		result.pop(len(result)-1)
		adj_paragraph = last_appended_paragraph + paragraph
		result.append(adj_paragraph)

result_string = " ".join(result)

#retrieve index of key words in the text and sort them in ascending order
extraction_index_list = list()
for key in key_set:
	for m in re.finditer(key, result_string):
		extraction_index_list.append(m.start())
extraction_index_list_sorted = sorted(extraction_index_list)

#extract the sentences
extraction_list = list()
for idx in extraction_index_list_sorted:
	counter = 0
	text = list()
	while result_string[idx + counter] != ".":
		char = result_string[idx + counter]
		text.append(char)
		counter = counter + 1
	text.append(".")
	text = "".join(text)
	text = text.replace('\n', " ")
	text = text.replace('- ',"")
	text = re.sub(RE_XML_ILLEGAL, "?", text)
	extraction_list.append(text)

print 'there are %i sentences extracted.' % len(extraction_list)

# top 5 most frequent word list to use as tags
d = defaultdict(int)

for word in result_string.split():
	d[word] += 1

del_key_list = ['I','i','o', 'you', 'your', 'We', 'we', 'You', 'the', 'The', 'of', 'Of', 'and', 'or', 'in', 'on', 'On', 'In', 'to', 'To', 'is', 'are', 'a', 'that', 'That', 'Those', 'those', 'by', 'until', 'has', 'from', 'it', 'be', 'at', 'This', 
				'this', 'under', 'own', 'as', 'As', 'have', 'between', 'an', 'with', 'not', 'its', 'over', 'also', 'more','for', 'their', 'Their' ]

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
if len(extraction_list) > 0:

	auth_token = "S=s101:U=aa4acd:E=154aa4bebfd:C=14d529abc30:P=1cd:A=en-devtoken:V=2:H=974dd38b2d7e3e0b9bac7432b2658cd5"

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
	for sentence in extraction_list:
	    pasted_sentence = '<li>'+sentence + '</li>'
	    note.content += pasted_sentence
	note.content += '</ul></en-note>'
	note.tagNames = tag_list

	created_note = note_store.createNote(note)

	print "Successfully created a new note with GUID: ", created_note.guid

{% endhighlight %}