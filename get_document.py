import requests
import feedparser
import json
import os
from urllib import parse
import time


def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = requests.get(url)
        # write to file
        file.write(response.content)

topics = [
	#"Artificial Intelligence",
	"Computation and Language",
	"Computational Complexity",
	"Computational Engineering, Finance, and Science",
	"Graphics",
	"Information Theory",
	"Operating Systems",
	"Systems and Control",
	"Formal Languages and Automata Theory"]
for topic in topics:
	if not os.path.exists(topic):
		os.makedirs(topic)
	url = 'http://export.arxiv.org/api/query?search_query=all:{0}&start=90&max_results=1000'.format(parse.quote(topic))
	print(url)
	# data = requests.get(url).content
	# print(json.dumps(feedparser.parse(url) , indent=4))
	content = feedparser.parse(url).entries
	# print(content)
	for d in content:
		# print(d.summary)
		for link in d.links:
			# print(d.links)
			if(link.type =='application/pdf'):
				# with open()
				file_path = link.href
				file_name = topic + "/" + file_path.split("/")[-1] + ".pdf"
				print(file_name)
				summary_fp = file_name + ".txt"
				with open(summary_fp,"w") as w:
					w.write(d.summary)
					w.close()
				time.sleep(2)
				download(file_path , file_name)
				time.sleep(5)
				
