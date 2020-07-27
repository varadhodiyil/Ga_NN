import glob
import pandas as pd
from rake_nltk import Rake , Metric
import os , re

for subdir, dirs, files in os.walk("."):

	data = glob.glob("{0}/*.txt".format(subdir))
	print(subdir)
	to_csv = list()
	for d in data:
		try:
			print(d)
			id = d.split(".pdf.txt")[0]
			id = id.split("\\")[2]
			d  = open(d).read()
			d = d.lower()
			re.sub('[^A-Za-z0-9]+ ', '', d)
			r = Rake(ranking_metric=Metric.WORD_FREQUENCY ,max_length=2 )

			r.extract_keywords_from_text(d)
			_resp = r.get_ranked_phrases()[:5]
			response = dict()
			response['id'] = id
			response['summary'] = d
			response['keywords_freq'] = ",".join(_resp)

			r = Rake(ranking_metric=Metric.WORD_DEGREE ,max_length=2 )

			r.extract_keywords_from_text(d)
			_resp = r.get_ranked_phrases()[:5]
			response['keywords_deg'] = ",".join(_resp)

			r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO ,max_length=2 )

			r.extract_keywords_from_text(d)
			_resp = r.get_ranked_phrases()[:5]
			response['keywords_ratio'] = ",".join(_resp)



			to_csv.append(response)
		except Exception as e:
			print(e)

	if len(to_csv) > 0:
		import csv
		header = to_csv[0].keys()
		with open("data.csv","a",newline='') as w:
			_w = csv.DictWriter(w , fieldnames=header)
			_w.writeheader()
			_w.writerows(to_csv)



