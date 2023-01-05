import urllib.request
from html_table_parser.parser import HTMLTableParser
import pandas as pd
import numpy as np

# Opens a website and read its
# binary contents (HTTP Response Body)
def url_get_contents(url):

	# Opens a website and read its
	# binary contents (HTTP Response Body)

	#making request to the website
	req = urllib.request.Request(url=url)
	f = urllib.request.urlopen(req)

	#reading contents of the website
	return f.read()

def get_tables(url):
	# defining the html contents of a URL.
	xhtml = url_get_contents(url).decode('utf-8')

	# Defining the HTMLTableParser object
	p = HTMLTableParser()

	# feeding the html contents in the
	# HTMLTableParser object
	p.feed(xhtml)
	return p.tables