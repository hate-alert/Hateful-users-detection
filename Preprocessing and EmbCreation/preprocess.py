import re
from nltk.tokenize import TweetTokenizer
import emoji

remove_puncts="[\{\};,.[!@#$%^&*()_+=?/\'\"\]]"
apostrophe_file=open('apostrophe.txt')
apostrophe_dict={}
tknzr=TweetTokenizer(strip_handles=True,reduce_len=True)

for line in apostrophe_file:
	line=line.rstrip().split('\t')
	poss_vals=line[1].split(',')
	if line[0] not in apostrophe_dict:
		apostrophe_dict[line[0]]=poss_vals

try:
	# Wide UCS-4 build
	myre = re.compile(u'['
		u'\U0001F300-\U0001F64F'
		u'\U0001F680-\U0001F6FF'
		u'\u2600-\u26FF\u2700-\u27BF]+', 
		re.UNICODE)
except re.error:
	# Narrow UCS-2 build
	myre = re.compile(u'('
		u'\ud83c[\udf00-\udfff]|'
		u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
		u'[\u2600-\u26FF\u2700-\u27BF])+', 
		re.UNICODE)		

emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
remoji = re.compile('|'.join(re.escape(p) for p in emojis_list))

count=0		

web_url="http[s]?:[a-zA-Z._0-9/]+[a-zA-Z0-9]"
replacables="RT\s|-\s|\s-|#|@|[|}|]|{|(|)"
prop_name="([A-Z][a-z]+)"
num="([0-9]+)"
name="([A-Za-z]+)"
and_rate="([&][a][m][p][;])"
ellipses="([A-Za-z0-9]+[…])"
hashtags_2="([#][a-zA-z0-9]+[\s\n])"

def tweet_preprocess2(text):
	
	text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
	text = re.sub('http://', '', text)
	text = re.sub('https://', '', text)
	text = re.sub('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',text)
	text = re.sub('@', '',text)
	text=re.sub(and_rate,'and',text)

	text=re.sub(replacables,'',text)
	# text="".join(tknzr.tokenize(text))

	prev_text=''
	while text!=prev_text:
		prev_text=str(text)
		text=re.sub(str(num)+''+name,"\\1 \\2",text)
		text=re.sub(name+''+str(num),"\\1 \\2",text)
		text=re.sub(prop_name+''+prop_name,"\\1 \\2",text)

	text = myre.sub('',text)
	
	return text.strip()
