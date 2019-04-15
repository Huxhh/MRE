#encoding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re


def match(sub_string,string):
	if string.startswith(sub_string):
		return 'B'
	if string.endswith(sub_string):
		return 'E'
	if sub_string in string:
		return 'M'
	return 'O'

def re_match(sub_string,string):
	if re.fullmatch(sub_string+'.*',string):
		return 'B'
	if re.fullmatch('.+'+sub_string,string):
		return 'E'
	if re.fullmatch('.+'+sub_string+'.+',string):
		return 'M'
	return "O"
def match_str(string_list,string):
	string = string.lower()
	for i,s in enumerate(string_list):
		s = s.lower()
		if string == s or string in s:
			return i,i
		if string.startswith(s):
			for j in range(i+1,len(string_list)):
				s = s + string_list[j].lower()
				if s == string or string in s:
					return i,j
	return -1,-1
def match_char(text,string):
	string = string.lower()
	for i,s in enumerate(text):
		s = s.lower()
		if string == s:
			return i,i
		if string.startswith(s):
			if ''.join(text[i:i+len(string)]).lower() == string:
				return i,i+len(string)-1
	return -1,-1

def process(filename,out_filename,is_char=True):
	if is_char:
		log = open(out_filename+'_char.log','w',encoding='utf-8')
		out = open(out_filename+'_char.json','w',encoding='utf-8')
		match_list = match_char
	else:
		log = open(out_filename+'_word.log','w',encoding='utf-8')
		out = open(out_filename+'_word.json','w',encoding='utf-8')
		match_list = match_str
	#
	with open(filename,encoding='utf-8') as file:

		for line in file:

			line = json.loads(line)
			pos_list = line['postag']
			text = line['text']
			if is_char:
				new_pos_list = []
				for pos in pos_list:
					word = pos['word']
					p = pos['pos']
					for char in word:
						new_pos_list.append({'word':char,'pos':p})
				pos_list = new_pos_list
			if len(pos_list) == 0:
				pos_list = [{'word':w,'pos':"#UNK"} for w in text]

			word_list = [x['word'] for x in pos_list]

			spo_list = line['spo_list']
			flag = False
			relations = []
			for spo in spo_list:
				obj = spo['object']
				predicate = spo['predicate']
				sbj = spo['subject']
				ob,oe = match_list(word_list,obj)
				sb,se = match_list(word_list,sbj)
				if ob == -1:
					log.write(json.dumps(line,ensure_ascii=False))
					log.write('\n')
					flag = True
					continue
				if sb == -1:
					log.write(json.dumps(line, ensure_ascii=False))
					log.write('\n')
					flag = True
					continue
				relations.append({'object_begin':ob,'object_end':oe,'subject_begin':sb,'subject_end':se,'predicate':predicate})
			if not flag:
				out.write(json.dumps({'text':text,'pos_list': pos_list, 'relations': relations},ensure_ascii=False))
				out.write('\n')
			else:
				out.write(json.dumps({'text':text,'pos_list': pos_list, 'relations':relations},ensure_ascii=False))
				out.write('\n')
def process_test(filename,out_filename,is_char=True):
	if is_char:
		out = open(out_filename+'_char.json','w',encoding='utf-8')
	else:
		out = open(out_filename+'_word.json','w',encoding='utf-8')
	#
	with open(filename,encoding='utf-8') as file:

		for line in file:

			line = json.loads(line)
			pos_list = line['postag']
			text = line['text']
			if is_char:
				new_pos_list = []
				for pos in pos_list:
					word = pos['word']
					p = pos['pos']
					for char in word:
						new_pos_list.append({'word':char,'pos':p})
				pos_list = new_pos_list
			if len(pos_list) == 0:
				pos_list = [{'word':w,'pos':"#UNK"} for w in text]

			out.write(json.dumps({'text':text,'pos_list':pos_list},ensure_ascii=False))
			out.write('\n')


if __name__ == '__main__':
	process('../data/train_data.json','../data/train_data')
	process('../data/dev_data.json','../data/dev_data')
	process('../data/train_data.json','../data/train_data',False)
	process('../data/dev_data.json','../data/dev_data',False)
	process_test('../data/test1_data_postag.json','../data/test_data')
	process_test('../data/test1_data_postag.json', '../data/test_data',False)