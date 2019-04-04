from collections import Counter
import json
with open('../data/all_50_schemas',encoding='utf-8') as file:
	type_set = set([])
	relation_set = set([])
	for line in file:
		line = json.loads(line)
		type_set.add(line['object_type'])
		relation_set.add(line['predicate'])
		type_set.add(line['subject_type'])
with open('../type_map.txt','w',encoding='utf-8') as file:
	for t in type_set:
		file.write('%s_B\n'%t)
		file.write('%s_M\n'%t)
		file.write('%s_E\n'%t)
	file.write('O\n')

with open('../relation_map.txt','w',encoding='utf-8') as file:
	for r in relation_set:
		file.write('%s\n'%r)
	file.write('O\n')

word_counter = Counter()
char_counter = Counter()
pos_set = set([])
with open('../data/train_data.json',encoding='utf-8') as file:
	for line in file:
		line = json.loads(line)
		for pos in line['postag']:
			word_counter.update([pos['word']])
			pos_set.add(pos['pos'])
		for char in line['text']:
			char_counter.update([char])

with open('../word_map.txt','w',encoding='utf-8') as file:
	word_counter = sorted(word_counter.items(),key=lambda x:-x[1])
	file.write('#PAD\n')
	file.write('#UNK\n')
	for word,count in word_counter:
		file.write(word)
		file.write('\n')

with open('../char_map.txt','w',encoding='utf-8') as file:
	char_counter = sorted(char_counter.items(),key=lambda x:-x[1])
	file.write('#PAD\n')
	file.write('#UNK\n')
	for char,count in char_counter:
		file.write(char)
		file.write('\n')

with open('../pos_map.txt','w',encoding='utf-8') as file:
	for pos in pos_set:
		file.write(pos)
		file.write('\n')



