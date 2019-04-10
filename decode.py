#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: decode.py.py
@time: 2019/4/8 16:10
"""
import json

out_file = open('./out.json',encoding='utf-8')
test_file = open('./data/test_data_char.json',encoding='utf-8')
result_file = open('./result.json',encoding='utf-8')
with open('./relation_map.txt',encoding='utf-8') as file:
    relations = [line.strip() for line in file]

for line1,line2 in zip(out_file.readlines(),test_file.readlines()):
    line1 = json.load(line1)
    line2 = json.load(line2)
    text = line2['text']
    pos_list = line2['pos_list']
    spo_list = []
    for r,t in line1.items():
        spo_list.append({'predicate':relations[r]})
        spo_list[-1]['subject'] = ''.join([spo_list[i][0] for i in range(t['subject_begin'],t['subject_end']+1)])
        spo_list[-1]['object'] = ''.join([spo_list[i][0] for i in range(t['object_begin'], t['object_end'] + 1)])
        spo_list[-1]['subject_type'] = ''
        spo_list[-1]['object_type'] = ''
    result_file.write({'text':text,'spo_list':spo_list})

out_file.close()
test_file.close()
result_file.close()




if __name__ == '__main__':
    pass