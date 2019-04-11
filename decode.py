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
test_file = open('./data/dev_data_char.json',encoding='utf-8')
result_file = open('./result.json','w',encoding='utf-8')
with open('./relation_map.txt',encoding='utf-8') as file:
    relations = [line.strip() for line in file]

for line1,line2 in zip(out_file.readlines(),test_file.readlines()):
    line1 = json.loads(line1)
    line2 = json.loads(line2)
    text = line2['text']
    pos_list = line2['pos_list']
    spo_list = []
    for r,t in line1.items():
        object_list = t['object']
        subject_list = t['subject']
        for obj in object_list:
            # print(obj)
            for sbj in subject_list:
                # print(sbj)
                spo_list.append({'predicate':relations[int(r)]})
                spo_list[-1]['object'] = ''.join([pos_list[i]['word'] for i in range(obj[0],obj[1]+1)])
                spo_list[-1]['subject'] = ''.join([pos_list[i]['word'] for i in range(sbj[0],sbj[1]+1)])
                spo_list[-1]['object_type'] = ''
                spo_list[-1]['subject_type'] = ''

    result_file.write(json.dumps({'text':text,'spo_list':spo_list},ensure_ascii=False))
    result_file.write('\n')

out_file.close()
test_file.close()
result_file.close()




if __name__ == '__main__':
    pass