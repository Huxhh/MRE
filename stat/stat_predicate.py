#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: stat_predicate.py.py
@time: 2019/4/12 13:17
"""
import json
from collections import Counter
def stat_predicate(filename):
    count_all = 0
    predicate_count = Counter()
    predicate_num = Counter()

    with open(filename,encoding='utf-8') as file:
        for line in file:
            count_all += 1
            item = json.loads(line)
            spo_list = item['spo_list']
            predicates = [spo['predicate'] for spo in spo_list]
            predicate_count.update(predicates)
            predicate_num.update(set(predicates))
    return predicate_num,predicate_count,count_all


if __name__ == '__main__':
    train_n,train_c,train_a = stat_predicate('../data/train_data.json')
    dev_n,dev_c,dev_a = stat_predicate('../data/dev_data.json')
    pred_n,pred_c,pred_a = stat_predicate('../result_dev_0.300.json')
    with open('stat_macro.csv','w',encoding='utf-8') as file:
        file.write('key,train_items_ratio,dev_items_ratio,pred_items_ratio,train_count,dev_count,pred_count\n')
        for key in train_n.keys():
            train_items = train_n.get(key,0)/train_a
            train_count = train_c.get(key,0)
            dev_items = dev_n.get(key,0)/dev_a
            dev_count = dev_c.get(key,0)
            pred_items = pred_n.get(key,0)/pred_a
            pred_count = pred_c.get(key,0)
            file.write('%s,%f,%f,%f,%d,%d,%d\n'%(key,train_items,dev_items,pred_items,train_count,dev_count,pred_count))
    with open('stat_micro.csv','w',encoding='utf-8') as file:

        t_a = sum(train_c.values())
        d_a = sum(dev_c.values())
        p_a = sum(pred_c.values())
        file.write('key,train_ratio,dev_ratio,pred_ratio\n')
        for key in train_n.keys():
            train_ratio = train_c.get(key,0)/t_a
            dev_ratio = dev_c.get(key,0)/d_a
            pred_ratio = pred_c.get(key,0)/p_a
            file.write('%s,%f,%f,%f\n'%(key,train_ratio,dev_ratio,pred_ratio))

