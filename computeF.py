# coding=utf-8

import json, os

test_file = './result_dev.json'
target_file = './data/dev_data.json'


def compute_Fscore(test_file, target_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    with open(target_file, 'r', encoding='utf-8') as f:
        target_data = f.readlines()

    data_num = min(len(test_data), len(target_data))
    test_data = test_data[:data_num]
    target_data = target_data[:data_num]

    target_spo_num = 0
    predict_spo_num = 0
    predict_true_num = 0
    for i in range(data_num):
        test_item = json.loads(test_data[i])
        target_item = json.loads(target_data[i])

        test_spo_list = test_item['spo_list']
        target_spo_list = target_item['spo_list']

        target_spo_num += len(target_spo_list)
        predict_spo_num += len(test_spo_list)

        for pre in test_spo_list:
            for tar in target_spo_list:
                if pre['object'] == tar['object'] and pre['subject'] == tar['subject'] and pre['predicate'] == tar['predicate']:
                    predict_true_num += 1

    recall = predict_true_num * 1.0 / (target_spo_num+1e-10)
    precision = predict_true_num * 1.0 / (predict_spo_num+1e-10)
    F = 2 * recall * precision / (recall + precision+1e-10)

    return [precision, recall, F]


if __name__ == '__main__':
    [a,b,c] = compute_Fscore(test_file, target_file)
    print(a,b,c)
