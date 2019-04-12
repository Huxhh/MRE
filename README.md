# RE
re

```
cd ./preprocess
python make_ner_data.py
python make_code_map.py
cd ..
python model.py
python model.py --mode infer
python decode.py
```

## 计划

### 针对稀有关系的优化：
有些关系过于稀疏，似乎无法训练出来

### 模型更改
目前针对每种关系，分别抽出主客体(每个位置标注数为关系数量*4)

接下来考虑：

针对每个实体对，直接判断关系(每个位置标注数为2，任意两个位置增加关系数个标注)