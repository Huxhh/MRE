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