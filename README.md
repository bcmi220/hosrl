# HO-SRL
Code for EMNLP-2020 paper "High-order Semantic Role Labeling"

## Under Construction


## 1. Prepare the training data

```bash
python scripts/conll09_to_conllu.py ./data/conll09/conll09-english/conll09_ENG_train.dataset ./data/conll09-converted/conll09_ENG_train.conllu 0
python scripts/conll09_to_conllu.py ./data/conll09/conll09-english/conll09_ENG_dev.dataset ./data/conll09-converted/conll09_ENG_dev.conllu 0
python scripts/conll09_to_conllu.py ./data/conll09/conll09-english/conll09_ENG_test.dataset ./data/conll09-converted/conll09_ENG_test.conllu 0
python scripts/conll09_to_conllu.py ./data/conll09/conll09-english/conll09_ENG_test_ood.dataset ./data/conll09-converted/conll09_ENG_test_ood.conllu 0
```

## 2. Predict the predicate sense with given predicate
```bash
# Use any sequence labeling model
```

## 3. Train the SRL model
```bash
python -u main.py train GraphParserNetwork --config_file config_srl/glove_01lr_5decay_srl_switch_new1.cfg --noscreen > ./logs/sec_order_baseline.log 2>&1
```

