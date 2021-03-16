
def read_conll(srl_filename):
    data = []
    sent = []
    with open(srl_filename, 'r', encoding='utf-8') as srl_f:
        for line in srl_f:
            if len(line.strip()) > 0:
                sent.append(line.strip().split('\t'))
            else:
                if len(sent) > 0:
                    data.append(sent)
                    sent = []
    if len(sent) > 0:
        data.append(sent)
        sent = []
    return data



if __name__ == '__main__':
    # train_data = read_conll('./data/conll09/conll09-catalan/conll09_CAT_train.dataset')
    # dev_data = read_conll('./data/conll09/conll09-catalan/conll09_CAT_dev.dataset')
    # test_data = read_conll('./data/conll09/conll09-catalan/conll09_CAT_test.dataset')

    # train_data = read_conll('./data/conll09/conll09-chinese/conll09_ZHO_train.dataset')
    # dev_data = read_conll('./data/conll09/conll09-chinese/conll09_ZHO_dev.dataset')
    # test_data = read_conll('./data/conll09/conll09-chinese/conll09_ZHO_test.dataset')

    
    train_data = read_conll('./data/conll09/conll09-czech/conll09_CZE_train.dataset')
    dev_data = read_conll('./data/conll09/conll09-czech/conll09_CZE_dev.dataset')
    test_data = read_conll('./data/conll09/conll09-czech/conll09_CZE_test.dataset')
    test_ood_data = read_conll('./data/conll09/conll09-czech/conll09_CZE_test_ood.dataset')

    # train_data = read_conll('./data/conll09/conll09-english/conll09_ENG_train.dataset')
    # dev_data = read_conll('./data/conll09/conll09-english/conll09_ENG_dev.dataset')
    # test_data = read_conll('./data/conll09/conll09-english/conll09_ENG_test.dataset')
    # test_ood_data = read_conll('./data/conll09/conll09-english/conll09_ENG_test_ood.dataset')

    # train_data = read_conll('./data/conll09/conll09-german/conll09_GER_train.dataset')
    # dev_data = read_conll('./data/conll09/conll09-german/conll09_GER_dev.dataset')
    # test_data = read_conll('./data/conll09/conll09-german/conll09_GER_test.dataset')
    # test_ood_data = read_conll('./data/conll09/conll09-german/conll09_GER_test_ood.dataset')

    # train_data = read_conll('./data/conll09/conll09-japanese/conll09_JPN_train.dataset')
    # dev_data = read_conll('./data/conll09/conll09-japanese/conll09_JPN_dev.dataset')
    # test_data = read_conll('./data/conll09/conll09-japanese/conll09_JPN_test.dataset')

    # train_data = read_conll('./data/conll09/conll09-spanish/conll09_SPA_train.dataset')
    # dev_data = read_conll('./data/conll09/conll09-spanish/conll09_SPA_dev.dataset')
    # test_data = read_conll('./data/conll09/conll09-spanish/conll09_SPA_test.dataset')

    all_data = train_data + dev_data + test_data + test_ood_data

    # all_data = train_data + dev_data + test_data

    words = set()
    for sent in all_data:
        for line in sent:
            words.add(line[1])
            words.add(line[1].lower())

    with open('./data/fasttext_vecs/cc.cs.300.vec', 'r', encoding='utf-8') as fin:
        with open('./data/fasttext_vecs/cc.cs.300.filter.vec', 'w', encoding='utf-8') as fout:
            for line in fin:
                if len(line.strip()) > 0:
                    line = line.rstrip().split(' ')
                    if len(line) == 2:
                        continue
                    if line[0] in words:
                        fout.write(' '.join(line)+'\n')