
import sys

def read_conll(path):
    data = []
    sent = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
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

def has_ho(sent):
    pred_num = len(sent[0]) - 14
    if pred_num == 0:
        return False
    for idx in range(len(sent)):
        if sent[idx][12] == 'Y':
            for jdx in range(pred_num):
                if sent[idx][14+jdx] != '_': # predicate as argument
                    return True
        else:
            pred_count = 0
            for jdx in range(pred_num):
                if sent[idx][14+jdx] != '_':
                    pred_count += 1
            if pred_count > 1: # multiple predicate
                return True
    return False


if __name__ == '__main__':

    conll_data = read_conll(sys.argv[1])

    print('total instances:', len(conll_data))

    ho_count = 0
    for conll_sent in conll_data:
        if has_ho(conll_sent):
            ho_count += 1
    
    print('with HO instances:', ho_count)


