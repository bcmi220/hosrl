import sys

def to_conll09(conllu_filename, raw_filename, srl_filename, lang='default', mode='end2end'):
    data = []
    sent = []
    with open(conllu_filename, 'r', encoding='utf-8') as conllu_f:
        for line in conllu_f:
            if len(line.strip()) > 0:
                sent.append(line.strip().split('\t'))
            else:
                if len(sent) > 0:
                    data.append(sent)
                    sent = []
    if len(sent) > 0:
        data.append(sent)
        sent = []

    raw_data = []
    sent = []
    with open(raw_filename, 'r', encoding='utf-8') as raw_f:
        for line in raw_f:
            if len(line.strip()) > 0:
                sent.append(line.strip().split('\t'))
            else:
                if len(sent) > 0:
                    raw_data.append(sent)
                    sent = []
    if len(sent) > 0:
        raw_data.append(sent)
        sent = []

    assert len(data) == len(raw_data)

    with open(srl_filename, 'w', encoding='utf-8') as srl_f:
        srl_data = []

        # convert
        for sent_idx in range(len(data)):
            assert len(data[sent_idx]) == len(raw_data[sent_idx])

            sent = data[sent_idx]

            preds = set()
            for line in sent:
                if line[3] != '_':
                    preds.add(int(line[0]))
            if mode == 'end2end': # the predgiven mode has already golden predicate identified, no need to use the semantic role relationship to help identifying
                for line in sent:
                    if line[10] != '_':
                        semrels = line[10].split('|')
                        for semrel in semrels:
                            semrel = semrel.split(':')
                            preds.add(int(semrel[0]))
            preds = list(preds)
            preds.sort()

            srl_sent = []
            for line_idx in range(len(sent)):
                line = sent[line_idx]
                # id, form, gold_lemma, pred_lamma, gold_pos, pred_pos, gold_feat, pred_feat, gold_dephead, pred_dephead, gold_deprel, pred_deprel 
                fields = [line[0], line[1], raw_data[sent_idx][line_idx][2], raw_data[sent_idx][line_idx][3], raw_data[sent_idx][line_idx][4], raw_data[sent_idx][line_idx][5], raw_data[sent_idx][line_idx][6], raw_data[sent_idx][line_idx][7], raw_data[sent_idx][line_idx][8], raw_data[sent_idx][line_idx][9], raw_data[sent_idx][line_idx][10], raw_data[sent_idx][line_idx][11]]
                # is_pred, pred_sense
                if (line_idx + 1) in preds:
                    if lang == 'japanese':
                        fields += ['Y', line[2]]
                    elif lang == 'czech':
                        if line[4] == '[LEMMA]':
                            fields += ['Y', line[2]]
                        else:
                            if raw_data[sent_idx][line_idx][13].startswith('v-'): # v-w[xxx]f[xxx]
                                fields += ['Y', raw_data[sent_idx][line_idx][13].split('f')[0]+line[4]]
                            else:
                                fields += ['Y', line[4]]
                    else:
                        if lang == 'german':
                            if line[4] == '_':
                                line[4] = '1' # to fix the possible error
                        else:
                            if line[4] == '_':
                                line[4] = '01' # to fix the possible error
                        fields += ['Y', line[2]+'.'+line[4]]
                else:
                    fields += ['_', '_']
                fields += ['_' for _ in range(len(preds))]
                if line[10] != '_':
                    semrels = line[10].split('|')
                    for semrel in semrels:
                        semrel = semrel.split(':')
                        pred = int(semrel[0])
                        if mode == 'end2end' or (mode == 'predgiven' and pred in preds):
                            pred_index = preds.index(pred)
                            if (lang == 'japanese' or lang == 'czech') and '/' in semrel[1]:
                                fields[14+pred_index] = semrel[1].replace('/', '|')
                            else:
                                fields[14+pred_index] = semrel[1]
                srl_sent.append(fields)
            
            srl_data.append(srl_sent)
        
        # write
        for srl_sent_idx in range(len(srl_data)):
            for line in srl_data[srl_sent_idx]:
                srl_f.write('\t'.join(line)+'\n')
            srl_f.write('\n')

if __name__ == '__main__':
    ''' translate conll09-format to conllu-format.
	'''
    if len(sys.argv) > 5:
        assert sys.argv[5] in ['end2end', 'predgiven']
        to_conll09(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) > 4:
        to_conll09(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        to_conll09(sys.argv[1], sys.argv[2], sys.argv[3])