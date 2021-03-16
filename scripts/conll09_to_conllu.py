import sys

def to_conllu(srl_filename, conllu_filename, is_predict='0', lang='default'):

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
    
    with open(conllu_filename, 'w', encoding='utf-8') as conllu_f:
        conllu_data = []
        
        # convert
        for sent_idx in range(len(data)):
            sent = data[sent_idx]

            # get all predicate positions
            pred_pos = [line_idx for line_idx in range(len(sent)) if sent[line_idx][12] == 'Y']

            conllu_sent = []
            for word_idx in range(len(sent)):
                fields = []
                is_predicate =  sent[word_idx][12]
                if lang == 'japanese':
                    pred_sense = '_'
                elif lang == 'czech':
                    if sent[word_idx][12] == 'Y':
                        if sent[word_idx][13].startswith('v-'): # v-w[xxx]f[xxx]
                            pred_sense =  'f'+sent[word_idx][13].split('f')[1]
                        else:
                            pred_sense = '[LEMMA]'
                    else:
                        pred_sense = '_'
                else:
                    pred_sense =  '_' if sent[word_idx][12] == '_' else sent[word_idx][13].split('.')[-1] # maybe the lemma is "." !important!
                # id, form, lemma, predicate, psense, upos, xpos
                fields += [sent[word_idx][0], sent[word_idx][1], sent[word_idx][3], is_predicate, pred_sense, '_', sent[word_idx][5]]
                # feats,dephead,deprel,semhead,semrel
                fields += ['_', sent[word_idx][8], sent[word_idx][10], '_', '_']
                if is_predict == '0':
                    semrels = []
                    for pred_idx, pred_p in enumerate(pred_pos):
                        if sent[word_idx][14+pred_idx] != '_':
                            if (lang == 'japanese' or lang == 'czech') and '|' in sent[word_idx][14+pred_idx]:
                                semrels.append(str(pred_p+1)+':'+sent[word_idx][14+pred_idx].replace('|', '/')) # replace
                            else:
                                semrels.append(str(pred_p+1)+':'+sent[word_idx][14+pred_idx])
                    if len(semrels) > 0:
                        fields[10]='|'.join(semrels)
                conllu_sent.append(fields)
            conllu_data.append(conllu_sent)

        # write
        for conllu_sent_idx in range(len(conllu_data)):
            for line in conllu_data[conllu_sent_idx]:
                conllu_f.write('\t'.join(line)+'\n')
            conllu_f.write('\n')

if __name__ == '__main__':
    ''' translate conll09-format to conllu-format.
	'''
    if len(sys.argv) > 4:
        to_conllu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        to_conllu(sys.argv[1], sys.argv[2], sys.argv[3])
    