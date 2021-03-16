
import argparse
import logging
import torch
import os
from transformers import *
import h5py
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (AlbertConfig, BertConfig, XLNetConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertModel, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
}

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

def read_conll_data(file_path):
    # load the plain conll format file
    with open(file_path, encoding="utf-8") as f:
        plain_data = f.readlines()
    
    conll_data = []
    conll_sent = []
    for line in plain_data:
        if len(line.strip()) == 0:
            if len(conll_sent) > 0:
                conll_data.append(conll_sent)
                conll_sent = []
        else:
            conll_sent.append(line.strip().split('\t'))
    if len(conll_sent) > 0:
        conll_data.append(conll_sent)
        conll_sent = []

    return conll_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2009 SRL task.",
    )

    parser.add_argument(
        "--conllu_files",
        default='',
        type=str,
        help="The conllu file names.",
    )
    parser.add_argument(
        "--extract_layer",
        default=-1,
        type=int,
        help="The layer used for extraction.",
    )

    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The hdf5 output path.",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size GPU/CPU.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")


    args = parser.parse_args()

    args.conllu_files = args.conllu_files.split(',')

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    print("Tokenizer arguments: %s" % tokenizer_args)


    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.output_attentions = True
    config.output_hidden_states = True
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    cls_token_at_end=bool(args.model_type in ["xlnet"])
    # xlnet has a cls token at the end
    cls_token=tokenizer.cls_token
    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0
    sep_token=tokenizer.sep_token
    sep_token_extra=bool(args.model_type in ["roberta"])
    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
    pad_on_left=bool(args.model_type in ["xlnet"])
    # pad on the left for xlnet
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0
    sequence_a_segment_id=0
    mask_padding_with_zero=True

    hdf5_f = h5py.File(args.output_path, "w")

    for conllu_file in args.conllu_files:
        conll_data = read_conll_data(os.path.join(args.data_dir, conllu_file))

        all_t2w_index_map = []
        all_w2t_start_index_map = []
        all_w2t_end_index_map = []
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []

        for conll_idx, conll_sent in enumerate(conll_data):
            t2w_index_map = []
            w2t_start_index_map = []
            w2t_end_index_map = []
            tokens = []
            for line_idx, line in enumerate(conll_sent):
                word_idx = line_idx
                word = line[1]

                word_tokens = tokenizer.tokenize(word)

                t2w_index_map.extend([word_idx] * len(word_tokens))
                w2t_start_index_map.append(len(tokens))
                w2t_end_index_map.append(len(tokens) + len(word_tokens) - 1)
                tokens.extend(word_tokens)

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()

            assert len(tokens) <= args.max_seq_length - special_tokens_count

            tokens += [sep_token]
            t2w_index_map += [None]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                t2w_index_map += [None]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                t2w_index_map += [None]
                w2t_start_index_map = [len(tokens)-1] + w2t_start_index_map
                w2t_end_index_map = [len(tokens)-1] + w2t_end_index_map
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                t2w_index_map = [None] + t2w_index_map
                w2t_start_index_map = [0] + [item +1 if item is not None else None for item in w2t_start_index_map]
                w2t_end_index_map = [0] + [item +1 if item is not None else None for item in w2t_end_index_map]
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                t2w_index_map = ([None] * padding_length) + t2w_index_map
                w2t_start_index_map = [item + padding_length if item is not None else None for item in w2t_start_index_map]
                w2t_end_index_map = [item + padding_length if item is not None else None for item in w2t_end_index_map]
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                t2w_index_map += [None] * padding_length

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length
            assert len(t2w_index_map) == args.max_seq_length
            assert len(w2t_start_index_map) == len(conll_sent) + 1
            assert len(w2t_end_index_map) == len(conll_sent) + 1

            if conll_idx < 2:
                print(tokens)
                print(input_ids)
                print(input_mask)
                print(segment_ids)
                print(w2t_start_index_map)
                print()

            all_t2w_index_map.append(t2w_index_map)
            all_w2t_start_index_map.append(w2t_start_index_map)
            all_w2t_end_index_map.append(w2t_end_index_map)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)


        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

        extract_idx = 0
        for batch in tqdm(dataloader, desc="Extracting"):
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = model(**inputs)
                if args.extract_layer == -1:
                    outputs = outputs[0]
                else:
                    if args.model_type == 'xlnet':
                        outputs = outputs[1][args.extract_layer]
                    else:
                        outputs = outputs[2][args.extract_layer]
            # B x L x D
            outputs = outputs.detach().cpu().numpy()

            for batch_idx in range(outputs.shape[0]):
                hidden_states = outputs[batch_idx]
                w2t_start_index = all_w2t_start_index_map[extract_idx]
                sentence_hidden_states = hidden_states[w2t_start_index, :]
                hdf5_f.create_dataset("%s-%d"%(conllu_file, extract_idx),data=sentence_hidden_states)
                extract_idx += 1

    hdf5_f.close()


            



