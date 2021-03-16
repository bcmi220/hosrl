from __future__ import absolute_import

from parser.structs.vocabs.index_vocabs import IDIndexVocab, DepheadIndexVocab, SemheadGraphIndexVocab
from parser.structs.vocabs.token_vocabs import FormTokenVocab, LemmaTokenVocab, PREDICATETokenVocab, PSENSETokenVocab, UPOSTokenVocab, XPOSTokenVocab, DeprelTokenVocab, SemrelGraphTokenVocab, SemrelGraphLabelVocab, AttributeTokenVocab, FrameTokenVocab
from parser.structs.vocabs.feature_vocabs import LemmaFeatureVocab, XPOSFeatureVocab, UFeatsFeatureVocab
from parser.structs.vocabs.subtoken_vocabs import FormSubtokenVocab, LemmaSubtokenVocab, PREDICATESubtokenVocab, PSENSESubtokenVocab, UPOSSubtokenVocab, XPOSSubtokenVocab, DeprelSubtokenVocab
from parser.structs.vocabs.pretrained_vocabs import FormPretrainedVocab, LemmaPretrainedVocab, PREDICATEPretrainedVocab, PSENSEPretrainedVocab, UPOSPretrainedVocab, XPOSPretrainedVocab, DeprelPretrainedVocab
from parser.structs.vocabs.bert_vocabs import FormBertVocab, DepheadBertVocab
from parser.structs.vocabs.albert_vocabs import FormAlbertVocab, DepheadAlbertVocab
from parser.structs.vocabs.xlnet_vocabs import FormXlnetVocab, DepheadXlnetVocab
from parser.structs.vocabs.elmo_vocabs import FormElmoVocab, DepheadElmoVocab
from parser.structs.vocabs.multivocabs import FormMultivocab, LemmaMultivocab, PREDICATEMultivocab, PSENSEMultivocab, UPOSMultivocab, XPOSMultivocab, XPOSMultivocab, DeprelMultivocab, Seq2SeqLabelMultivocab, LabelTokenVocab
#our vocab
from parser.structs.vocabs.second_order_vocab import SecondOrderGraphIndexVocab
from parser.structs.vocabs.second_order_token_vocab import SecondOrderGraphTokenVocab
from parser.structs.vocabs.second_order_LBP_vocab import SecondOrderGraphLBPVocab
#from parser.structs.vocabs.RNNDecoder_vocab import RNNDecoderVocab, Seq2SeqDecoderVocab, Seq2SeqAnchorPredictionVocab, Seq2SeqIDVocab, Seq2SeqGraphTokenVocab, Seq2SeqGraphIndexVocab, Seq2SeqSecondOrderGraphIndexVocab, Seq2SeqNodeLabelPredictionVocab
# from parser.structs.vocabs.RNNDecoder_vocab import *