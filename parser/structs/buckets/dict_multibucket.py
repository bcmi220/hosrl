#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2017 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import numpy as np

from .base_multibucket import BaseMultibucket
from .dict_bucket import DictBucket
# import pdb

import os
import bert
from bert import tokenization

import sys
sys.path.insert(0, './xlnet')
import xlnet
import sentencepiece as spm

#***************************************************************
class DictMultibucket(BaseMultibucket, dict):
	""""""
	
	#=============================================================
	def __init__(self, vocabs, max_buckets=2, config=None):
		""""""
		
		super(DictMultibucket, self).__init__(max_buckets, config=config)
		dict.__init__(self)
		
		for vocab in vocabs:
			self[vocab.classname] = [DictBucket(idx, vocab.depth, config=config) for idx in six.moves.range(max_buckets)]
		
		self._lengths = []
		self._indices = {vocab.classname: [] for vocab in vocabs}
		self._tokens = {vocab.classname: [] for vocab in vocabs}
		self._max_lengths = []
		
		input_cfg=self._config.get('GraphParserNetwork','input_vocab_classes')
		if 'Bert' in input_cfg:
			try:
				is_pretrained = self._config.get('BertVocab', 'use_pretrained_file')=='True'
			except:
				is_pretrained = False
			if is_pretrained:
				self.pretrained_bert=True
			else:
				self.pretrained_bert=False
				bert_path = self._config.get('BertVocab', 'bert_path')
				bert_vocab_file = os.path.join(bert_path,'vocab.txt')
				try:
					tokenize_chinese_chars = self._config.get('BertVocab', 'tokenize_chinese_chars')
				except:
					tokenize_chinese_chars = True
				self.bert_tokenizer=tokenization.FullTokenizer(bert_vocab_file, do_lower_case=self._config.get('BertVocab', 'cased'), tokenize_chinese_chars=tokenize_chinese_chars)
				self.bert_sep_token=self.bert_tokenizer.convert_tokens_to_ids(['[CLS]','[SEP]'])
		if 'Elmo' in input_cfg:
			try:
				is_pretrained = self._config.get('ElmoVocab', 'use_pretrained_file')=='True'
			except:
				is_pretrained = False
			if is_pretrained:
				self.pretrained_elmo=True
			else:
				self.pretrained_elmo=False
		if 'Albert' in input_cfg:
			import albert
			from albert import tokenization as albert_tokenization
			try:
				is_pretrained = self._config.get('AlbertVocab', 'use_pretrained_file')=='True'
			except:
				is_pretrained = False
			if is_pretrained:
				self.pretrained_albert=True
			else:
				self.pretrained_albert=False
				albert_path = self._config.get('AlbertVocab', 'albert_path')
				albert_vocab_file = os.path.join(albert_path,'30k-clean.vocab')
				albert_spm_model_file = os.path.join(albert_path,'30k-clean.model')
				self.albert_tokenizer=albert_tokenization.FullTokenizer(albert_vocab_file,True, albert_spm_model_file)
				self.albert_sep_token=self.albert_tokenizer.convert_tokens_to_ids(['[CLS]','[SEP]'])
		if 'Xlnet' in input_cfg:
			try:
				is_pretrained = self._config.get('XlnetVocab', 'use_pretrained_file')=='True'
			except:
				is_pretrained = False
			if is_pretrained:
				self.pretrained_xlnet=True
			else:
				self.pretrained_xlnet=False
				xlnet_path = self._config.get('XlnetVocab', 'xlnet_path')
				xlnet_vocab_file = os.path.join(xlnet_path,'spiece.model')
				self.xlnet_tokenizer=spm.SentencePieceProcessor()
				self.xlnet_tokenizer.Load(xlnet_vocab_file)
				CLS_ID = 3
				SEP_ID = 4
				self.xlnet_sep_token= [CLS_ID, SEP_ID]
		output_cfg=self._config.get('GraphParserNetwork','output_vocab_classes')
		self.use_seq2seq=False
		if 'seq2seq' in output_cfg.lower():
			self.use_seq2seq=True
		return
	
	#=============================================================
	def reset(self, vocabs):
		""""""
		
		self._lengths = []
		self._indices = {vocab.classname: [] for vocab in vocabs}
		self._tokens = {vocab.classname: [] for vocab in vocabs}
		for vocab_classname in self:
			for bucket in self[vocab_classname]:
				bucket.reset()
		return
	
	#=============================================================
	def add(self, indices, tokens, length=0):
		""""""
		
		assert self._is_open, 'The DictMultibucket is not open for adding entries'
		
		if length <= 1:
			return
		for vocab_classname in indices.keys():
			self._indices[vocab_classname].append(indices[vocab_classname])
		for vocab_classname in tokens.keys():
			self._tokens[vocab_classname].append(tokens[vocab_classname])
		super(DictMultibucket, self).add(length)
		return 
	
	#=============================================================
	def close(self):
		""""""
		
		# Decide where everything goes
		self._lengths = np.array(self._lengths)
		self._max_lengths = self.compute_max_lengths(self._lengths, self.max_buckets)
		len2bkt = self.get_len2bkt(self._max_lengths)
		
		# Open the buckets
		shape = len(self._lengths)
		dtype = [('bucket', 'i4')] + [(vocab_classname, 'i4') for vocab_classname in self.keys()]
		data = np.zeros(shape, dtype=dtype)
		indices_size=len(self._indices[list(self.keys())[0]])
		to_resume=[]
		#pdb.set_trace()
		vocab_names=list(self.keys())
		vocab_names.sort()
		for i, vocab_classname in enumerate(vocab_names):
			for bucket in self[vocab_classname]:
				bucket.open()
		
			# Add sentences to them
			#pdb.set_trace()
			if 'seq2seq' in vocab_classname.lower():
				to_resume.append(vocab_classname)
				continue
			for j, (indices, tokens) in enumerate(zip(self._indices[vocab_classname], self._tokens[vocab_classname])):
				bucket_index = len2bkt[len(indices)]
				sequence_index = self[vocab_classname][bucket_index].add(indices, tokens)
				data[vocab_classname][j] = sequence_index
				if i == 0:
					data['bucket'][j] = bucket_index
				else:
					assert data['bucket'][j] == bucket_index, 'CoNLLU data is somehow misaligned'
		if self.use_seq2seq:
			tar_bucket=data['FormMultivocab']
			for vocab_classname in to_resume:
				data[vocab_classname]=data['FormMultivocab']
		# pdb.set_trace()
		# Close the buckets
		for vocab_classname in self:
			#pdb.set_trace()
			for bucket in self[vocab_classname]:

				if vocab_classname=='FormBertVocab':
					#pdb.set_trace()
					if self.pretrained_bert:
						bucket.bert_close(is_pretrained=True)
					else:
						bucket.bert_close(sep_token=self.bert_sep_token)
				elif vocab_classname=='DepheadBertVocab':
					bucket.bert_close(is_pretrained=True,get_dephead=True)
				elif vocab_classname=='FormElmoVocab':
					#pdb.set_trace()
					if self.pretrained_elmo:
						bucket.elmo_close(is_pretrained=True)
					else:
						bucket.elmo_close()
				elif vocab_classname=='DepheadElmoVocab':
					bucket.elmo_close(is_pretrained=True,get_dephead=True)
				elif vocab_classname=='FormAlbertVocab':
					#pdb.set_trace()
					if self.pretrained_albert:
						bucket.albert_close(is_pretrained=True)
					else:
						bucket.albert_close(sep_token=self.albert_sep_token)
				elif vocab_classname=='DepheadAlbertVocab':
					bucket.albert_close(is_pretrained=True,get_dephead=True)
				elif vocab_classname=='FormXlnetVocab':
					#pdb.set_trace()
					if self.pretrained_xlnet:
						bucket.xlnet_close(is_pretrained=True)
					else:
						bucket.xlnet_close(sep_token=self.xlnet_sep_token)
				elif vocab_classname=='DepheadXlnetVocab':
					bucket.xlnet_close(is_pretrained=True,get_dephead=True)
				else:
					# if 'seq2seq' in vocab_classname.lower():
					# 	pdb.set_trace()
					bucket.close()
		super(DictMultibucket, self).close(data)
		
		return
	
	#=============================================================
	def get_data(self, vocab_classname, indices):
		""""""
		
		bucket_index = np.unique(self.bucket_indices[indices])
		assert len(bucket_index) == 1, 'Requested data from multiple (or no) buckets'
		
		bucket_index = bucket_index[0]
		data_indices = self.data[vocab_classname][indices]
		if vocab_classname=='FormBertVocab' and not self.pretrained_bert:
			data={}
			for key in self[vocab_classname][bucket_index].data:
				data[key] = self[vocab_classname][bucket_index].data[key][data_indices]
			#pdb.set_trace()
		elif vocab_classname=='FormElmoVocab' and not self.pretrained_elmo:
			data={}
			for key in self[vocab_classname][bucket_index].data:
				data[key] = self[vocab_classname][bucket_index].data[key][data_indices]
			#pdb.set_trace()
		elif vocab_classname=='FormAlbertVocab' and not self.pretrained_albert:
			data={}
			for key in self[vocab_classname][bucket_index].data:
				data[key] = self[vocab_classname][bucket_index].data[key][data_indices]
			#pdb.set_trace()
		elif vocab_classname=='FormXlnetVocab' and not self.pretrained_xlnet:
			data={}
			for key in self[vocab_classname][bucket_index].data:
				data[key] = self[vocab_classname][bucket_index].data[key][data_indices]
			#pdb.set_trace()
		else:
			data = self[vocab_classname][bucket_index].data[data_indices]
		return data
	
	#=============================================================
	def get_tokens(self, vocab_classname, indices):
		""""""
		
		return [self._tokens[vocab_classname][index] for index in indices]
	
	#=============================================================
	@property
	def lengths(self):
		return self._lengths
	@property
	def max_lengths(self):
		return self._max_lengths
	@property
	def bucket_indices(self):
		return self.data['bucket']
	@property
	def data_indices(self):
		return self.data[:, 2:]
