#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2019 Xinyu Wang
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
# import pdb
import os
import codecs
import zipfile
import gzip
try:
  import lzma
except:
  try:
    from backports import lzma
  except:
    import warnings
    warnings.warn('Install backports.lzma for xz support')
try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl
from collections import Counter

import numpy as np
import tensorflow as tf
 
from parser.structs.vocabs.base_vocabs import SetVocab
from . import conllu_vocabs as cv
from parser.neural import embeddings,nonlin,classifiers
import sys
sys.path.insert(0, './albert')

import albert

#***************************************************************
# TODO maybe change self.name to something more like self._save_str?
# Ideally there should be Word2vecVocab, GloveVocab, FasttextVocab,
# each with their own _save_str
class AlbertVocab(SetVocab):
  """"""
  
  #=============================================================
  def __init__(self, Albert_file=None, name=None, config=None, is_eval=False):
    """"""
    
    if (Albert_file is None) != (name is None):
      raise ValueError("You can't pass in a value for only one of Albert_file and name to AlbertVocab.__init__")
    super(AlbertVocab, self).__init__(config=config, is_eval=is_eval)
    self._Albert_file = Albert_file
    self._name = name
    self.variable = None
    self.pretrained = self.get_pretrained
    if not self.pretrained:
      self.placeholder = {'input_ids':tf.placeholder(tf.int32, [None,None], name=self.classname+'input_ids'),\
      'input_mask':tf.placeholder(tf.int32, [None,None], name=self.classname+'input_mask'),\
      'segment_ids':tf.placeholder(tf.int32, [None,None], name=self.classname+'segment_ids'),\
      'mapping':tf.placeholder(tf.int32, [None,None], name=self.classname+'mapping')}
      self.albert_model = self.start_albert(is_eval)
    else:
      self.placeholder = tf.placeholder(tf.float32, [None,None,1024], name=self.classname+'input_feats')

    return
  
  def start_albert(self, is_eval=False):
    from albert import tokenization
    from albert import modeling

    #default: "../tfhub_models/albert_uncased_L-12_H-768_A-12"
    albert_path = self.get_albert_path
    albert_config_file = os.path.join(albert_path, 'albert_config.json')
    albert_vocab_file = os.path.join(albert_path,'30k-clean.vocab')
    albert_spm_model_file = os.path.join(albert_path,'30k-clean.model')
    self.init_checkpoint = os.path.join(albert_path,'model.ckpt-best')
    
    albert_config = modeling.AlbertConfig.from_json_file(albert_config_file)
    self.tokenizer=tokenization.FullTokenizer(albert_vocab_file,True, albert_spm_model_file)
    # input_ids_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='input_ids_p')
    # input_mask_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='input_mask_p')
    # segment_ids_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='segment_ids_p')
    input_ids=self.placeholder['input_ids']
    self.input_mask=self.placeholder['input_mask']
    segment_ids=self.placeholder['segment_ids']
    print('ALBERT Training:', self.is_training)
    albert_model = modeling.AlbertModel(
        config = albert_config,
        is_training = self.is_training and not is_eval,
        input_ids = input_ids,
        input_mask = self.input_mask,
        token_type_ids = segment_ids,
        use_one_hot_embeddings = False,
    )
    return albert_model
  def modelInit(self,sess):
    #assert 0, 'Please init later than global init'
    self.restore_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='bert'), max_to_keep=1)
    self.restore_saver.restore(sess, self.init_checkpoint)
  def modelRestore(self,sess,albertlist,model_dir=None):
    #pdb.set_trace()
    self.restore_saver = tf.train.Saver(albertlist, max_to_keep=1)
    if self.is_training:
      self.restore_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    else:
      self.restore_saver.restore(sess, self.init_checkpoint)
  def modelSave(self,sess,save_dir,global_step):
    #assert 0, 'Please init later than global init'
    #self.restore_saver.restore(sess, self.init_checkpoint)
    self.restore_saver.save(sess, os.path.join(save_dir, 'albert-ckpt'), global_step=global_step, write_meta_graph=False)
  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    #pdb.set_trace()
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    #pdb.set_trace()
    if self.pretrained:
      outputs=self.placeholder
    else:
      if self.layer_index != -1:
        outputs=self.albert_model.get_all_encoder_layers()[self.layer_index]
      else:
        outputs=self.albert_model.get_sequence_output()
      mapping=self.placeholder['mapping']
      if self.strategy=="first_value":
        outputs=tf.batch_gather(outputs,mapping)*tf.cast((mapping>0),dtype=tf.float32)[:,:,None]
      elif self.strategy=="average":
        assert 0, "not implemented"
      else:
        assert 0, "please specify albert strategy"
    with tf.variable_scope('albert_vocab'):
      layer=classifiers.hidden(outputs,self.linear_size,hidden_func=self.hidden_func)
    return layer
  #=============================================================
  def get_embedding(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    #pdb.set_trace()
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    #pdb.set_trace()
    outputs=self.albert_model.get_sequence_output()
    mapping=self.placeholder['mapping']
    if self.strategy=="first_value":
      outputs=tf.batch_gather(outputs,mapping)*tf.cast((mapping>0),dtype=tf.float32)[:,:,None]
    elif self.strategy=="average":
      assert 0, "not implemented"
    else:
      assert 0, "please specify albert strategy"
    return outputs
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    #pdb.set_trace()
    if not self.pretrained:
      feed_dict[self.placeholder['input_ids']] = indices['input_ids']
      feed_dict[self.placeholder['input_mask']] = indices['input_mask']
      feed_dict[self.placeholder['segment_ids']] = indices['segment_ids']
      feed_dict[self.placeholder['mapping']] = indices['mapping']
    else:
      feed_dict[self.placeholder] = indices
    return feed_dict
  def add(self, token):
    tokens=self.tokenizer.tokenize(token)
    token_ids=self.tokenizer.convert_tokens_to_ids(tokens)

    return token_ids
  #=============================================================
  def count(self, *args):
    """"""
    return True
    max_embed_count = self.max_embed_count
    if self.Albert_file.endswith('.zip'):
      open_func = zipfile.Zipfile
      kwargs = {}
    elif self.Albert_file.endswith('.gz'):
      open_func = gzip.open
      kwargs = {}
    elif self.Albert_file.endswith('.xz'):
      open_func = lzma.open
      kwargs = {'errors': 'ignore'}
    else:
      open_func = codecs.open
      kwargs = {'errors': 'ignore'}
    
    cur_idx = len(self.special_tokens)
    tokens = []
    # Determine the dimensions of the embedding matrix
    with open_func(self.Albert_file, 'rb') as f:
      reader = codecs.getreader('utf-8')(f, **kwargs)
      first_line = reader.readline().rstrip().split(' ')
      if len(first_line) == 2: # It has a header that gives the dimensions
        has_header = True
        shape = [int(first_line[0])+cur_idx, int(first_line[1])]
      else: # We have to compute the dimensions ourself
        has_header = False
        for line_num, line in enumerate(reader):
          pass
        shape = [cur_idx+line_num+1, len(line.split())-1]
      shape[0] = min(shape[0], max_embed_count+cur_idx) if max_embed_count else shape[0]
      embeddings = np.zeros(shape, dtype=np.float32)
      
      # Fill in the embedding matrix
      #with open_func(self.Albert_file, 'rt', encoding='utf-8') as f:
      with open_func(self.Albert_file, 'rb') as f:
        for line_num, line in enumerate(f):
          if line_num:
            if cur_idx < shape[0]:
              line = line.rstrip()
              if line:
                line = line.decode('utf-8', errors='ignore').split(' ')
                embeddings[cur_idx] = line[1:]
                tokens.append(line[0])
                self[line[0]] = cur_idx
                cur_idx += 1
            else:
              break
    #pdb.set_trace()
    self._embed_size = shape[1]
    self._tokens = tokens
    self._embeddings = embeddings
    self.dump()
    return True

  #=============================================================
  def dump(self):
    if self.save_as_pickle and not os.path.exists(self.vocab_loadname):
      os.makedirs(os.path.dirname(self.vocab_loadname), exist_ok=True)
      with open(self.vocab_loadname, 'wb') as f:
        pkl.dump((self._tokens, self._embeddings), f, protocol=pkl.HIGHEST_PROTOCOL)
    return

  #=============================================================
  def load(self):
    """"""
    return True
    if self.vocab_loadname and os.path.exists(self.vocab_loadname):
      vocab_filename = self.vocab_loadname
    else:
      self._loaded = False
      return False

    with open(vocab_filename, 'rb') as f:
      self._tokens, self._embeddings = pkl.load(f, encoding='utf-8', errors='ignore')
    cur_idx = len(self.special_tokens)
    for token in self._tokens:
      self[token] = cur_idx
      cur_idx += 1
    self._embedding_size = self._embeddings.shape[1]
    self._loaded = True
    return True

  #=============================================================
  @property
  def Albert_file(self):
    return self._config.getstr(self, 'Albert_file')
  @property
  def vocab_loadname(self):
    return self._config.getstr(self, 'vocab_loadname')
  @property
  def name(self):
    return self._name
  @property
  def max_embed_count(self):
    return self._config.getint(self, 'max_embed_count')
  @property
  def embeddings(self):
    return self._embeddings
  @property
  def embed_keep_prob(self):
    return self._config.getfloat(self, 'max_embed_count')
  @property
  def embed_size(self):
    return self._embed_size
  @property
  def save_as_pickle(self):
    return self._config.getboolean(self, 'save_as_pickle')
  @property
  def linear_size(self):
    return self._config.getint(self, 'linear_size')
  @property
  def is_training(self):
    return self._config.getboolean(self, 'is_training')
  @property
  def get_albert_path(self):
    return self._config.getstr(self, 'albert_path')
  @property
  def strategy(self):
    return self._config.getstr(self, 'strategy')
  @property
  def layer_index(self):
    return self._config.getint(self, 'layer_index')
  @property
  def get_pretrained(self):
    try:
      return self._config.getboolean(self, 'use_pretrained_file')
    except:
      return False
  @property
  def get_pretrained_albert_path(self):
    return self._config.getstr(self, 'albert_pretrained_file_path')
  @property
  def hidden_func(self):
    hidden_func = self._config.getstr(self, 'hidden_func')
    if hasattr(nonlin, hidden_func):
      return getattr(nonlin, hidden_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
#***************************************************************
class FormAlbertVocab(AlbertVocab, cv.FormVocab):
  pass
# class LemmaAlbertVocab(AlbertVocab, cv.LemmaVocab):
#   pass
# class UPOSAlbertVocab(AlbertVocab, cv.UPOSVocab):
#   pass
# class XPOSAlbertVocab(AlbertVocab, cv.XPOSVocab):
#   pass
class DepheadAlbertVocab(AlbertVocab, cv.DepheadVocab):
  def __init__(self, Albert_file=None, name=None, config=None, is_eval=False):
    super(DepheadAlbertVocab, self).__init__(config=config, is_eval=is_eval)  
    # self.placeholder = {'features':tf.placeholder(tf.float32, [None,None,1024], name=self.classname+'features'),\
    #   'indices':tf.placeholder(tf.int32, [None,None], name=self.classname+'indices')}
    self.placeholder = tf.placeholder(tf.float32, [None,None,1024], name=self.classname+'features')
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    feed_dict[self.placeholder] = indices
    return feed_dict
  def get_input_tensor(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    #pdb.set_trace()
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    #pdb.set_trace()
    outputs=self.placeholder
    with tf.variable_scope('dephead_albert_vocab'):
      layer=classifiers.hidden(outputs,self.linear_size,hidden_func=self.hidden_func)
    return layer