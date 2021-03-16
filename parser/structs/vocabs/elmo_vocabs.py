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

class ElmoVocab(SetVocab):
  """"""
  
  #=============================================================
  def __init__(self, Elmo_file=None, name=None, config=None, is_eval=False):
    """"""
    if (Elmo_file is None) != (name is None):
      raise ValueError("You can't pass in a value for only one of Elmo_file and name to ElmoVocab.__init__")
    super(ElmoVocab, self).__init__(config=config, is_eval=is_eval)
    self._Elmo_file = Elmo_file
    self._name = name
    self.variable = None
    self.pretrained = self.get_pretrained
    if not self.pretrained:
      self.placeholder = {'token_ph':tf.placeholder(tf.string, [None, None], name=self.classname+'token_ph'),\
      'len_ph':tf.placeholder(tf.int32, [None], name=self.classname+'len_ph')}
      self.elmo_emb = self.start_elmo(is_eval)
    else:
      self.placeholder = tf.placeholder(tf.float32, [None,None,1024], name=self.classname+'input_feats')

    return

  def start_elmo(self, is_eval=False):
    import tensorflow_hub as hub
    
    #default: "https://tfhub.dev/google/elmo/2"
    elmo_path = self.get_elmo_path
    
    # input_ids_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='input_ids_p')
    # input_mask_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='input_mask_p')
    # segment_ids_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='segment_ids_p')
    token_ph=self.placeholder['token_ph']
    len_ph=self.placeholder['len_ph']
    
    elmo_model = hub.Module(elmo_path, trainable=True, name='elmo')

    lm_embeddings = elmo_model(
      inputs={"tokens": token_ph, "sequence_len": len_ph},
      signature="tokens", as_dict=True)
    lm_emb = lm_embeddings["elmo"]

    return lm_emb

  def modelInit(self,sess):
    pass
    
  def modelRestore(self,sess,elmolist,model_dir=None):
    pass

  def modelSave(self,sess,save_dir,global_step):
    pass

  #=============================================================
  def get_input_tensor(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    #pdb.set_trace()
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    #pdb.set_trace()
    if self.pretrained:
      outputs=self.placeholder
    else:
      outputs = self.elmo_emb
    with tf.variable_scope('elmo_vocab'):
      layer=classifiers.hidden(outputs,self.linear_size,hidden_func=self.hidden_func)
    return layer


  #=============================================================
  def get_embedding(self, embed_keep_prob=None, variable_scope=None, reuse=True):
    """"""
    #pdb.set_trace()
    embed_keep_prob = embed_keep_prob or self.embed_keep_prob
    #pdb.set_trace()
    outputs = self.elmo_emb
    return outputs
  
  #=============================================================
  def set_placeholders(self, indices, feed_dict={}):
    """"""
    #pdb.set_trace()
    if not self.pretrained:
      feed_dict[self.placeholder['token_ph']] = indices['token_ph']
      feed_dict[self.placeholder['len_ph']] = indices['len_ph']
    else:
      feed_dict[self.placeholder] = indices
    return feed_dict
  def add(self, token):
    return token
  #=============================================================
  def count(self, *args):
    """"""
    return True
    max_embed_count = self.max_embed_count
    if self.Elmo_file.endswith('.zip'):
      open_func = zipfile.Zipfile
      kwargs = {}
    elif self.Elmo_file.endswith('.gz'):
      open_func = gzip.open
      kwargs = {}
    elif self.Elmo_file.endswith('.xz'):
      open_func = lzma.open
      kwargs = {'errors': 'ignore'}
    else:
      open_func = codecs.open
      kwargs = {'errors': 'ignore'}
    
    cur_idx = len(self.special_tokens)
    tokens = []
    # Determine the dimensions of the embedding matrix
    with open_func(self.Elmo_file, 'rb') as f:
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
      #with open_func(self.Elmo_file, 'rt', encoding='utf-8') as f:
      with open_func(self.Elmo_file, 'rb') as f:
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
  
  #=============================================================
  @property
  def Elmo_file(self):
    return self._config.getstr(self, 'Elmo_file')
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
  def get_elmo_path(self):
    return self._config.getstr(self, 'elmo_path')
  @property
  def get_pretrained(self):
    try:
      return self._config.getboolean(self, 'use_pretrained_file')
    except:
      return False
  @property
  def get_pretrained_elmo_path(self):
    return self._config.getstr(self, 'elmo_pretrained_file_path')
  @property
  def hidden_func(self):
    hidden_func = self._config.getstr(self, 'hidden_func')
    if hasattr(nonlin, hidden_func):
      return getattr(nonlin, hidden_func)
    else:
      raise AttributeError("module '{}' has no attribute '{}'".format(nonlin.__name__, hidden_func))
#***************************************************************
class FormElmoVocab(ElmoVocab, cv.FormVocab):
  pass
# class LemmaElmoVocab(ElmoVocab, cv.LemmaVocab):
#   pass
# class UPOSElmoVocab(ElmoVocab, cv.UPOSVocab):
#   pass
# class XPOSElmoVocab(ElmoVocab, cv.XPOSVocab):
#   pass
class DepheadElmoVocab(ElmoVocab, cv.DepheadVocab):
  def __init__(self, Elmo_file=None, name=None, config=None, is_eval=False):
    super(DepheadElmoVocab, self).__init__(config=config)  
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
    with tf.variable_scope('dephead_elmo_vocab'):
      layer=classifiers.hidden(outputs,self.linear_size,hidden_func=self.hidden_func)
    return layer