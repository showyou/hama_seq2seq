#import math
import os
#import random
#import sys
#import time

#import tensorflow.python.platform

import numpy as np
import tensorflow as tf

import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
#from tensorflow.python.platform import gfile
import MeCab

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 100000, "input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 100000, "output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./datas", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./datas", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# seq2seqで反応を試みます

class Decoder():
    def __init__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        model = self.create_model(self.sess, True)
        model.batch_size = 1

        in_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab_in.txt")
        out_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab_out.txt" )

        self.in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)
        _, self.rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)

    def __del__(self):
        self.sess.close()

    def create_model(self, session, forward_only):
        self.model = seq2seq_model.Seq2SeqModel(
          FLAGS.in_vocab_size, FLAGS.out_vocab_size, _buckets,
          FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
          forward_only=forward_only)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        #if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            #add
        if ckpt and not os.path.isabs(ckpt.model_checkpoint_path):
            ckpt.model_checkpoint_path= os.path.abspath(os.path.join(os.getcwd(), ckpt.model_checkpoint_path))
            #so far
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
        return self.model

    def wakati(self, input_str):
        '''分かち書き用関数
        引数 input_str : 入力テキスト
        返値 m.parse(wakatext) : 分かち済みテキスト'''
        wakatext = input_str
        m = MeCab.Tagger('-Owakati')
        #print(m.parse(wakatext))
        return m.parse(wakatext)

    def decode(self,text):

        sentence = self.wakati(text)

        try:
            token_ids = data_utils.sentence_to_token_ids(sentence, self.in_vocab)

            bucket_id = min([b for b in range(len(_buckets))
                           if _buckets[b][0] > len(token_ids)])

            encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)

            _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)

            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]

            print("> ","".join([self.rev_out_vocab[output] for output in outputs]))
        except Exception as ex:
            print(type(ex), end="")
            print(ex.args)
