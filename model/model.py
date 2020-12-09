import sys
sys.path.append('./model')
from transformer import Transformer
import tensorflow as tf
from modules.attention import Pre_Net
from modules.input_mask import create_combined_mask
import numpy as np
from utils import  AttrDict
import yaml

class Speech_transformer(tf.keras.Model):
    def __init__(self,config):
        super(Speech_transformer, self).__init__()
        self.pre_net  = Pre_Net(config.model.num_M,config.model.n,config.model.c)
        self.transformer = Transformer(config=config)

    def call(self, inputs, targets, training, enc_padding_mask,
            look_ahead_mask, dec_padding_mask):

        out = self.pre_net(inputs,training)

        final_out,attention_weights = self.transformer((out,targets) ,training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask)

        return final_out,attention_weights

class Text_transformer(tf.keras.Model):
    def __init__(self, config):
        super(Text_transformer, self).__init__()

        self.transformer = Transformer(config=config)

        self.final_layer = tf.keras.layers.Dense(config.model.vocab_size)

    def call(self, inputs, targets, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask):

        dec_output, attention_weights = self.transformer((inputs, targets), training, enc_padding_mask,
            look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


if __name__=='__main__':
    configfile = open('./Speech_Transformer/config/hparams_transcriber.yaml')
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    print(config.data_name)
    inputs = np.random.randn(32,233,80,3)
    targets = np.random.randint(0,31,[32,55])
    combined_mask = create_combined_mask(targets)

    st = Speech_transformer(config,None)
    final_out, attention_weights = st(inputs,targets,True,None,combined_mask,None)

    print('speech final_out.shape:', final_out.shape)
    print('speech final_out:', final_out)

    tt = Text_transformer(config, None)
    final_out, attention_weights = tt(inputs, targets, True, None, combined_mask, None)
    print ('text final_out.shape:', final_out.shape)
    print('text final_out:', final_out)

