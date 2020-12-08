import sys
sys.path.append('./model')
from transformer import Transformer
import tensorflow as tf
from modules.attention import Pre_Net
from modules.input_mask import create_combined_mask
from modules.encoder import Encoder
from modules.decoder import Decoder
import numpy as np
from utils import  AttrDict
import yaml

class Speech_transformer(tf.keras.Model):
    def __init__(self,config):
        super(Speech_transformer, self).__init__()
        self.pre_net  = Pre_Net(config.model.num_M,config.model.n,config.model.c)
        self.transformer = Transformer(config=config)

    def call(self,inputs,targets,training,enc_padding_mask,look_ahead_mask,dec_padding_mask):

        out = self.pre_net(inputs,training)

        final_out,attention_weights = self.transformer((out,targets) ,training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask)

        return final_out,attention_weights

class Text_transformer(tf.keras.Model):
    def __init__(self, config, input_vocab_size, 
                target_vocab_size, pe_input, pe_target):
        super(Text_transformer, self).__init__()

        self.encoder = Encoder(num_layers=config.model.num_layers, d_model=config.model.d_model, num_heads=config.model.num_heads, dff=config.model.dff, 
                            name='Encoder' , pe_max_len=pe_input, dp=config.model.rate, input_vocab_size=input_vocab_size)

        self.decoder = Decoder(num_layers=config.model.num_layers, d_model=config.model.d_model, num_heads=config.model.num_heads, dff=config.model.dff, 
                            target_vocab_size=target_vocab_size, name='decoder', pe_max_len=pe_target, rate=config.model.rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder((inp, enc_padding_mask), training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            (tar, enc_output, look_ahead_mask, dec_padding_mask), training)

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

    print('final_out.shape:',final_out.shape)
    print('final_out:',final_out)





