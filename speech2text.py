import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.append('./model')
from model.transformer import Transformer
from model.modules.attention import Pre_Net
from model.modules.input_mask import create_combined_mask
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from datasets.datareader import DataReader
from datasets.datafeeder import DataFeeder
import numpy as np
from utils import  AttrDict
import yaml

class S2T(tf.keras.Model):
    def __init__(self,config):
        super(S2T, self).__init__()
        config_spe = config[0]
        config_txt = config[1]

        num_enc_layers_spe = config_spe.model.N_encoder
        d_model_spe = config_spe.model.d_model
        num_heads_spe = config_spe.model.n_heads
        dff_spe = config_spe.model.d_ff
        pe_max_len_spe = config_spe.model.pe_max_len
        rate_spe = config_spe.model.dropout
        
        num_enc_layers_txt = config_txt.model.N_encoder
        num_dec_layers_txt = config_txt.model.N_decoder
        d_model_txt = config_txt.model.d_model
        num_heads_txt = config_txt.model.n_heads
        dff_txt = config_txt.model.d_ff
        pe_max_len_txt = config_txt.model.pe_max_len
        target_vocab_size_txt = config_txt.model.vocab_size
        rate_txt = config_txt.model.dropout
      

        self.pre_net  = Pre_Net(config_spe.model.num_M,config_spe.model.n,config_spe.model.c)

        self.encoder_spe = Encoder(num_enc_layers_spe, d_model_spe, num_heads_spe,
            dff_spe, pe_max_len_spe, 'encoder', rate_spe, True)
        
        self.encoder_txt = Encoder(num_enc_layers_txt, d_model_txt, num_heads_txt,
            dff_txt, pe_max_len_txt, 'encoder', rate_txt, False)
        
        self.decoder = Decoder(num_dec_layers_txt, d_model_txt, num_heads_txt,
            dff_txt, target_vocab_size_txt, 'decoder', pe_max_len_txt, rate_txt)
        
        self.final_layer = tf.keras.layers.Dense(config_txt.model.vocab_size)
    
    def call(self, inputs, targets, training, enc_padding_mask_spe, 
            enc_padding_mask_txt, look_ahead_mask, dec_padding_mask):

        inputs = tf.cast(inputs, tf.float32)
        targets = tf.cast(targets, tf.int32)

        pre = self.pre_net(inputs,training)

        spe_output = self.encoder_spe((inputs, enc_padding_mask_spe), training)

        txt_output = self.encoder_txt((spe_output, enc_padding_mask_txt), training)

        dec_output, attention_weights = self.decoder((targets, txt_output, look_ahead_mask,
            dec_padding_mask), training)

        final_output = dec_output

        return final_output, attention_weights

"""
The overall audio corpus is over 68GB which is not affordable
for this project. 
Due to the size of corpus and limitation of computing resource, 
we fail to train the integrated model. Here we just show the code.

Links of two overall corpora that we planned to use:
https://persyval-platform.univ-grenoble-alpes.fr/DS91/detaildataset
https://ict.fbk.eu/must-c/
"""
if __name__ == "__main__":
    configfile = open('./config/hparams_transcriber.yaml')
    config_spe = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    configfile = open('./config/hparams_translator.yaml')
    config_txt = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    # we replace the overall corpora with our simplified corpora
    dr = DataReader(config_txt)
    train_examples, val_examples = dr()
    tokenizer_tgt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (tgt.numpy() for src, tgt in train_examples), target_vocab_size=2**13)
    tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (src.numpy() for src, tgt in train_examples), target_vocab_size=2**13)

    def encode(lang1, lang2):
        lang1 = [tokenizer_src.vocab_size] + tokenizer_src.encode(lang1.numpy()) + [tokenizer_src.vocab_size + 1]
        lang2 = [tokenizer_tgt.vocab_size] + tokenizer_tgt.encode(lang2.numpy()) + [tokenizer_tgt.vocab_size + 1]
        return lang1, lang2

    def tf_encode(src, tgt):
        result_src, result_tgt = tf.py_function(encode, [src, tgt], [tf.int64, tf.int64])
        result_src.set_shape([None])
        result_tgt.set_shape([None])
        return result_src, result_tgt

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(
        lambda x, y: tf.logical_and(tf.size(x) <= config_txt.data.MAX_LENGTH, tf.size(y) <= config_txt.data.MAX_LENGTH)
    )
    train_dataset = train_dataset.shuffle(config_txt.data.BUFFER_SIZE).padded_batch(config_txt.data.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    input_vocab_size = tokenizer_src.vocab_size + 2
    target_vocab_size = tokenizer_tgt.vocab_size + 2

    learning_rate = CustomSchedule(config_txt.model.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=config_txt.optimizer.beta_1, beta_2=config_txt.optimizer.beta_2, 
                                        epsilon=config_txt.optimizer.epsilon)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    
    config_txt.model.pe_max_len = input_vocab_size
    config_txt.model.vocab_size = target_vocab_size

    df = DataFeeder(config_spe,'debug')
    spe_batch = df.get_batch()

    transformer = S2T((config_spe, config_txt))

    checkpoint_path = "./logdir/Transformer/checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer,
                              optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    
    def train_step(inp, tar):
        spe_batch_data = next(spe_batch)
        inp = spe_batch_data['the inputs']
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, 
                                        True, 
                                        enc_padding_mask, 
                                        combined_mask, 
                                        dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)
    
    for epoch in range(config.train.EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                      epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    


