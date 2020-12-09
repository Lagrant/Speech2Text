import tensorflow_datasets as tfds
import tensorflow as tf

import time, yaml
import numpy as np
import matplotlib.pyplot as plt

from model.modules.attention import *
from model.modules.decoder import *
from model.modules.encoder import *
from model.modules.input_mask import *
from model.modules.layers import *
from model.modules.loss import loss_function
from model.modules.optimizer import *
from model.modules.positional_encoding import *
from model.model import Text_transformer as Transformer
from model.utils import AttrDict

from datasets.datareader import DataReader


def main():
    configfile = open('./config/hparams_translator.yaml')
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    dr = DataReader(config)
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
        lambda x, y: tf.logical_and(tf.size(x) <= config.data.MAX_LENGTH, tf.size(y) <= config.data.MAX_LENGTH)
    )

    train_dataset = train_dataset.shuffle(config.data.BUFFER_SIZE).padded_batch(config.data.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(
        lambda x, y: tf.logical_and(tf.size(x) <= config.data.MAX_LENGTH, tf.size(y) <= config.data.MAX_LENGTH)
    ).padded_batch(config.data.BATCH_SIZE)

    src_batch, tgt_batch = next(iter(val_dataset))

    input_vocab_size = tokenizer_src.vocab_size + 2
    target_vocab_size = tokenizer_tgt.vocab_size + 2

    learning_rate = CustomSchedule(config.model.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=config.optimizer.beta_1, beta_2=config.optimizer.beta_2, 
                                        epsilon=config.optimizer.epsilon)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    
    config.model.pe_max_len = input_vocab_size
    config.model.vocab_size = target_vocab_size
    transformer = Transformer(config)
    
    checkpoint_path = "./logdir/logging/T_Transformer/checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                              optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    
    # 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
    # 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
    # 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
    # 更多的通用形状。
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
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
    
    return tokenizer_tgt, tokenizer_src, config.data.MAX_LENGTH

    # def point_wise_feed_forward_network(d_model, dff):
    #   return tf.keras.Sequential([
    #       tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
    #       tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    #   ])

def translate(sentence, tokenizer_tgt, tokenizer_src, MAX_LENGTH, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_tgt.decode([i for i in result 
                                              if i < tokenizer_tgt.vocab_size])  

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)

    def evaluate(inp_sentence):
        start_token = [tokenizer_src.vocab_size]
        end_token = [tokenizer_src.vocab_size + 1]

        # 输入语句是葡萄牙语，增加开始和结束标记
        inp_sentence = start_token + tokenizer_src.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # 因为目标是英语，输入 transformer 的第一个词应该是
        # 英语的开始标记。
        decoder_input = [tokenizer_tgt.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
            # 从 seq_len 维度选择最后一个词
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # 如果 predicted_id 等于结束标记，就返回结果
            if predicted_id == tokenizer_tgt.vocab_size+1:
                return tf.squeeze(output, axis=0), attention_weights
            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = tokenizer_src.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head+1)

            # 画出注意力权重
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence)+2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result)-1.5, -0.5)

            ax.set_xticklabels(
                ['<start>']+[tokenizer_src.decode([i]) for i in sentence]+['<end>'], 
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([tokenizer_tgt.decode([i]) for i in result 
                              if i < tokenizer_tgt.vocab_size], 
                              fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head+1))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tokenizer_tgt, tokenizer_src, MAX_LENGTH = main()

    translate("este é um problema que temos que resolver.", 
        tokenizer_tgt, tokenizer_src, MAX_LENGTH
    )
    print ("Real translation: this is a problem we have to solve .")

    translate("os meus vizinhos ouviram sobre esta ideia.", 
        tokenizer_tgt, tokenizer_src, MAX_LENGTH
    )
    print ("Real translation: and my neighboring homes heard about this idea .")

    translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.", 
        tokenizer_tgt, tokenizer_src, MAX_LENGTH
    )
    print ("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")

    translate("este é o primeiro livro que eu fiz.", 
        tokenizer_tgt, tokenizer_src, MAX_LENGTH, 
        plot='decoder_layer4_block2'
    )
    print ("Real translation: this is the first book i've ever done.")