import tensorflow as tf
from model.modules.attention import *
from model.modules.decoder import *
from model.modules.encoder import *
from model.modules.input_mask import *
from model.modules.layers import *
from model.modules.loss import *
from model.modules.optimizer import *
from model.modules.positional_encoding import *
from model.utils import AttrDict,ValueWindow
from model.model import Speech_transformer as Transformer
import yaml,os,time
from tensorflow.python.ops import summary_ops_v2
from datasets.datafeeder import DataFeeder

def main():

    configfile = open('./config/hparams_transcriber.yaml')
    config = AttrDict(yaml.load(configfile,Loader=yaml.FullLoader))

    log_name = config.model.name
    log_folder = os.path.join(os.getcwd(),'logdir/logging',log_name)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    # logger = init_logger(log_folder+'/'+train.log)

    train_datafeeder = DataFeeder(config,'debug')

    global global_step
    global_step = 0
    learning_rate = CustomSchedule(config.model.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=config.optimizer.beta1, beta_2=config.optimizer.beta2,
                                         epsilon=config.optimizer.epsilon)
    print('config.optimizer.beta1:' + str(config.optimizer.beta1))
    print('config.optimizer.beta2:' + str(config.optimizer.beta2))
    print('config.optimizer.epsilon:' + str(config.optimizer.epsilon))
    
    model = Transformer(config=config)

    #Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every n epochs.
    checkpoint_path = log_folder
    ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    else:
        print('Start new run')


    # define metrics and summary writer
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # summary_writer = tf.keras.callbacks.TensorBoard(log_dir=log_folder)
    summary_writer = summary_ops_v2.create_file_writer_v2(log_folder+'/train')


    # @tf.function
    def train_step(batch_data):
        inp = batch_data['the_inputs'] # batch*time*feature
        tar = batch_data['the_labels'] # batch*time
        # inp_len = batch_data['input_length']
        # tar_len = batch_data['label_length']
        gtruth = batch_data['ground_truth']
        tar_inp = tar
        tar_real = gtruth
        # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp[:,:,0], tar_inp)
        combined_mask = create_combined_mask(tar=tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = model(inp, tar_inp, True, None,
                                   combined_mask, None)
            loss = LableSmoothingLoss(tar_real, predictions,config.model.vocab_size,config.train.label_smoothing_epsilon)
        gradients = tape.gradient(loss, model.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    acc_window = ValueWindow(100)
    print('config.train.epoches:' + str(config.train.epoches))
    first_time = True
    for epoch in range(config.train.epoches):
        print('start epoch '+ str(epoch))
        print('total wavs: '+ str(len(train_datafeeder)))
        print('batch size: ' + str(train_datafeeder.batch_size))
        print('batch per epoch: ' + str(len(train_datafeeder)//train_datafeeder.batch_size))
        train_data = train_datafeeder.get_batch()
        start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for step in range(len(train_datafeeder)//train_datafeeder.batch_size):
            batch_data = next(train_data)
            step_time = time.time()
            train_step(batch_data)
            if first_time:
                model.summary()
                first_time=False
            time_window.append(time.time()-step_time)
            loss_window.append(train_loss.result())
            acc_window.append(train_accuracy.result())
            message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, acc=%.05f, avg_acc=%.05f]' % (
                    global_step, time_window.average, train_loss.result(), loss_window.average, train_accuracy.result(),acc_window.average)
            print(message)

            if global_step % 10 == 0:
                with summary_ops_v2.always_record_summaries():
                    with summary_writer.as_default():
                        summary_ops_v2.scalar('train_loss', train_loss.result(), step=global_step)
                        summary_ops_v2.scalar('train_acc', train_accuracy.result(), step=global_step)

            global_step += 1

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start_time))

if __name__=='__main__':
    main()