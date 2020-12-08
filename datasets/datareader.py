import os, sys
sys.path.append('..')
from model.utils import AttrDict,ValueWindow
import yaml
import tensorflow as tf

class DataReader(object):

    def __init__(self, config):
        self.corpus = config.data.corpus_path
        self.languages = config.data.languages
        self.src_eg = os.path.join(self.corpus, self.languages, config.data.source_train)
        self.src_val = os.path.join(self.corpus, self.languages, config.data.source_dev)
        self.tgt_eg = os.path.join(self.corpus, self.languages, config.data.target_train)
        self.tgt_val = os.path.join(self.corpus, self.languages, config.data.target_dev)
        
    def read_files(self, path):
        with open(path,'r') as f:
            line = f.readline()
            data = []
            while line:
                data.append(line)
                line = f.readline()
        return data
    
    def __call__(self):
        src_egdata = self.read_files(self.src_eg)
        tgt_egdata = self.read_files(self.tgt_eg)
        src_valdata = self.read_files(self.src_val)
        tgt_valdata = self.read_files(self.tgt_val)

        train = tf.data.Dataset.from_tensor_slices((tgt_egdata, src_egdata))
        val = tf.data.Dataset.from_tensor_slices((tgt_valdata, src_valdata))

        return train, val

if __name__ == "__main__":
    configfile = open('../config/hparams_translator.yaml')
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    dr = DataReader(config)
    train, val = dr()    
    for i, j in train:
        print(i.numpy())
        print(j.numpy())
        break

