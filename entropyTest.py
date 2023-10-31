import os
import numpy as np
import pacmap
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import tensorflow as tf
from sklearn.utils import shuffle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import sentencepiece as spm
from IPython.display import clear_output
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
import pickle

from tensorflow.keras.utils import plot_model
import subprocess
import sys
import time
from termcolor import colored, cprint
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm
import pacmap
from random import shuffle
import sys

TEST = sys.argv[1]
def loadData(typ, name):
    ret = None
    with open('./data/' +  name + "_" + typ + '.pkl', 'rb') as f:
        ret = pickle.load(f)
    assert ret is not None
    return ret

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        #ffn_output = self.ffn(inputs + attn_output)
        
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        #return (ffn_output)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'att': self.att,
            'ffn': self.ffn
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        print(config)
        return cls(**config)

#TEST="100k_rnd"
ENT_RUNS=10
pathPred = "./out" + TEST

d = loadData("combined", "all")  # numeric, weights
inSize = len(d["numeric"][0])
outSize = len(d["weights"][0])
print(f"Input Size: {inSize}, Output Size: {outSize}")
print(f"Corpus count: {len(d['numeric'])}")

test = keras.models.load_model("./data/" + TEST + "_transformer.hdf5"
     , custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding
     , "TransformerBlock": TransformerBlock})

transformerOnly_model = keras.Model(test.inputs, test.layers[3].output)

#transformerOnly_model = keras.Model(model.inputs, model.get_layer("transformer_block_1").output)
transformerOnly_model.summary()


tfOut = transformerOnly_model.predict(d["numeric"])
tfOut.shape

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def get_entropy(X):
   if len(X.shape)==1:
       X=X.reshape(-1,1)
   params = {'bandwidth': np.logspace(-10, 10, 20)}
   gs = GridSearchCV(KernelDensity(), params)
   gs.fit(X)
   kde=gs.best_estimator_
   log_probs=kde.score_samples(X)
   return -np.mean(log_probs)

shuffle(tfOut)

#e = get_entropy(tfOut[:1000])
kde = KernelDensity(bandwidth=0.5).fit(tfOut)

log_p = kde.score_samples(tfOut[:10000])  # returns log(p) of data sample
p = np.exp(log_p)                # estimate p of data sample
entropy = -np.sum(p*log_p)       # evaluate entropy
print(f"ENTROPY {TEST}: {entropy}")

