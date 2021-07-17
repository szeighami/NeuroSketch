import tensorflow as tf
import os
physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
from utils import mse, PrintEpochNo, MAE, SaveBestModelCallBack
from base_model import Phi 
import json
import sys
import math
import numpy as np
import pandas as pd


f = open('conf.json') 
config = json.load(f) 
no = int(sys.argv[1])
base_name = sys.argv[2]
path = sys.argv[3]

test = np.load('test'+str(no)+'_queries.npy')
test_res = np.load('test'+str(no)+'_res.npy').astype(float)
train = np.load('queries'+str(no)+'.npy')
res = np.load('res'+str(no)+'.npy').astype(float)

model = Phi(config['out_dim'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'])

base_lr = config['lr']
min_lr = config['min_lr']
decay_factor = 3
times_to_decay = math.log(base_lr/min_lr)/math.log(decay_factor)
decay_freq = config['EPOCHS']//times_to_decay
def schedule(epoch):
    lr = base_lr/(decay_factor**(epoch//decay_freq))
    return lr
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=base_name+path, save_weights_only=True, monitor='val_mae', mode='min', save_best_only=True)
callbacks = [PrintEpochNo(config["print_freq"]), model_checkpoint_callback]
metrics = [MAE()]
optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
loss = mse

model.phi.compile(optimizer, loss=loss, metrics=metrics)
h = model.phi.fit(train, res, epochs=config['EPOCHS'], batch_size=train.shape[0]//config['batch_size'], callbacks=callbacks, validation_data=(test, test_res), validation_steps=1, verbose=0, shuffle=False)

hist_df = pd.DataFrame(h.history) 
with open(base_name+str(no)+'_hist.json', 'w') as f:
    hist_df.to_json(f)

model.phi.load_weights(base_name+path)
model.save_params(base_name+path+".m")
