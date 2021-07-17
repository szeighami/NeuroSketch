import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras

class PrintEpochNo(tf.keras.callbacks.Callback):
    def __init__(self, print_freq, **kwargs):
        self.print_freq = print_freq
    def on_epoch_end(self, epoch, logs={}):
        if  (epoch+1)%self.print_freq == 0:
            tf.print("Epoch no " + str(epoch+1)+ " loss " + str(logs.get('loss'))  + " val_loss " + str(logs.get('val_loss')) + " mae " + str(logs.get('mae'))+ " val_mae " + str(logs.get('val_mae')), output_stream=sys.stdout)


class SaveBestModelCallBack(tf.keras.callbacks.Callback):
    #def __init__(self, test, test_res, model, save_path, **kwargs):
    def __init__(self, model, save_path, **kwargs):
        #self.test = test
        #self.test_res = test_res
        self.min_err = 1000000
        self.model = model
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        #y_pred = tf.cast(self.model.call(self.test), tf.float64)
        #y_true = tf.cast(self.test_res, tf.float64)
        #err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((y_true-y_pred)**2, axis=1)))

        if logs.get('val_mae') < self.min_err:
            #self.model.save_params(self.save_path)
            self.model.phi.save(self.save_path)
            self.min_err = logs.get('val_mae')

class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae', **kwargs):
        super(MAE, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        err = tf.math.abs(y_pred-y_true)

        self.accuracy.assign_add(tf.reduce_mean(tf.cast(err, tf.float64)))
        self.count.assign_add(1)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.accuracy.assign(0.)
      self.count.assign(0.)

def mse(y_true, y_pred):
    return  K.mean((K.sum(K.square(y_true-y_pred), axis=-1)))

