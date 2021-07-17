import struct
import tensorflow as tf


def print_fc(f, f_d, w, b):
    f.write(str(w.shape[0])+','+str(w.shape[1])+'\n')
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            f_d.write(struct.pack('f', w[i, j]))

    f.write(str(b.shape[0])+'\n')
    for i in range(b.shape[0]):
        f_d.write(struct.pack('f', b[i]))

class Phi():
    def __init__(self, out_dim, in_dim, init_width, mid_width, no_layers, name='base'):
        self.no_layers = no_layers

        input = tf.keras.Input(shape=(in_dim,))
        output = tf.keras.layers.Dense(init_width, activation=tf.nn.relu)(input)
        for i in range(no_layers):
            output = tf.keras.layers.Dense(mid_width, activation=tf.nn.relu)(output)
        output = tf.keras.layers.Dense(out_dim)(output)
        self.phi = tf.keras.Model(inputs=input, outputs=output)
        self.phi.compile()


    def save_params(self, file_name):
        weights = self.phi.get_weights()
        with open(file_name, 'w') as f:
            with open(file_name+"d", 'wb') as f_d:
                no_layers = self.no_layers+2
                if self.no_layers == 0:
                    no_layers = 1
                f.write(str(no_layers)+'\n')
                for i in range(no_layers):
                    print_fc(f, f_d, weights[2*(i)], weights[2*(i)+1])
        return

    def call(self, x, p=None):
        return self.phi(x)

