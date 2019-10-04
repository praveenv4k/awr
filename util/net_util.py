import tensorflow.compat.v1 as tf
import tensorflow as tf2


def build_fc_net(input_tfs, layers,
                  activation=tf.nn.relu,
                  # weight_init=tf.contrib.layers.xavier_initializer(),
                  weight_init=tf2.initializers.GlorotUniform(),
                  reuse=False):
    curr_tf = tf.concat(axis=-1, values=input_tfs)       
    for i, size in enumerate(layers):
        with tf.variable_scope(str(i), reuse=reuse):
            curr_tf = tf.layers.dense(inputs=curr_tf,
                                    units=size,
                                    kernel_initializer=weight_init,
                                    activation=activation)
    return curr_tf