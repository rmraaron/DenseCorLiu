import tensorflow as tf
import numpy as np


def variable_on_cpu(name, shape, initializer, use_fp16=False):
    with tf.device('/device:cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.compat.v1.get_variable(name, shape, dtype=dtype, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    if use_xavier:
        initializer = tf.initializers.GlorotNormal()
    else:
        initializer = tf.initializers.TruncatedNormal(stddev=stddev)
    var = variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_decay)
    return var


def batch_norm_for_conv2d(inputs, is_training, moments_dims, bn_decay, scope):
    with tf.compat.v1.variable_scope(scope) as sc:
        depth = inputs.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def conv2d(inputs, output_channels, kernel_size, scope,
           stride=[1, 1], padding='SAME', use_xavier=True,
           stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
           bn=False, bn_decay=None, is_training=None):
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        in_channels = inputs.get_shape()[-1]
        kernel_shape = [kernel_h, kernel_w, in_channels, output_channels]
        kernel = variable_with_weight_decay('weights', shape=kernel_shape, stddev=stddev,
                                            wd=weight_decay, use_xavier=use_xavier)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel, [1, stride_h, stride_w, 1], padding=padding)
        biases = variable_on_cpu('biases', [output_channels], tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

    if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training, moments_dims=[0, 1, 2], bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs, kernel_size, scope, stride=[2, 2], padding='VALID'):
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs, [1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding, name=sc.name)
        return outputs


def full_connected(inputs, num_outputs, scope, use_xavier=True, stddev=1e-3,
                   weight_decay=0.0, activation_fn=tf.nn.relu):
    with tf.compat.v1.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1]
        weights = variable_with_weight_decay('weights', shape=[num_input_units, num_outputs],
                                             use_xavier=use_xavier, stddev=stddev, wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = variable_on_cpu('biases', [num_outputs], tf.constant_initializer(0.0))
        outputs = tf.add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def placeholder_inputs(batch_size, num_points):
    pointclouds = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_points, 3))
    return pointclouds


def get_model(point_cloud, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    net = conv2d(input_image, 64, [1, 3], scope='conv1', stride=[1, 1],
                 padding='VALID', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = conv2d(net, 64, [1, 1], scope='conv2', stride=[1, 1],
                 padding='VALID', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = conv2d(net, 64, [1, 1], scope='conv3', stride=[1, 1],
                 padding='VALID', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = conv2d(net, 128, [1, 1], scope='conv4', stride=[1, 1],
                 padding='VALID', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = conv2d(net, 1024, [1, 1], scope='conv5', stride=[1, 1],
                 padding='VALID', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = max_pool2d(net, [num_point, 1], scope='maxpool', padding='VALID')

    net = tf.reshape(net, [batch_size, -1])

    f_id = full_connected(net, 512, scope='fc1_parallel', activation_fn=None)

    f_exp = full_connected(net, 512, scope='fc2_parallel', activation_fn=None)

    d_id_net = full_connected(f_id, 1024, scope='fc_de_id')

    d_exp_net = full_connected(f_exp, 1024, scope='fc_de_exp')

    s_id = full_connected(d_id_net, num_point * 3, scope='fc_shape_id', activation_fn=None)

    s_exp = full_connected(d_exp_net, num_point * 3, scope='fc_shape_exp', activation_fn=None)

    return s_id, s_exp, end_points


# def get_loss():


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 29495, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)