import tensorflow as tf
import data_preprosessing


def variable_on_gpu(name, shape, initializer, use_fp16=False, trainable=True):
    with tf.device('/device:gpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.compat.v1.get_variable(name, shape, dtype=dtype, initializer=initializer, trainable=trainable)
    return var


def variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True, trainable=True):
    if use_xavier:
        initializer = tf.initializers.GlorotNormal()
    else:
        initializer = tf.initializers.TruncatedNormal(stddev=stddev)
    var = variable_on_gpu(name, shape, initializer, trainable=trainable)
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
           bn=False, bn_decay=None, is_training=None, trainable=True):
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        in_channels = inputs.get_shape()[-1]
        kernel_shape = [kernel_h, kernel_w, in_channels, output_channels]
        kernel = variable_with_weight_decay('weights', shape=kernel_shape, stddev=stddev,
                                            wd=weight_decay, use_xavier=use_xavier, trainable=trainable)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel, [1, stride_h, stride_w, 1], padding=padding)
        biases = variable_on_gpu('biases', [output_channels], tf.constant_initializer(0.0), trainable=trainable)
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
                   weight_decay=0.0, activation_fn='ReLU', trainable=True):
    with tf.compat.v1.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1]
        weights = variable_with_weight_decay('weights', shape=[num_input_units, num_outputs],
                                             use_xavier=use_xavier, stddev=stddev, wd=weight_decay, trainable=trainable)
        outputs = tf.matmul(inputs, weights)
        biases = variable_on_gpu('biases', [num_outputs], tf.constant_initializer(0.0), trainable=trainable)
        outputs = tf.add(outputs, biases)

        if activation_fn is not None:
            if activation_fn is not 'Tanh':
                outputs = tf.nn.relu(outputs)
            else:
                outputs = tf.nn.tanh(outputs)

        return outputs


def placeholder_inputs(batch_size, num_points):
    pointclouds = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_points, 3))
    label_points = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_points, 3))
    faces_tri = tf.compat.v1.placeholder(tf.int32, shape=(58366, 3))
    return pointclouds, label_points, faces_tri


def get_model_encoder(point_cloud, is_training, bn_decay=None, trainable=True):

    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    net = conv2d(input_image, 64, [1, 3], scope='conv1', stride=[1, 1], padding='VALID',
                 bn=True, bn_decay=bn_decay, is_training=is_training, trainable=trainable)
    net1 = conv2d(net, 64, [1, 1], scope='conv2', stride=[1, 1], padding='VALID',
                  bn=True, bn_decay=bn_decay, is_training=is_training, trainable=trainable)
    net2 = conv2d(net1, 64, [1, 1], scope='conv3', stride=[1, 1], padding='VALID',
                  bn=True, bn_decay=bn_decay, is_training=is_training, trainable=trainable)
    net3 = conv2d(net2, 128, [1, 1], scope='conv4', stride=[1, 1], padding='VALID',
                  bn=True, bn_decay=bn_decay, is_training=is_training, trainable=trainable)
    net4 = conv2d(net3, 1024, [1, 1], scope='conv5', stride=[1, 1], padding='VALID',
                  bn=True, bn_decay=bn_decay, is_training=is_training, trainable=trainable)
    net5 = max_pool2d(net4, [num_point, 1], scope='maxpool', padding='VALID')

    net6 = tf.reshape(net5, [batch_size, -1])

    return net6, num_point, end_points


def get_model_repre(net, trainable_id=True, trainable_exp=True):

    f_id = full_connected(net, 512, scope='fc1_parallel', activation_fn=None, trainable=trainable_id)

    f_exp = full_connected(net, 512, scope='fc2_parallel', activation_fn=None, trainable=trainable_exp)

    return f_id, f_exp


def get_model_decoder(f_id, f_exp, num_point, end_points, trainable_id=True, trainable_exp=True):

    d_id_net = full_connected(f_id, 1024, scope='fc_de_id', trainable=trainable_id)

    d_exp_net = full_connected(f_exp, 1024, scope='fc_de_exp', trainable=trainable_exp)

    s_id = full_connected(d_id_net, num_point * 3, scope='fc_shape_id', activation_fn='Tanh', trainable=trainable_id)

    s_exp = full_connected(d_exp_net, num_point * 3, scope='fc_shape_exp', activation_fn='Tanh', trainable=trainable_exp)

    s_pred = tf.add(s_id, s_exp)

    return s_id, s_exp, s_pred, end_points


def get_loss(s_id, faces, label_points, end_points, lambda1, lambda2):
    # faces = data_preprosessing.open_face_file('./subjects/sub0_exp0.obj')

    # second line to fourth line points_data should be replaced by label_points

    # points_data = data_preprosessing.load_npyfile('./points_sampling/sub{0}_rand{1}.npy'.format(0, 0))
    # points_data = tf.convert_to_tensor(points_data, dtype=tf.float32)
    # points_data = tf.expand_dims(points_data, 0)


    label_points = tf.squeeze(label_points)
    shp_id = tf.reshape(s_id, shape=(29495, 3))
    s_target = tf.reshape(label_points, shape=(1, 88485))
    shp_target = tf.reshape(s_target, shape=(29495, 3))

    normals_pred = data_preprosessing.normals_cal(shp_id, faces)
    normals_target = data_preprosessing.normals_cal(label_points, faces)

    edge_0_pred, edge_1_pred, edge_2_pred = data_preprosessing.edge_cal(shp_id, faces)
    edge_0_target, edge_1_target, edge_2_target = data_preprosessing.edge_cal(shp_target, faces)

    # L1 loss for vertices
    l_vt = tf.reduce_sum(tf.abs(s_target - s_id))

    # l_normal is the loss for surface normals
    l_normal = tf.reduce_sum(1 - tf.reduce_sum(normals_target * normals_pred, axis=1)) / normals_target.get_shape()[0]

    # l_edge is the loss for edge length
    l_edge = tf.reduce_mean(tf.reduce_sum(tf.abs(edge_0_pred / edge_0_target - 1)) +
                            tf.reduce_sum(tf.abs(edge_1_pred / edge_1_target - 1)) +
                            tf.reduce_sum(tf.abs(edge_2_pred / edge_2_target - 1))) / 58366

    loss_supervised = l_vt + lambda1 * l_normal + lambda2 * l_edge
    # loss_supervised = l_vt + lambda1 * l_normal

    return loss_supervised


def get_loss_real(s_id, faces, label_points, end_points, lambda1, lambda2):

    shp_id = tf.reshape(s_id, shape=(29495, 3))

    dist1, idx1, dist2, idx2 = data_preprosessing.nn_distance(shp_id, label_points)
    loss_unsupervised = tf.reduce_sum(dist1) + tf.reduce_sum(dist2)
    return loss_unsupervised







'''
if __name__ == '__main__':

    with tf.Graph().as_default():
        inputs = tf.zeros((1, 29495, 3))
        s_id, s_exp, s_pred, end_points = get_model(inputs, tf.constant(True))
        get_loss(s_id, s_exp, s_pred, 0, end_points, 1.6e-4, 1.6e-4)
'''