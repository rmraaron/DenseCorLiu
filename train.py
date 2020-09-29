import tensorflow as tf
import numpy as np
import model_new as model
import os
import sys
import data_preprosessing
import random

DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

BATCH_SIZE = 1
NUM_POINT = 29495
MAX_EPOCH_ID = 150
MAX_EPOCH_EXP = 30

LAMBDA1 = 1.6e-4
LAMBDA2 = 1.6e-4

BASE_LEARNING_RATE = 1e-4


if not os.path.exists('./log'):
    os.mkdir('./log')

if not os.path.exists('./log/fixed'):
    os.mkdir('./log/fixed')


logfile_train = open('./log/log_train.txt', 'w')

logfile_eval = open('./log/log_eval.txt', 'w')


def get_learning_rate_id(epoch, max_epoch, num):
    epoch_n = tf.divide(epoch - 1, num) + 1

    global_step = tf.divide(epoch_n - 1, 5)

    global_step = tf.cast(global_step, tf.int32)

    lr = tf.compat.v1.train.exponential_decay(learning_rate=BASE_LEARNING_RATE, decay_rate=0.5,
                                              global_step=global_step, decay_steps=1, staircase=True)

    return lr


def get_learning_rate_exp(epoch, max_epoch):

    epoch_n = tf.math.floormod(epoch - 1, max_epoch)

    global_step = tf.divide(epoch_n, 5)

    global_step = tf.cast(global_step, tf.int32)

    lr = tf.compat.v1.train.exponential_decay(learning_rate=BASE_LEARNING_RATE, decay_rate=0.5,
                                              global_step=global_step, decay_steps=1, staircase=True)

    return lr


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(BN_INIT_DECAY, batch * BATCH_SIZE, BN_DECAY_DECAY_STEP,
                                                       BN_DECAY_DECAY_RATE, staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def log_writing(logfile, str_written):
    logfile.write(str_written + '\n')
    logfile.flush()


def train():
    with tf.Graph().as_default():
        with tf.device('/device:gpu:0'):

            point_clouds, label_points, faces_tri = model.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_supervised = tf.compat.v1.placeholder(tf.bool, shape=())

            batch = tf.compat.v1.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)
            net6, num_point, end_points = model.get_model_encoder(point_clouds, is_training_supervised, bn_decay=bn_decay)
            f_id, f_exp = model.get_model_repre(net6)
            s_id, s_exp, s_pred, end_points = model.get_model_decoder(f_id, f_exp, num_point, end_points)
            # loss = model.get_loss(s_id, faces_tri, label_points, end_points, LAMBDA1, LAMBDA2)
            loss = model.get_loss_real(s_id, faces_tri, label_points, end_points, LAMBDA1, LAMBDA2)
            tf.compat.v1.summary.scalar('loss', loss)

            epoch_lr = tf.compat.v1.Variable(1)
            learning_rate = get_learning_rate_id(epoch_lr, MAX_EPOCH_ID, 5)
            # learning_rate = BASE_LEARNING_RATE
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_op_adam = optimizer.minimize(loss, global_step=epoch_lr)

            # saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.variable_scope("", reuse=True):
                weight_conv1 = tf.compat.v1.get_variable("conv1/weights", shape=[1, 3, 1, 64])
                bias_conv1 = tf.compat.v1.get_variable("conv1/biases", shape=[64, ])
                weight_conv2 = tf.compat.v1.get_variable("conv2/weights", shape=[1, 1, 64, 64])
                bias_conv2 = tf.compat.v1.get_variable("conv2/biases", shape=[64, ])
                weight_conv3 = tf.compat.v1.get_variable("conv3/weights", shape=[1, 1, 64, 64])
                bias_conv3 = tf.compat.v1.get_variable("conv3/biases", shape=[64, ])
                weight_conv4 = tf.compat.v1.get_variable("conv4/weights", shape=[1, 1, 64, 128])
                bias_conv4 = tf.compat.v1.get_variable("conv4/biases", shape=[128, ])
                weight_conv5 = tf.compat.v1.get_variable("conv5/weights", shape=[1, 1, 128, 1024])
                bias_conv5 = tf.compat.v1.get_variable("conv5/biases", shape=[1024, ])

                weight_fc_id = tf.compat.v1.get_variable("fc1_parallel/weights", shape=[1024, 512])
                bias_fc_id = tf.compat.v1.get_variable("fc1_parallel/biases", shape=[512, ])
                weight_fc_de_id = tf.compat.v1.get_variable("fc_de_id/weights", shape=[512, 1024])
                bias_fc_de_id = tf.compat.v1.get_variable("fc_de_id/biases", shape=[1024, ])
                weight_fc_shape_id = tf.compat.v1.get_variable("fc_shape_id/weights", shape=[1024, 88485])
                bias_fc_shape_id = tf.compat.v1.get_variable("fc_shape_id/biases", shape=[88485, ])
            saver = tf.compat.v1.train.Saver([weight_conv1, bias_conv1, weight_conv2, bias_conv2, weight_conv3,
                                              bias_conv3, weight_conv4, bias_conv4, weight_conv5, bias_conv5,
                                              weight_fc_id, bias_fc_id, weight_fc_de_id, bias_fc_de_id,
                                              weight_fc_shape_id, bias_fc_shape_id])

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
        train_writer_id = tf.compat.v1.summary.FileWriter('./train', sess.graph)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init, {is_training_supervised: True})

        ops = {'point_clouds': point_clouds,
               'label_points': label_points,
               'is_training_supervised': is_training_supervised,
               's_id': s_id,
               's_exp': s_exp,
               's_pred': s_pred,
               'faces_tri': faces_tri,
               'loss': loss,
               'train_op_adam': train_op_adam,
               'merged': merged,
               'step': batch}

        fn_len = len(os.listdir('./subject_points'))
        subject_num = int(fn_len / 7)
        for epoch in range(1, MAX_EPOCH_ID + 1):
            for i in range(1):

                faces_triangle = data_preprosessing.open_face_obj('./subjects/sub{0}_exp0.obj'.format(i))

                log_writing(logfile_train, '************************* EPOCH %d *************************' % epoch)
                log_writing(logfile_train, '***************** LEARNING RATE: %f *****************' % learning_rate.eval(session=sess))
                sys.stdout.flush()
                print('************************* EPOCH %d *************************' % epoch)
                print(learning_rate.eval(session=sess))
                print(epoch_lr.eval(session=sess))

                train_one_epoch_id(sess, ops, train_writer_id, i, faces_triangle, epoch)


        save_path = saver.save(sess, './log/fixed/model.ckpt')
        log_writing(logfile_train, 'model saved in file: %s' % save_path)
        print('model saved in file: %s' % save_path)


def train_one_epoch_id(sess, ops, train_writer, i, faces_triangle, epoch):
    is_training = True

    points_data = data_preprosessing.load_npyfile('./subject_points/sub{0}_exp0.npy'.format(i))
    # points_data = data_preprosessing.load_real_npyfile('./test.npy')
    # points_data = tf.convert_to_tensor(points_data, dtype=tf.float32)
    points_data = np.expand_dims(points_data, axis=0)

    # points_data for training is same as label_points
    # label_points = tf.expand_dims(points_data, 0)
    label_points = points_data

    # faces_triangle = data_preprosessing.open_face_file('./pc_sampling/sub{0}_rand{1}.npy'.format(i, j))

    # points_data.eval(session=sess)

    # BATCH_SIZE is equal to 1, thus the shape of model inputs is (1, NUM_POINT, 3)

    for batch in range(BATCH_SIZE):
        feed_dict = {ops['point_clouds']: points_data,
                     ops['label_points']: label_points,
                     ops['faces_tri']: faces_triangle,
                     ops['is_training_supervised']: is_training}
        summary, step, _, loss_value, s_id = sess.run([ops['merged'], ops['step'], ops['train_op_adam'],
                                                       ops['loss'], ops['s_id']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        log_writing(logfile_train, 'loss_train: %f' % loss_value)
        print('loss_train: %f' % loss_value)

        if epoch == MAX_EPOCH_ID:
            np.save('./sub{}_exp0'.format(i), s_id.reshape(29495, 3))


def train_exp():

    with tf.Graph().as_default():
        with tf.device('/device:gpu:0'):
            point_clouds, label_points, faces_tri = model.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_supervised = tf.compat.v1.placeholder(tf.bool, shape=())

            batch = tf.compat.v1.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)
            net6, num_point, end_points = model.get_model_encoder(point_clouds, is_training_supervised,
                                                                      bn_decay=bn_decay)
            f_id, f_exp = model.get_model_repre(net6, trainable_id=False)
            s_id, s_exp, s_pred, end_points = model.get_model_decoder(f_id, f_exp, num_point, end_points, trainable_id=False)
            loss = model.get_loss(s_exp, faces_tri, label_points, end_points, LAMBDA1, LAMBDA2)

            tf.compat.v1.summary.scalar('loss', loss)

            epoch_lr = tf.compat.v1.Variable(1)
            learning_rate = get_learning_rate_id(epoch_lr, MAX_EPOCH_EXP, 20)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_op_adam = optimizer.minimize(loss, global_step=epoch_lr)

            # tf.compat.v1.reset_default_graph()
            with tf.compat.v1.variable_scope("", reuse=True):
                weight_conv1 = tf.compat.v1.get_variable("conv1/weights", shape=[1, 3, 1, 64])
                bias_conv1 = tf.compat.v1.get_variable("conv1/biases", shape=[64, ])
                weight_conv2 = tf.compat.v1.get_variable("conv2/weights", shape=[1, 1, 64, 64])
                bias_conv2 = tf.compat.v1.get_variable("conv2/biases", shape=[64, ])
                weight_conv3 = tf.compat.v1.get_variable("conv3/weights", shape=[1, 1, 64, 64])
                bias_conv3 = tf.compat.v1.get_variable("conv3/biases", shape=[64, ])
                weight_conv4 = tf.compat.v1.get_variable("conv4/weights", shape=[1, 1, 64, 128])
                bias_conv4 = tf.compat.v1.get_variable("conv4/biases", shape=[128, ])
                weight_conv5 = tf.compat.v1.get_variable("conv5/weights", shape=[1, 1, 128, 1024])
                bias_conv5 = tf.compat.v1.get_variable("conv5/biases", shape=[1024, ])

                weight_fc_id = tf.compat.v1.get_variable("fc1_parallel/weights", shape=[1024, 512])
                bias_fc_id = tf.compat.v1.get_variable("fc1_parallel/biases", shape=[512, ])
                weight_fc_de_id = tf.compat.v1.get_variable("fc_de_id/weights", shape=[512, 1024])
                bias_fc_de_id = tf.compat.v1.get_variable("fc_de_id/biases", shape=[1024, ])
                weight_fc_shape_id = tf.compat.v1.get_variable("fc_shape_id/weights", shape=[1024, 88485])
                bias_fc_shape_id = tf.compat.v1.get_variable("fc_shape_id/biases", shape=[88485, ])

            saver = tf.compat.v1.train.Saver([weight_conv1, bias_conv1, weight_conv2, bias_conv2, weight_conv3,
                                              bias_conv3, weight_conv4, bias_conv4, weight_conv5, bias_conv5,
                                              weight_fc_id, bias_fc_id, weight_fc_de_id, bias_fc_de_id,
                                              weight_fc_shape_id, bias_fc_shape_id])

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True

        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
        train_writer_exp = tf.compat.v1.summary.FileWriter('./train', sess.graph)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init, {is_training_supervised: True})

        saver.restore(sess, './log/fixed/model.ckpt')
        log_writing(logfile_train, 'model restored')

        # print("weight_fc_id : %s" % weight_fc_id.eval(sess))
        # print("bias_fc_id : %s" % bias_fc_id.eval(sess))
        # print("weight_fc_de_id : %s" % weight_fc_de_id.eval(sess))
        # print("bias_fc_de_id : %s" % bias_fc_de_id.eval(sess))
        # print("weight_fc_shape_id : %s" % weight_fc_shape_id.eval(sess))
        # print("bias_fc_shape_id : %s" % bias_fc_shape_id.eval(sess))

        ops = {'point_clouds': point_clouds,
               'label_points': label_points,
               'is_training_supervised': is_training_supervised,
               'f_id': f_id,
               'f_exp': f_exp,
               's_id': s_id,
               's_exp': s_exp,
               's_pred': s_pred,
               'faces_tri': faces_tri,
               'loss': loss,
               'train_op_adam': train_op_adam,
               'merged': merged,
               'step': batch}

        fn_len = len(os.listdir('./subject_points'))
        subject_num = int(fn_len / 7)

        faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

        pc_list = []

        label_list = []

        for epoch in range(1, MAX_EPOCH_EXP + 1):

            for j in range(1, 5):

                for i in range(5):

                    log_writing(logfile_train, '_________________ sub: %d' % i + ' ' + 'exp: %d' % j + ' _________________')
                    print('_________________ sub: %d' % i + ' ' + 'exp: %d' % j + ' _________________')

                    points_data = data_preprosessing.load_npyfile('./subject_points/sub{}_exp{}.npy'.format(i, j))
                    # points_data = tf.convert_to_tensor(points_data, dtype=tf.float32)
                    points_data = np.expand_dims(points_data, axis=0)

                    # points_data for training is same as label_points
                    # label_points = tf.expand_dims(points_data, 0)
                    label_points = points_data

                    # pc_list.append(points_data)

                    # label_list.append(label_points)

        # indices = list(zip(pc_list, label_list))

        # random.shuffle(indices)

        # pc_list, label_list = zip(* indices)

        # for epoch in range(1, MAX_EPOCH_EXP + 1):

            # for batch in range(BATCH_SIZE):

                # for point_data, label_point in zip(pc_list, label_list):

                    for batch in range(BATCH_SIZE):

                        log_writing(logfile_train, '************************* EPOCH %d *************************' % epoch)
                        log_writing(logfile_train,
                                    '***************** LEARNING RATE: %f *****************' % learning_rate.eval(
                                        session=sess))
                        sys.stdout.flush()
                        print('************************* EPOCH %d *************************' % epoch)
                        print(learning_rate.eval(session=sess))
                        print(epoch_lr.eval(session=sess))

                        is_training = True

                        feed_dict = {ops['point_clouds']: points_data,
                                     ops['label_points']: label_points,
                                     ops['faces_tri']: faces_triangle,
                                     ops['is_training_supervised']: is_training}
                        summary, step, _, loss_value, s_exp, s_pred = sess.run([ops['merged'], ops['step'],
                                                                                ops['train_op_adam'], ops['loss'],
                                                                                ops['s_exp'], ops['s_pred']], feed_dict=feed_dict)
                        train_writer_exp.add_summary(summary, step)

                        log_writing(logfile_train, 'loss_train: %f' % loss_value)
                        print('loss_train: %f' % loss_value)
                        if epoch == MAX_EPOCH_EXP:
                            np.save('./sub{}_exp{}'.format(i, j), s_exp.reshape(29495, 3))
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, './log/model.ckpt')
        log_writing(logfile_train, 'model saved in file: %s' % save_path)
        print('model saved in file: %s' % save_path)




def evaluate():

    is_training = False

    with tf.Graph().as_default():
        with tf.device('/device:gpu:0'):
            point_clouds, label_points, faces_tri = model.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_supervised = tf.compat.v1.placeholder(tf.bool, shape=())

            # s_id, s_exp, s_pred, end_points = model.get_model(point_clouds, is_training_supervised)
            # loss = model.get_loss(s_id, s_exp, s_pred, faces_tri, label_points, end_points, LAMBDA1, LAMBDA2)

            net6, num_point, end_points = model.get_model_encoder(point_clouds, is_training_supervised)
            f_id, f_exp = model.get_model_repre(net6)
            s_id, s_exp, s_pred, end_points = model.get_model_decoder(f_id, f_exp, num_point, end_points)
            loss = model.get_loss(s_pred, faces_tri, label_points, end_points, LAMBDA1, LAMBDA2)

            saver = tf.compat.v1.train.Saver()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True

        sess = tf.compat.v1.Session(config=config)
        saver.restore(sess, './log/model.ckpt')
        log_writing(logfile_eval, 'model restored')

        ops = {'point_clouds': point_clouds,
               'label_points': label_points,
               'is_training_supervised': is_training_supervised,
               'f_id': f_id,
               'f_exp': f_exp,
               's_id': s_id,
               's_exp': s_exp,
               's_pred': s_pred,
               'faces_tri': faces_tri,
               'loss': loss}

        fn_len = len(os.listdir('./subject_points'))
        subject_num = int(fn_len / 7)
        for i in range(5):

            faces_triangle = data_preprosessing.open_face_obj('./subjects/sub{0}_exp0.obj'.format(i))

            for j in range(5):
                # 1500 subjects with 6 expressions
                log_writing(logfile_eval, '_________________ sub: %d' % i + ' ' + 'exp: %d' % j + ' _________________')
                print('_________________ sub: %d' % i + ' ' + 'exp: %d' % j + ' _________________')
                points_data = data_preprosessing.load_npyfile('./subject_points/sub{0}_exp{1}.npy'.format(i, j))

                # points_data = data_preprosessing.load_npyfile('./pc_sampling/sub{0}_rand9.npy'.format(i))
                points_data = np.expand_dims(points_data, axis=0)

                label_points = points_data

                # faces_triangle = data_preprosessing.open_face_file('./pc_sampling/sub{0}_rand0.npy'.format(i))

                for batch in range(BATCH_SIZE):
                    feed_dict = {ops['point_clouds']: points_data,
                                 ops['label_points']: label_points,
                                 ops['faces_tri']: faces_triangle,
                                 ops['is_training_supervised']: is_training}
                    loss_value, s_pred = sess.run([ops['loss'], ops['s_pred']], feed_dict=feed_dict)

                    log_writing(logfile_eval, 'loss_test: %f' % loss_value)
                    print('loss_test: %f' % loss_value)

                np.save('./sub{}_exp{}'.format(i, j), s_pred.reshape(29495, 3))


if __name__ == '__main__':
    train()
    # train_exp()
    # logfile_train.close()
    # evaluate()
    # logfile_eval.close()