import tensorflow as tf
import numpy as np
from tqdm import tqdm

import model_new as model
import os
import sys
import data_preprosessing

DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

BATCH_SIZE = 1
NUM_POINT = 29495
MAX_EPOCH_ID = 10
MAX_EPOCH_EXP = 10
MAX_EPOCH_END = 10

LAMBDA1 = 1.6e-4
LAMBDA2 = 1.6e-4

BASE_LEARNING_RATE = 1e-4


if not os.path.exists('./log'):
    os.mkdir('./log')

if not os.path.exists('./log/fixed'):
    os.mkdir('./log/fixed')


def get_learning_rate(epoch_lr, num):
    epoch_n = tf.divide(epoch_lr, num) + 1

    global_step = tf.divide(epoch_n - 1, 5)

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
    logfile_train = open('./log/log_train.txt', 'w')
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
            learning_rate = get_learning_rate(epoch_lr, 1500)
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
               'step': epoch_lr}

        faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

        for epoch in tqdm(range(1, MAX_EPOCH_ID + 1)):
            log_writing(logfile_train, '************************* EPOCH %d *************************' % epoch)
            log_writing(logfile_train,
                        '***************** LEARNING RATE: %f *****************' % learning_rate.eval(session=sess))
            sys.stdout.flush()
            print('************************* EPOCH %d *************************' % epoch)

            epoch_mean_loss = train_one_epoch_id(sess, ops, train_writer_id, logfile_train, faces_triangle, epoch, epoch_lr)
            print('epoch mean loss: %f' % epoch_mean_loss)


        save_path = saver.save(sess, './log/fixed/model.ckpt')
        log_writing(logfile_train, 'model saved in file: %s' % save_path)
        print('model saved in file: %s' % save_path)


def train_one_epoch_id(sess, ops, train_writer, logfile_train, faces_triangle, epoch, epoch_lr):
    is_training = True

    pc_data, pc_label = data_preprosessing.loadh5File('./dataset/subject_points.h5')

    pc_data, pc_label, shuffle_idx = data_preprosessing.shuffle_data(pc_data, pc_label)

    file_size = pc_data.shape[0]
    num_batches = file_size

    epoch_loss = 0

    for batch_idx in tqdm(range(num_batches)):

        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        point_clouds = pc_data[start_idx:end_idx, :, :]

        point_label = pc_label[start_idx:end_idx, :, :]

        feed_dict = {ops['point_clouds']: point_clouds,
                     ops['label_points']: point_label,
                     ops['faces_tri']: faces_triangle,
                     ops['is_training_supervised']: is_training}
        summary, step, _, loss_value, s_id = sess.run([ops['merged'], ops['step'], ops['train_op_adam'],
                                                       ops['loss'], ops['s_id']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        log_writing(logfile_train, 'loss_train: %f' % loss_value)

        epoch_loss += loss_value

        # if epoch == MAX_EPOCH_ID:
        #     np.save('./real{}_exp0'.format(batch), s_id.reshape(29495, 3))

    epoch_loss_ave = epoch_loss / float(num_batches)
    log_writing(logfile_train, 'mean_loss: %f' % epoch_loss_ave)
    return epoch_loss_ave



def train_exp():
    logfile_train_exp = open('./log/log_train_exp.txt', 'w')

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
            learning_rate = get_learning_rate(epoch_lr, 1500 * 6)
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
        log_writing(logfile_train_exp, 'model restored')

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
               'step': epoch_lr}


        faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

        for epoch in tqdm(range(1, MAX_EPOCH_EXP + 1)):

            log_writing(logfile_train_exp, '************************* EPOCH %d *************************' % epoch)
            log_writing(logfile_train_exp,
                        '***************** LEARNING RATE: %f *****************' % learning_rate.eval(session=sess))
            sys.stdout.flush()
            print('************************* EPOCH %d *************************' % epoch)

            epoch_mean_loss = train_one_epoch_exp(sess, ops, train_writer_exp, logfile_train_exp, faces_triangle, epoch, epoch_lr)
            print('epoch mean loss: %f' % epoch_mean_loss)

        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, './log/model.ckpt')
        log_writing(logfile_train_exp, 'model saved in file: %s' % save_path)
        print('model saved in file: %s' % save_path)


def train_one_epoch_exp(sess, ops, train_writer_exp, logfile_train_exp, faces_triangle, epoch):
    is_training = True

    pc_data, pc_label = data_preprosessing.loadh5File('./dataset/expression_points.h5')

    pc_data, pc_label, shuffle_idx = data_preprosessing.shuffle_data(pc_data, pc_label)

    file_size = pc_data.shape[0]
    num_batches = file_size

    epoch_loss = 0

    for batch_idx in tqdm(range(num_batches)):

        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        point_clouds = pc_data[start_idx:end_idx, :, :]

        point_label = pc_label[start_idx:end_idx, :, :]

        feed_dict = {ops['point_clouds']: point_clouds,
                     ops['label_points']: point_label,
                     ops['faces_tri']: faces_triangle,
                     ops['is_training_supervised']: is_training}
        summary, step, _, loss_value, s_pred = sess.run([ops['merged'], ops['step'], ops['train_op_adam'],
                                                       ops['loss'], ops['s_pred']], feed_dict=feed_dict)
        train_writer_exp.add_summary(summary, step)

        log_writing(logfile_train_exp, 'loss_train: %f' % loss_value)

        epoch_loss += loss_value

        # if epoch == MAX_EPOCH_EXP:
        #     np.save('./real{}_exp0'.format(batch), s_pred.reshape(29495, 3))

    epoch_loss_ave = epoch_loss / float(num_batches)
    log_writing(logfile_train_exp, 'mean_loss: %f' % epoch_loss_ave)
    return epoch_loss_ave


def end_to_end_train():
    logfile_endtoend = open('./log/log_train_endtoend.txt', 'w')

    with tf.Graph().as_default():
        with tf.device('/device:gpu:0'):
            point_clouds, label_points, faces_tri = model.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_unsupervised = tf.compat.v1.placeholder(tf.bool, shape=())

            net6, num_point, end_points = model.get_model_encoder(point_clouds, is_training_unsupervised)
            f_id, f_exp = model.get_model_repre(net6)
            s_id, s_exp, s_pred, end_points = model.get_model_decoder(f_id, f_exp, num_point, end_points)
            loss = model.get_loss_real(s_pred, faces_tri, label_points, end_points, LAMBDA1, LAMBDA2)
            tf.compat.v1.summary.scalar('loss', loss)

            epoch_lr = tf.compat.v1.Variable(1)
            learning_rate = get_learning_rate(epoch_lr, 6000)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_op_adam = optimizer.minimize(loss, global_step=epoch_lr)

            saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True

        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
        train_writer_endtoend = tf.compat.v1.summary.FileWriter('./train', sess.graph)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init, {is_training_unsupervised: True})

        saver.restore(sess, './log/model.ckpt')
        log_writing(logfile_endtoend, 'model restored')

        ops = {'point_clouds': point_clouds,
               'label_points': label_points,
               'is_training_unsupervised': is_training_unsupervised,
               'f_id': f_id,
               'f_exp': f_exp,
               's_id': s_id,
               's_exp': s_exp,
               's_pred': s_pred,
               'faces_tri': faces_tri,
               'loss': loss,
               'train_op_adam': train_op_adam,
               'merged': merged,
               'step': epoch_lr
               }

        faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

        for epoch in tqdm(range(1, MAX_EPOCH_END + 1)):
            log_writing(logfile_endtoend, '************************* EPOCH %d *************************' % epoch)
            log_writing(logfile_endtoend,
                        '***************** LEARNING RATE: %f *****************' % learning_rate.eval(session=sess))
            sys.stdout.flush()
            print('************************* EPOCH %d *************************' % epoch)

            epoch_mean_loss = train_one_epoch_end(sess, ops, train_writer_endtoend, logfile_endtoend, faces_triangle,
                                                  epoch, epoch_lr)
            print('epoch mean loss: %f' % epoch_mean_loss)

        if not os.path.exists('./log/end_to_end'):
            os.mkdir('./log/end_to_end')
        save_path = saver.save(sess, './log/end_to_end/model.ckpt')
        log_writing(logfile_endtoend, 'model saved in file: %s' % save_path)
        print('model saved in file: %s' % save_path)


def train_one_epoch_end(sess, ops, train_writer_endtoend, logfile_endtoend, faces_triangle, epoch):
    is_training = True

    pc_data, pc_label = data_preprosessing.loadh5File('./dataset/all_points.h5')
    pc_data, pc_label, shuffle_idx = data_preprosessing.shuffle_data(pc_data, pc_label)

    file_size = pc_data.shape[0]
    num_batches = file_size

    epoch_loss = 0

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        point_clouds = pc_data[start_idx:end_idx, :, :]

        point_label = pc_label[start_idx:end_idx, :, :]

        feed_dict = {ops['point_clouds']: point_clouds,
                     ops['label_points']: point_label,
                     ops['faces_tri']: faces_triangle,
                     ops['is_training_unsupervised']: is_training}
        summary, step, _, loss_value, s_pred = sess.run([ops['merged'], ops['step'], ops['train_op_adam'],
                                                         ops['loss'], ops['s_pred']], feed_dict=feed_dict)
        train_writer_endtoend.add_summary(summary, step)

        log_writing(logfile_endtoend, 'loss_train: %f' % loss_value)

        epoch_loss += loss_value

        # if epoch == MAX_EPOCH_END:
        #     np.save('./real{}_exp0'.format(batch), s_pred.reshape(29495, 3))

    epoch_loss_ave = epoch_loss / float(num_batches)
    log_writing(logfile_endtoend, 'mean_loss: %f' % epoch_loss_ave)
    return epoch_loss_ave


if __name__ == '__main__':
    # train()
    # train_exp()
    end_to_end_train()
    # evaluate()
    # logfile_train.close()
    # logfile_eval.close()