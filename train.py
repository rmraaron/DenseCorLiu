import tensorflow as tf
import numpy as np
import model
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
MAX_EPOCH = 30

LAMBDA1 = 1.6e-4
LAMBDA2 = 1.6e-4

BASE_LEARNING_RATE = 1e-4


if not os.path.exists('./log'):
    os.mkdir('./log')


logfile = open('./log/log_train.txt', 'w')


def get_learning_rate(epoch, sub_num, rand_num):
    global_step = tf.divide(tf.divide(tf.divide(epoch, rand_num), sub_num), 5)
    global_step = tf.cast(global_step, tf.int32)
    lr = tf.compat.v1.train.exponential_decay(learning_rate=BASE_LEARNING_RATE, decay_rate=0.5,
                                              global_step=global_step, decay_steps=1, staircase=True)

    return lr


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(BN_INIT_DECAY, batch * BATCH_SIZE, BN_DECAY_DECAY_STEP,
                                                       BN_DECAY_DECAY_RATE, staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def log_writing(str_written):
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
            s_id, s_exp, s_pred, end_points = model.get_model(point_clouds, is_training_supervised, bn_decay=bn_decay)
            loss = model.get_loss(s_id, s_exp, s_pred, faces_tri, label_points, end_points, LAMBDA1, LAMBDA2)
            tf.compat.v1.summary.scalar('loss', loss)

            epoch_lr = tf.compat.v1.Variable(1)
            learning_rate = get_learning_rate(epoch_lr, 1500, 10)
            # learning_rate = BASE_LEARNING_RATE
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_op_adam = optimizer.minimize(loss, global_step=epoch_lr)

            saver = tf.compat.v1.train.Saver()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter('./train', sess.graph)

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

        fn_len = len(os.listdir('./pc_sampling'))
        subject_num = int(fn_len / 10)
        for i in range(subject_num):
            for j in range(10):
                for epoch in range(1, MAX_EPOCH+1):

                    log_writing('************************* EPOCH %d *************************' % epoch)
                    log_writing('***************** LEARNING RATE: %f *****************' % learning_rate.eval(session=sess))
                    sys.stdout.flush()
                    print('************************* EPOCH %d *************************' % epoch)
                    print(learning_rate.eval(session=sess))
                    print(epoch_lr.eval(session=sess))

                    is_training = True

                    # 1500 subjects with 10 neutral expressions

                    log_writing('_________________ sub: %d' % i + ' ' + 'rand: %d' % j + ' _________________')
                    print('_________________ sub: %d' % i + ' ' + 'rand: %d' % j + ' _________________')
                    points_data = data_preprosessing.load_npyfile('./pc_sampling/sub{0}_rand{1}.npy'.format(i, j))
                    # points_data = tf.convert_to_tensor(points_data, dtype=tf.float32)
                    points_data = np.expand_dims(points_data, axis=0)

                    # points_data for training is same as label_points
                    # label_points = tf.expand_dims(points_data, 0)
                    label_points = points_data

                    faces_triangle = data_preprosessing.open_face_file('./pc_sampling/sub{0}_rand{1}.npy'.format(i, j))

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

                        log_writing('loss: %f' % loss_value)
                        print('loss: %f' % loss_value)
            # np.save('./sub0_rand0_epoch{}'.format(epoch), s_id.reshape(29495, 3))

            if epoch % 5 == 0:
                save_path = saver.save(sess, './log/model.ckpt')
                log_writing('model saved in file: %s' % save_path)
                print('model saved in file: %s' % save_path)


if __name__ == '__main__':
    train()
    logfile.close()