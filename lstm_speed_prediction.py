# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
Run this script on tensorflow r0.10. Errors appear when using lower versions.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_load import load

BATCH_START = 0
TIME_STEPS = 70 #500 #20
BATCH_SIZE = 50
INPUT_SIZE = 255#1
OUTPUT_SIZE = 1
CELL_SIZE = 50
LR = 0.001
feature = 'feature1_short'
label_file = 'label1_short'
train_data, val_data, test_data, train_label, val_lable, test_label = load()
after = 70

def avaliable_time(after, data):
    ava_list = []
    startime = np.arange(100, data.shape[0] - 70, 170)
    for i in startime:
        tmp = list(range(i, i + after))
        ava_list.append(tmp)
    ava_list = np.asarray(ava_list).flatten()

    return ava_list.tolist()



def get_batch(data, label, ava_list):
    TIME_LENGTH  = len(ava_list)

    index = np.random.permutation(TIME_LENGTH)[0:BATCH_SIZE]
    #index = np.random.shuffle(T)[0:BATCH_SIZE]
    #global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    #xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    #seq = np.sin(xs)
    #res = np.cos(xs)
    #BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    #in_data, out_data = data_load
    x = np.empty([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
    y = np.empty([BATCH_SIZE, 1, OUTPUT_SIZE])
    for i in range(BATCH_SIZE):
        x[i, :, :] = data[ava_list[index[i]] - 70: ava_list[index[i]], :]
        y[i, :, :] = label[ ava_list[index[i]], :]
    return x, y


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, 1, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size) share weight
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        #Ws_in = self._weight_variable([self.input_size, self.output_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        #bs_in = self._bias_variable([self.output_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            pred = tf.matmul(l_out_x, Ws_out) + bs_out
            pred = tf.reshape(pred, [-1, self.n_steps,  self.output_size], name='2_3D')
            self.pred = pred[:, -1, :]

    def compute_cost(self):
        ''''
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size *  self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        '''
        losses = tf.losses.mean_squared_error(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            weights=1.0,
            scope=None,
            loss_collection=tf.GraphKeys.LOSSES,
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    step = []
    error_train = []
    error_test = []
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()
    plt.show()
    ava_list_train = avaliable_time(after, train_data)
    ava_list_test = avaliable_time(after, test_data)
    ava_list_val = avaliable_time(after, val_data)
    for i in range(50000):
        seq, res  = get_batch(train_data, train_label, ava_list_train)
        #seq, res  = get_batch(data, lable)
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)


        if i % 50 == 0:
            v_x, v_y =  get_batch(val_data, val_lable, ava_list_val)
            feed_dict_val = {
                model.xs: v_x,
                model.ys: v_y,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

            co = sess.run(
                model.cost,
                feed_dict=feed_dict)
            step.append(i)
            error_test.append(co)
            error_train.append(cost)

'''
    test_list = []
    for i in range(1000):
        v_x, v_y = get_batch(test_data, test_label, ava_list_test)
        feed_dict_val = {
            model.xs: v_x,
            model.ys: v_y,
            model.cell_init_state: state  # use last state as the initial state for this run
        }

        pred = sess.run(
            model.cost,
            feed_dict=feed_dict)
        #error = tf.reduce_mean(tf.square(tf.subtract(pred, v_y)))
        test_list.append(pred)
    print('test error=', tf.reduce_mean(test_list))
'''
fig = plt.figure(1)
plt.plot(step, error_train,'r', label = "training error")
plt.plot(step, error_test, 'b--', label= 'test error')
plt.ylabel('Mean Squared Error')
plt.xlabel('Step')
plt.title('Training process of LSTM')
fig.savefig('LSTM_result_lr1', dpi='figure')