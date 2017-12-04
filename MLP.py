import numpy as np
import cPickle as pickle
import tensorflow as tf
from numpy.random import uniform

from sklearn.preprocessing import StandardScaler
# HyperParameters
flags = tf.flags
flags.DEFINE_integer('n_in', 4, 'input size')
flags.DEFINE_integer('n_out', 1, 'output size')
# has to be between n_in and n_out
flags.DEFINE_integer('n_hidden', 5, 'hidden layer size')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
FLAGS = flags.FLAGS


def construct_MLP():    
    W_hid = tf.Variable(uniform( low=-0.01 , high=0.01, size=(FLAGS.n_in,FLAGS.n_hidden)).astype('float32'), 
                        name='W_h')
    b_hid = tf.Variable( np.zeros( [FLAGS.n_hidden], dtype='float32'),name='b_h')
    W_out = tf.Variable( uniform( low=-0.01, high=0.01, size=(FLAGS.n_hidden, FLAGS.n_out)).astype('float32'),
                         name='W_o')
    b_out =  tf.Variable( tf.zeros([FLAGS.n_out]), name='b_o')

    #creation of symbolic input and label
    input_x = tf.placeholder( "float", [None, FLAGS.n_in])
    label_y = tf.placeholder( "float", [None] )

    # hout = sigmoid(x*W + b)
    h_in = tf.nn.bias_add(tf.matmul(input_x,W_hid),b_hid)
    h_out = tf.sigmoid(h_in)

    # o_out = sigmoid(x*W + b)
    o_in = tf.nn.bias_add(tf.matmul(h_out, W_out), b_out)
    y_pred = tf.sigmoid(o_in)

    # mean squared value as cost
    cost = tf.reduce_mean((y_pred - label_y)**2)
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)
    return input_x, label_y, cost, train_op, y_pred;

def train_network(epochs):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_costs = np.zeros(epochs, dtype='float32');
    train_accuracies = np.zeros(epochs, dtype='float32');
    ### TRAINING BEGIN ###
    print "Epoch  Cost "
    for i in xrange(epochs):
        train_cost, _ = sess.run([cost, train_op], feed_dict={input_x: train_X, label_y: train_y})
        train_costs[i] = train_cost
        if i % 200 == 0:
            print "%05d  %5.3f" % (i,train_cost);
    print "%05d  %5.3f" % (i,train_cost);        
    ### TRAINING END ###
    
    return train_costs, y_pred, sess;

if __name__ == '__main__':
    with open("preprocessed_data.pkl", 'rb') as f:
        train, valid, test = pickle.load(f)

    train_X, train_y = train
    valid_X, valid_y = valid
    # test_X = test

    sc = StandardScaler().fit(train_X)
    train_X = sc.transform(train_X)
    valid_X = sc.transform(valid_X)


    scY = StandardScaler().fit(train_y.values.reshape(-1,1))
    train_y = scY.transform(train_y.values.reshape(-1,1))
    valid_y = scY.transform(valid_y.values.reshape(-1,1))
    train_y =  train_y.reshape(1,-1)[0]
    valid_y =  valid_y.reshape(1,-1)[0]

#    print scY.inverse_transform(train_y.reshape(-1,1))

    input_x, label_y,cost, train_op, y_pred = construct_MLP()
    train_costs, y_pred, sess = train_network(2000)

    #print y_pred

    outputs = sess.run( y_pred, feed_dict={input_x: train_X, label_y: train_y});

    print scY.inverse_transform(outputs[0:5].reshape(-1,1))
    print scY.inverse_transform(train_y[0:5].reshape(-1,1))
    
