import tensorflow as tf

def fm_model(paras):
    n = paras['feature_num']
    k = paras['intent_dims']
    w0 = tf.Variable(0.1)
    w1 = tf.Variable(tf.truncated_normal(n))
    w2 = tf.Variable(tf.truncated_normal([ n, k] ), mean = 0, stddev = 0.1)


    x_ = tf.placeholder(tf.float32, [None, n])
    y_ = tf.placeholder( tf.float32, [None] )

    one_ = tf.add( w0, tf.reduce_sum( tf.multiply(w1, x_), axis = 1 ) )
    two_ = tf.multiply( 0.5, \
                        tf.reduce_sum \
                            ( \
                                tf.subtract( \
                                            tf.pow( tf.matmul(x_, w2), 2 ), \
                                            tf.matmul(tf.pow(x_, 2), tf.pow(w2, 2)) \
                                ), 1 \
                            ) \
                        ) \

    y_fm = one_ + two_
    y_prob = tf.sigmoid(y_fm)
    cost_ = tf.losses.sigmoid_cross_entropy(y_, y_fm)
    train_op = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            _, cost = \
                sess.run( [train_op, cost_], feed_dict={x_: paras['train_x'], y_: paras['train_y']} )
            _, cost_test = \
                sess.run([train_op, cost_], feed_dict={x_: paras['test_x'], y_: paras['test_y']})
            if i % 20 == 0:
                print("train_mse is %s", cost)
                print("test_mse is %s", cost_test)

        weights = sess.run((w0, w1, w2 ), feed_dict = { x_ : paras['test_x'], y_: paras['test_y'] })

    return weights

if __name__ == '__main__':
    paras = {}
    fm_weights = fm_model(paras)









