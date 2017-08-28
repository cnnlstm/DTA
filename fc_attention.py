import tensorflow as tf


def fc_attention(inputs, attention_size, time_major=False):

    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape  #[B,T,D]
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer   #[T]
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer                                        #[D]

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))                         #[D,A]
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))                                      #[A]
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))                                      #[A]

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))      #inputs = [T*B,D], *[D,A] = [T*B,A],  v = inputs + [1, A] = [T*B,A]
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))                                                            #vu = [T*B,A]*[A,1] = [T*B,1]                                         
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])                                                       #exps = [B,T]
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])                                                # / = [B,1]      alphas = [B,T] 
    output = inputs * tf.reshape(alphas, [-1, sequence_length, 1])

    return output
############################################################# remove the time_dims


def fc_attention_sum(inputs, attention_size, time_major=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape  #[B,T,D]
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer   #[T]
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer                                        #[D]

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))                         #[D,A]
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))                                      #[A]
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))                                      #[A]

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))      #inputs = [T*B,D], *[D,A] = [T*B,A],  v = inputs + [1, A] = [T*B,A]
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))                                                            #vu = [T*B,A]*[A,1] = [T*B,1]                                         
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])                                                       #exps = [B,T]
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])                                                # / = [B,1]      alphas = [B,T] 
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)                           #[B,T,D]*[B,T,1] = [B,T,D]  output = [B,D]


    return output
