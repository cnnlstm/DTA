import tensorflow as tf


def conv_attention_sum(inputs, attention_kernel, time_major=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,H,W,C) => (B,T,H,W,C) 
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2, 3, 4])

    inputs_shape = inputs.shape                                         #[B,T,H,W,C]
    feature_lengths = inputs_shape[1].value                             #[T]
    feature_windows = [inputs_shape[2].value,inputs_shape[3].value]     #[H,W]
    feature_channels = inputs_shape[4].value                            #[C]

    w_size = attention_kernel + [feature_channels,feature_channels]
    #W = tf.get_variable('kernel', attention_kernel + [feature_channels,feature_channels])            #[1,1,C,C]
    W = tf.Variable(tf.random_normal(w_size, stddev=0.1))  
    b = tf.Variable(tf.random_normal([feature_channels], stddev=0.1))                             #[C]
    u = tf.Variable(tf.random_normal([feature_channels], stddev=0.1))                             #[C]
    


    inputs_conv = tf.reshape(inputs, [-1, inputs_shape[2].value,inputs_shape[3].value,feature_channels])           #[B*T,H,W,C]
    y = tf.nn.conv2d(inputs_conv, W, strides=[1, 1, 1, 1], padding='SAME')                             #[B*T,H,W,C]
    v = tf.nn.relu(y + b)                                                                              #[B*T,H,W,C]

    
    vu = tf.matmul(tf.reshape(v,[-1,feature_channels]), tf.reshape(u, [-1, 1]))                 #[B*T*H*W,C]*[C,1] = [B*T*H*W,1]                                                                                          
    exps = tf.reshape(tf.exp(vu), [-1, feature_lengths])            #[B*W*H,T]                                                  
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])     #[B*W*H,T]                                          
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, feature_lengths, inputs_shape[2].value, inputs_shape[3].value, 1]), 1)        #[B,T,H,W,C]*[B,T,H,W,1] ===> [B,H,W,C]                 

    return output   #[B*H*W*C]
############################################################# remove the time_dims









def conv_attention(inputs, attention_kernel, time_major=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,H,W,C) => (B,T,H,W,C) 
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2, 3, 4])

    inputs_shape = inputs.shape                                         #[B,T,H,W,C]
    feature_lengths = inputs_shape[1].value                             #[T]
    feature_windows = [inputs_shape[2].value,inputs_shape[3].value]     #[H,W]
    feature_channels = inputs_shape[4].value                            #[C]

    w_size = attention_kernel + [feature_channels,feature_channels]
    W = tf.Variable(tf.random_normal(w_size, stddev=0.1))  
    b = tf.Variable(tf.random_normal([feature_channels], stddev=0.1))                             #[C]
    u = tf.Variable(tf.random_normal([feature_channels], stddev=0.1))                             #[C]
    


    inputs_conv = tf.reshape(inputs, [-1, inputs_shape[2].value,inputs_shape[3].value,feature_channels])           #[B*T,H,W,C]
    y = tf.nn.conv2d(inputs_conv, W, strides=[1, 1, 1, 1], padding='SAME')                             #[B*T,H,W,C]
    v = tf.nn.relu(y + b)                                                                              #[B*T,H,W,C]

    
    vu = tf.matmul(tf.reshape(v,[-1,feature_channels]), tf.reshape(u, [-1, 1]))                 #[B*T*H*W,C]*[C,1] = [B*T*H*W,1]                                                                                          
    
    exps = tf.reshape(tf.exp(vu), [-1, feature_lengths])            #[B*W*H,T]                                                  
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])     #[B*W*H,T]                                          

    # Output of Bi-RNN is reduced with attention vector
    output =inputs * tf.reshape(alphas, [-1, feature_lengths, inputs_shape[2].value, inputs_shape[3].value, 1])        #[B,T,H,W,C]*[B,T,H,W,1] ===> [B,H,W,C]                 

    return output   #[B*H*W*C]
