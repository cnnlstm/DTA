#coding=utf-8
from __future__ import division
import os
import tensorflow as tf
import numpy as np 
import glob
import sys
import h5py 
import time
import random
from tqdm import tqdm
from conv_cell import ConvLSTMCell
from fc_attention import fc_attention_sum,fc_attention
from conv_attention import conv_attention_sum,conv_attention

########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
conv_img_features = "/your_dir/ucf11_features/image_pool5/"
fc_img_features = "/your_dir/ucf11_features/image_fc6/"


train_list = "/your_dir/ucf11TrainTestlist/ucf11_train.txt"
test_list = "/your_dir/ucf11TrainTestlist/ucf11_test.txt"
classInd = "/your_dir/ucf11TrainTestlist/classInd.txt"






batch_size = 6
n_inputs = 4096   
n_steps = 40    
n_hidden_units = filters = 1024
fc_attention_size = 50
n_classes = 11
n_layers = 2
scale = 1
n_num =10

timesteps = num_frames = 40
shape_1 = [7, 7]
shape_2 = [5, 10]
kernel = [3, 3]
channels = 512
attention_kernel = [1,1]
basic_lr = 0.0001




g = open(classInd,"r")
labels = sorted(g.readlines())
nums = []
names = []
for label in labels:
  a = label.split(" ")
  nums.append(int(a[0])-1)
  names.append(a[1][:-2])
label_dict = dict(zip(names,nums))
print label_dict


train_lines = []
f = open(train_list,"r")  
lines_ = f.readlines()
for i in range(len(lines_)//scale):
  train_lines.append(lines_[i*scale])
len_train = len(train_lines)
print "successfully,len(train_list)",len_train
random.shuffle(train_lines)





h = open(test_list,"r")  
test_lines_ = sorted(h.readlines())
test_lines = [] 
for i in range(len(test_lines_)//scale):
  test_lines.append(test_lines_[i*scale])
len_test = len(test_lines)
print "successfully,len(test_list)",len_test





test_labels = []
ground_labels = []
for test_video in test_lines:
    video_class = str(test_video.split(" ")).split("_")[1]
    ground_label = label_dict[video_class]
    test_labels.append(ground_label)

print test_labels

#########################################################################



def fc_feature(video_name):

    g = h5py.File(fc_img_features+video_name)
    img_features = g['video_feature']
    img_features = img_features[:]
    g.close()
    return img_features


def conv_feature(video_name):
    g = h5py.File(conv_img_features+video_name)
    img_features = g['video_feature']
    img_features = img_features[:]
    g.close()
    return img_features


def accuracy(a,b):
    c = np.equal(a,b).astype(float)
    acc = sum(c)/len(c)
    return acc



def compute_score_loss(batch_fc_img,batch_conv_img,batch_labels):
    global fc_pred
    global conv_pred
    global loss
    fc_score,conv_score,video_loss =  sess.run([fc_pred,conv_pred,loss], feed_dict={
            fc_img : batch_fc_img,
            conv_img: batch_conv_img,
            ys : batch_labels
            })
    return fc_score,conv_score,video_loss





#################################################################################






fc_img = tf.placeholder(tf.float32, [None, 40, 4096]) 

conv_img = tf.placeholder(tf.float32, [None, timesteps] + shape_1 + [channels]) 



ys = tf.placeholder(tf.float32, [None, n_classes])
Lr = tf.placeholder(tf.float32) 





def FC_LSTM(X_spa,attention):
    weights_img = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
    biases_img = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))


    X_spa = tf.reshape(X_spa, [n_num*batch_size*n_steps, n_inputs])
    X_spa = tf.matmul(X_spa, weights_img) + biases_img
    X_spa = tf.reshape(X_spa, [-1, n_steps, n_hidden_units])


    cell_spa = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    mlstm_cell_spa = tf.contrib.rnn.MultiRNNCell([cell_spa for _ in range(n_layers)], state_is_tuple=True)     
    init_state_spa = mlstm_cell_spa.zero_state(n_num*batch_size, dtype=tf.float32)
    outputs_spa, final_state_spa = tf.nn.dynamic_rnn(mlstm_cell_spa, X_spa, initial_state=init_state_spa, time_major=False)
    
    attention_output_spa = fc_attention_sum(outputs_spa, fc_attention_size)
    if attention == True:
        attention_output_spa = tf.layers.batch_normalization(attention_output_spa)
        return attention_output_spa
    else:
        outputs_spa = tf.layers.batch_normalization(outputs_spa)
        outputs_spa = tf.reduce_sum(outputs_spa,axis = 1)
        return outputs_spa


def CONV_LSTM(conv_img,shape,attention):
    img_cell = ConvLSTMCell(shape, filters, kernel)
    img_outputs, img_state = tf.nn.dynamic_rnn(img_cell, conv_img, dtype=conv_img.dtype, time_major=True)
    if attention == True: 
        img_attention_output = conv_attention_sum(img_outputs, attention_kernel)
        img_attention_output = tf.layers.batch_normalization(img_attention_output)
        return img_attention_output
    else:
        img_outputs = tf.layers.batch_normalization(img_outputs)
        img_outputs = tf.reduce_sum(img_outputs,axis = 1)
        return img_outputs  




def FC_layer(inputs):
  

    weights2 =  tf.get_variable("weights2", [n_hidden_units, n_classes],
        initializer=tf.truncated_normal_initializer())
    biases2 = tf.get_variable("biases2", [n_classes],
        initializer=tf.truncated_normal_initializer())
    result = tf.nn.dropout((tf.matmul(inputs, weights2) + biases2), 0.5)

    return result









fc_img_out = FC_LSTM(fc_img, True)
conv_img_out = CONV_LSTM(conv_img, shape_1,True)


fc_img_drop = tf.nn.dropout(fc_img_out, 0.5)
conv_img_drop = tf.nn.dropout(conv_img_out, 0.5)
conv_img_drop = tf.nn.max_pool(conv_img_drop,[1,7,7,1],[1,1,1,1],padding='VALID')
conv_img_drop = tf.reshape(conv_img_drop,[-1,filters])





with tf.variable_scope("FC"):
    fc_result = FC_layer(fc_img_drop)
    tf.get_variable_scope().reuse_variables()
    conv_result = FC_layer(conv_img_drop)








fc_pred = tf.nn.softmax(fc_result)
conv_pred = tf.nn.softmax(conv_result)



fc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_result, labels=ys))
conv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conv_result, labels=ys))
loss = fc_loss+conv_loss
print loss






train_op = tf.train.AdamOptimizer(Lr).minimize(loss)


config = tf.ConfigProto() 


saver = tf.train.Saver()

if not os.path.exists('fc+conv_lstm_tmp_attention'):
    os.mkdir('fc+conv_lstm_tmp_attention/')
sess = tf.Session(config=config)





merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)
init_op = tf.global_variables_initializer() 
sess.run(init_op)


############################################start_training###########################################


k = 0
u = 0














full_loss = []
for l in range(1000):
    lr = basic_lr
    pbar = tqdm(total=((len_train//batch_size)+1))
    for i in range((len_train//batch_size)+1):

        batch_fc_img = np.zeros((10*batch_size,40,4096),dtype = np.float32)
        batch_conv_img = np.zeros((10*batch_size,40,7,7,512),dtype = np.float32)
        batch_labels = np.zeros((10*batch_size,n_classes),dtype = np.int8)


        time1 = time.time()
        k= 0
        for j in range(i*batch_size,(i+1)*batch_size):
            if j>=len_train:
                j = random.randint(0,len_train-1)
            one_hot=np.zeros((n_classes),dtype = np.int8)
            line = train_lines[j]

            video_name = line.split(" ")[0]

            video_class = str(line.split(" ")[0]).split("_")[1]
            feature_name = video_name+"_.h5"


            one_hot[label_dict[video_class]] = 1
            label = np.expand_dims(one_hot,axis = 0)
            label = label.repeat(10,axis=0)
          
            fc_img_fea = fc_feature(feature_name)
            conv_img_fea = conv_feature(feature_name)


            fc_img_fea = np.transpose(fc_img_fea,(1,0,2))
            conv_img_fea = np.transpose(conv_img_fea,(1,0,3,4,2))


            batch_fc_img[n_num*k:n_num*k+n_num] = fc_img_fea
            batch_conv_img[n_num*k:n_num*k+n_num] = conv_img_fea
            batch_labels[n_num*k:n_num*k+n_num] = label
            k = k+1
        time2 = time.time()
        print "batch_read_time:", '{0:.2f}'.format(time2-time1),"s"
        k = k+1

        loss_,_ = sess.run([loss,train_op], feed_dict={
            fc_img : batch_fc_img,
            conv_img : batch_conv_img,
            ys : batch_labels,
            Lr : lr
        })

        print "batch_loss:",loss_
        pbar.update(1)
    pbar.close()



    model_name = "fc+conv_lstm_tmp_attention/"+str(l)+"epoch_model.ckpt"
    saver.save(sess, model_name)

    print "successfully,start_test!"

############################################################################################


    pbar = tqdm(total=((len_test//batch_size)+1))
    fc_scores = []
    conv_scores = []
    for i in range((len_test//batch_size)+1):


        batch_fc_img = np.zeros((n_num*batch_size,40,4096),dtype = np.float32)
        batch_conv_img = np.zeros((n_num*batch_size,40,7,7,512),dtype = np.float32)
        batch_labels = np.zeros((n_num*batch_size,n_classes),dtype = np.int8)




        time1 = time.time()
        k= 0
        for j in range(i*batch_size,(i+1)*batch_size):
            if j>=len_test:
                j = random.randint(0,len_test-1)
            one_hot=np.zeros((n_classes),dtype = np.int8)
            line = test_lines[j]
            video_class = str(line.split(" ")[0]).split("_")[1]
            feature_name = line.split(" ")[0]+"_.h5"


            one_hot[label_dict[video_class]] = 1
            label = np.expand_dims(one_hot,axis = 0)
            label = label.repeat(10,axis=0)


            conv_img_fea = conv_feature(feature_name)
            conv_img_fea = np.transpose(conv_img_fea,(1,0,3,4,2))

            fc_img_fea = fc_feature(feature_name)
            fc_img_fea = np.transpose(fc_img_fea,(1,0,2))

            batch_fc_img[n_num*k:n_num*k+n_num] = fc_img_fea
            batch_conv_img[n_num*k:n_num*k+n_num] = conv_img_fea
            batch_labels[n_num*k:n_num*k+n_num] = label
            k = k+1
        test_fc_score, test_conv_score, test_loss = compute_score_loss(batch_fc_img,batch_conv_img,batch_labels)
        print "test_loss:",test_loss
        test_fc_score = np.sum(np.reshape(np.array(test_fc_score),(batch_size,n_num,n_classes)),axis=1)    #(batch_size, n_classes)
        test_conv_score = np.sum(np.reshape(np.array(test_conv_score),(batch_size,n_num,n_classes)),axis=1) 
    


        fc_scores.append(test_fc_score)
        conv_scores.append(test_conv_score)
        pbar.update(1)
    fc_scores = np.reshape(np.array(fc_scores),(-1,n_classes))[:len_test]
    conv_scores = np.reshape(np.array(conv_scores),(-1,n_classes))[:len_test]


    print fc_scores.shape
    print conv_scores.shape

    pbar.close()

    num_test_score = fc_scores
    test_label_pred = np.argmax(num_test_score,axis = 1)
    print test_label_pred.shape
    print l,"epoch:, attention，Fc_lstm_ACC:",accuracy(test_label_pred,test_labels)
    test_info = []
    for i in range(len_test):
      video_info = []
      video_info.append(num_test_score[i])
      video_info.append(test_labels[i])
      test_info.append(video_info)
      
    npz_name = "score_attention/"+str(l)+"_epoch_fc_score.npz"
    np.savez(npz_name, test_info=test_info)




    num_test_score = conv_scores
    test_label_pred = np.argmax(num_test_score,axis = 1)
    print test_label_pred.shape
    print l,"epoch:, attention，Conv_lstm_ACC:",accuracy(test_label_pred,test_labels)
    test_info = []
    for i in range(len_test):
      video_info = []
      video_info.append(num_test_score[i])
      video_info.append(test_labels[i])
      test_info.append(video_info)
      
    npz_name = "score_attention/"+str(l)+"_epoch_conv_score.npz"
    np.savez(npz_name, test_info=test_info)








    num_test_score = np.add(fc_scores,conv_scores)
    test_label_pred = np.argmax(num_test_score,axis = 1)
    print test_label_pred.shape
    print l,"epoch:, attention，Add_fusion_ACC:",accuracy(test_label_pred,test_labels)
    test_info = []
    for i in range(len_test):
      video_info = []
      video_info.append(num_test_score[i])
      video_info.append(test_labels[i])
      test_info.append(video_info)
      
    npz_name = "score_attention/"+str(l)+"_epoch_numscore.npz"
    np.savez(npz_name, test_info=test_info)



    mul_test_score = np.multiply(fc_scores,conv_scores)
    test_label_pred = np.argmax(mul_test_score,axis = 1)
    print test_label_pred.shape
    print l,"epoch:, attention，Mul_fusion_ACC:",accuracy(test_label_pred,test_labels)
    test_info = []
    for i in range(len_test):
      video_info = []
      video_info.append(mul_test_score[i])
      video_info.append(test_labels[i])
      test_info.append(video_info)
      
    npz_name = "score_attention/"+str(l)+"_epoch_mulscore.npz"
    np.savez(npz_name, test_info=test_info)
