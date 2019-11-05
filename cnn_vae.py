import tensorflow as tf
import numpy as np


data = []
r = open('./train.txt','r')
for line in r.readlines():
    data.append(line.strip())

z_dim=30
sents_len = 100

id2word=dict(enumerate(set(''.join(data))))
len_id = len(id2word)
id2word[len_id] = "PAD"

word2id={i:j for j,i in id2word.items()}
trainid = [[word2id[j] for j in i ]for i in data]
trainid_=[]
for k in trainid:
    if len(k)>=100:
        trainid_.append(k[:100])
    else:
        trainid_.append(k[:]+(100-len(k))*[len_id])

num_filter = 30
embedding_size = 100
input_ = tf.placeholder(shape=(None,sents_len),dtype=tf.int32,name='input')
embedding = tf.get_variable('embedding', [len(id2word),embedding_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))

with tf.variable_scope("encoder"):
    input_embedding = tf.nn.embedding_lookup(embedding,input_)
    input_embedding_expand = tf.expand_dims(input_embedding,-1)

    x = tf.layers.conv2d(input_embedding_expand, filters=64, kernel_size= 2, strides=2, padding="SAME")
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=128, kernel_size=2, strides=2, padding="SAME")
    x = tf.nn.relu(x)
    x_flat = tf.contrib.layers.flatten(x)

    mean = tf.layers.dense(inputs=x_flat, units=z_dim, activation=tf.nn.relu, name='mean')
    stddev = tf.layers.dense(inputs=x_flat, units=z_dim, activation=tf.nn.relu, name='stddev')

    tmp1 = tf.random.normal(tf.shape(mean), mean=0, stddev=1, dtype=tf.float32)
    z = mean+stddev*tmp1

with tf.variable_scope("decoder"):
    x = tf.layers.dense(z, units=25*25*128)
    x = tf.reshape(x, [-1, 25, 25, 128])
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=2, strides=2, padding="SAME")
    x = tf.nn.relu(x)

    x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=2, strides=2, padding="SAME")
    x = tf.nn.relu(x)

    x = tf.squeeze(x,-1)

    last_out = tf.layers.dense(inputs=x, units=len(id2word), activation=tf.nn.relu, name='last_out')
    predict_out = tf.argmax(last_out, axis=-1)
    input_labels = tf.one_hot(input_, depth=len(id2word))

    rec_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=last_out),1)
    kl_loss =  - 0.5 * tf.reduce_sum(1 + stddev - tf.square(mean) - tf.exp(stddev), axis=-1)
    vae_loss = tf.reduce_mean(rec_loss)

    opti = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = opti.minimize(vae_loss)

def itera(list_A):
    while True:
        temp=[]
        for i in range(len(list_A)-10):
            temp.append(list_A[i])
            if len(temp)==10:
                yield temp
                temp=[]

def get_next(trainid_):
    g = itera(trainid_)
    return g

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    batch = get_next(trainid_)
    while step < 100000:
        train_batch = next(batch)
        _, loss = sess.run([train_op, vae_loss], feed_dict={input_:train_batch})
        print(loss)
        if step%100==0:
            z_sample = np.random.normal(loc=0, scale=1, size=30)
            predict = sess.run(predict_out, feed_dict={z: [z_sample]})
            res=[]
            for i in predict.tolist()[0]:
                res.append(id2word[i])
            print(''.join(res))
        step+=1



