import tensorflow as tf
import numpy as np
print(tf.__version__)



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

embedding_size = 100

input_ = tf.placeholder(shape=(None,sents_len),dtype=tf.int32,name='input')

decoder_input = tf.placeholder(shape=(None,1),dtype=tf.int32,name='de_input')

decoder_target = tf.placeholder(shape=(None,sents_len),dtype=tf.int32,name='tar')

embedding = tf.get_variable('embedding', [len(id2word),embedding_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))

with tf.variable_scope("encoder"):
    input_embedding = tf.nn.embedding_lookup(embedding,input_)
    f_cell = tf.contrib.rnn.LSTMCell(embedding_size)
    b_cell = tf.contrib.rnn.LSTMCell(embedding_size)
    (_, (encoder_final_f, encoder_final_b)) = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell,
                                                                              input_embedding,
                                                                              dtype=tf.float32)

    encoder_final_f_c = encoder_final_f[0]
    encoder_final_f_h = encoder_final_f[1]
    encoder_final_b_c = encoder_final_b[0]
    encoder_final_b_h = encoder_final_b[1]
    encoder_final_c = tf.concat([encoder_final_f_c, encoder_final_b_c], -1)
    encoder_final_h = tf.concat([encoder_final_f_h, encoder_final_b_h], -1)
    encoder_final = tf.contrib.rnn.LSTMStateTuple(encoder_final_c, encoder_final_h)


    mean = tf.layers.dense(inputs=encoder_final_h, units=z_dim, activation=tf.nn.relu, name='mean')
    stddev = tf.layers.dense(inputs=encoder_final_h, units=z_dim, activation=tf.nn.relu, name='stddev')

    tmp1 = tf.random.normal(tf.shape(mean), mean=0, stddev=1, dtype=tf.float32)
    z = mean+stddev*tmp1

    w = tf.get_variable("w", [z_dim, 400],
                        dtype=tf.float32)
    b = tf.get_variable("b", [400], dtype=tf.float32)
    decoder_initial_state = tf.nn.relu(tf.matmul(z, w) + b)
    decoder_initial_state = tuple(tf.split(axis=1, num_or_size_splits=2, value=decoder_initial_state))
    state =decoder_initial_state

with tf.variable_scope('decoder'):

    decoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(decoder_input, axis=-1)]
    # print(decoder_inputs)
    cell = tf.contrib.rnn.LSTMCell(embedding_size*2)
    out_cell = []
    predict_=[]
    imp = decoder_inputs[0]
    for i in range(sents_len):
        out, state = cell(imp, state)
        vocab_dist = tf.layers.dense(out, len(id2word))
        out_cell.append(vocab_dist)
        predict = tf.arg_max(vocab_dist, -1)
        predict_.append(predict)
        # print(predict)
        imp = tf.nn.embedding_lookup(embedding, predict)

with tf.variable_scope('loss'):
    # w = tf.Variable(tf.random_normal(shape=[40, 10], dtype=tf.float32), name="W")
    # b = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32), name="b")
    loss_ = []
    pre = []
    decoder_tars = tf.unstack(decoder_target, axis=-1)
    # print(len(decoder_tars))
    # print(len(out_cell))

    for i in range(len(out_cell)):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_tars[i], depth=len(id2word)),
                                                            logits=out_cell[i])
        loss = tf.reduce_mean(loss)
        loss_.append(loss)
    loss = tf.reduce_mean(loss_)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    kl_loss =  - 0.5 * tf.reduce_sum(1 + stddev - tf.square(mean) - tf.exp(stddev), axis=-1)
    rec_loss = loss
    vae_loss = tf.reduce_mean(rec_loss+kl_loss)

    train_op = optimizer.minimize(vae_loss, global_step=global_step)



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
        decoder_target_ = [[len(word2id)-1]]*len(train_batch)
        _, loss = sess.run([train_op, vae_loss], feed_dict={input_:train_batch, decoder_input:decoder_target_, decoder_target:train_batch})
        if step%100==0:
            z_sample = np.random.normal(loc=0, scale=1, size=z_dim)
            print(z_sample)
            print(loss)
            predict = sess.run(predict_, feed_dict={z: [z_sample],decoder_input:[[len(word2id)-1]]})
            res=[]
            for i in predict:
                res.append(id2word[i.tolist()[0]])
            print(''.join(res))
            print()
        step+=1



