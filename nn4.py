# -*- coding: utf-8 -*-

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

def extract(data):
	i = 0
	extract = []

	while i < 489:
		a = data.loc[i,:]
		a = a[0].split(",")
		a = a[1:49]
		a = [int(i) for i in a]
		extract.append(a)
		i += 1

	return extract 

def main():

	MODEL_DIR = os.path.join(os.path.dirname(__file__), 'nn1')
	if os.path.exists(MODEL_DIR) is False:
		os.mkdir(MODEL_DIR)

	LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
	if os.path.exists(LOG_DIR) is False:
		os.mkdir(LOG_DIR)

	data0701 = pd.read_table('data/pop-2013-07-01.txt')
	data0701 = pd.DataFrame(data0701)
	data0707 = pd.read_table('data/pop-2013-07-07.txt')
	data0707 = pd.DataFrame(data0707)
	data1007 = pd.read_table('data/pop-2013-10-07.txt')
	data1007 = pd.DataFrame(data1007)
	data1013 = pd.read_table('data/pop-2013-10-13.txt')
	data1013 = pd.DataFrame(data1013)
	data1216 = pd.read_table('data/pop-2013-12-16.txt')
	data1216 = pd.DataFrame(data1216)
	data1222 = pd.read_table('data/pop-2013-12-22.txt')
	data1222 = pd.DataFrame(data1222)

	data0701 = extract(data0701)
	data0707 = extract(data0707)
	data1007 = extract(data1007)
	data1013 = extract(data1013)
	data1216 = extract(data1216)
	data1222 = extract(data1222)

	LAG = 1			#number of units in input layer 

	maxlen = 48 - LAG 		#the length of one time series data
	
	#for each train data and target data
	train01 = []
	train_target01 = []
	train11 = []
	train_target11 = []
	train21 = []
	train_target21 = []
	train31 = []
	train_target31 = []
	train41 = []
	train_target41 = []
	#for each validation data and target data
	val1 = []
	val_target = []
	#for final test data
	test1 = []
	test_target = []

	Loss = {
		'train_loss': [],
		'val_loss': []
		}

	i = 0
	train00 = []
	train10 = []
	train20 = []
	train30 = []
	train40 = []
	while i < 489:
		train00.append(data0701[i])
		train10.append(data0707[i])
		train20.append(data1007[i])
		train30.append(data1013[i])
		train40.append(data1216[i])
		i += 1

	train0 = [[]]
	train = []
	train_target0 = [[]]
	train_target = []

	train1 = [[]]
	traina = []
	train_target1 = [[]]
	train_targeta = []

	train2 = [[]]
	trainb = []
	train_target2 = [[]]
	train_targetb = []

	train3 = [[]]
	trainc = []
	train_target3 = [[]]
	train_targetc = []

	train4 = [[]]
	traind = []
	train_target4 = [[]]
	train_targetd = []

	i = 0
	k = 0
	while k < 489:
		hoge = train00[k]
		hoge2 = train10[k]
		hoge3 = train20[k]
		hoge4 = train30[k]
		hoge5 = train40[k]
		while i < maxlen:
			train0[0].append(hoge[i: i+LAG])
			train_target0[0].append([[hoge[i+LAG]]])
	
			train1[0].append(hoge2[i: i+LAG])
			train_target1[0].append([[hoge2[i+LAG]]])

			train2[0].append(hoge3[i: i+LAG])
			train_target2[0].append([[hoge3[i+LAG]]])

			train3[0].append(hoge4[i: i+LAG])
			train_target3[0].append([[hoge4[i+LAG]]])

			train4[0].append(hoge5[i: i+LAG])
			train_target4[0].append([[hoge5[i+LAG]]])

			i += 1
		train.append(train0[0])
		train_target.append(train_target0[0])

		traina.append(train1[0])
		train_targeta.append(train_target1[0])
	
		trainb.append(train2[0])
		train_targetb.append(train_target2[0])

		trainc.append(train3[0])
		train_targetc.append(train_target3[0])

		traind.append(train4[0])
		train_targetd.append(train_target4[0])

		train0 = [[]]
		train_target0 = [[]]
		train1 = [[]]
		train_target1 = [[]]
		train2 = [[]]
		train_target2 = [[]]
		train3 = [[]]
		train_target3 = [[]]
		train4 = [[]]
		train_target4 = [[]]

		i = 0
		k += 1

#### for big data
	s = []
	t = []
	i = 0
	while i < 489:
		train_set01 = np.array(train[i]).reshape(1, len(train[i]), LAG)		#3 dimensions data
		s.append(train_set01) 
		train_target01 = np.array(train_target[i]).reshape(1, len(train_target[i]), LAG)
		t.append(train_target01) 
		i += 1

	i = 0
	while i < 489:
		train_set11 = np.array(traina[i]).reshape(1, len(traina[i]), LAG)		#3 dimensions data
		s.append(train_set11) 
		train_target11 = np.array(train_targeta[i]).reshape(1, len(train_targeta[i]), LAG)
		t.append(train_target11) 
		i += 1

	i = 0
	while i < 489:
		train_set21 = np.array(trainb[i]).reshape(1, len(trainb[i]), LAG)		#3 dimensions data
		s.append(train_set21) 
		train_target21 = np.array(train_targetb[i]).reshape(1, len(train_targetb[i]), LAG)
		t.append(train_target21) 
		i += 1

	i = 0
	while i < 489:
		train_set31 = np.array(trainc[i]).reshape(1, len(trainc[i]), LAG)		#3 dimensions data
		s.append(train_set31) 
		train_target31 = np.array(train_targetc[i]).reshape(1, len(train_targetc[i]), LAG)
		t.append(train_target31) 
		i += 1

	i = 0
	while i < 489:
		train_set41 = np.array(traind[i]).reshape(1, len(traind[i]), LAG)		#3 dimensions data
		s.append(train_set41) 
		train_target41 = np.array(train_targetd[i]).reshape(1, len(train_targetd[i]), LAG)
		t.append(train_target41) 
		i += 1

	i = 0
	while i < 2445:
		if i == 0:
			train_set = s[i]
			train_target = t[i]
		else:
			train_set = np.concatenate((train_set, s[i]), axis=0)
			train_target = np.concatenate((train_target, t[i]), axis=0)
		i += 1

#train_set is 7/1-12/16 data in descending order
#train_target is 7/1-12/16 data in descending order
########################################

#	print(train_set[0:4])
#	print(train_target[:,0:4,:])

	epochs = 80000 
	batch_size = 1956 

	x_data = tf.placeholder(tf.float32, shape = [batch_size, maxlen, LAG])
	y_target = tf.placeholder(tf.float32, shape = [batch_size, maxlen, 1])
	n_batch = tf.placeholder(tf.int32, [])      #set [] for gpu

	hidden_layer_nodes = 10 

	w1 = tf.Variable(tf.random_normal(shape=[batch_size, LAG, hidden_layer_nodes]), name='w1')	# weight of hidden layer 
	b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]), name='b1')			# one biases for each hidden node
	w2 = tf.Variable(tf.random_normal(shape=[batch_size, hidden_layer_nodes, 1]), name='w2')		# weight of ouput layer 
	b2 = tf.Variable(tf.random_normal(shape=[1]), name='b2')					# 1 bias for the output

#	hidden_output = tf.tanh(tf.add(tf.matmul(x_data, w1), b1))
	hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, w1), b1))
	final_output = tf.add(tf.matmul(hidden_output, w2), b2)

	mse = tf.reduce_mean(tf.square(y_target - final_output), axis=1)

	my_opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
#	my_opt = tf.train.GradientDescentOptimizer(1/10000)
	train_step = my_opt.minimize(mse)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	print("-------start time----------")
	print(time.ctime())
	print("---------------------------")

	score = []
	i = 0
	while i < 5:
		sess = tf.Session()
		sess.run(init)

		file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 
		summaries = tf.summary.merge_all() 

		if i == 0:		#xxxxo
			val_set0 = train_set[489:2445]
			val_target0 = train_target[489:2445]
			train_set0 = train_set[0:1956]
			train_target0 = train_target[0:1956]
		elif i == 1:		#xxxox
			val_set0 = train_set[0:1956]
			val_target0 = train_target[0:1956]
			train_set0 = np.delete(train_set, range(1467,1956), axis=0)
			train_target0 = np.delete(train_target, range(1467,1956), axis=0)
		elif i == 2:		#xxoxx
			val_set0 = train_set[0:1956]
			val_target0 = train_target[0:1956]
			train_set0 = np.delete(train_set, range(978,1467), axis=0)
			train_target0 = np.delete(train_target, range(978,1467), axis=0)
		elif i == 3:		#xoxxx
			val_set0 = train_set[489:2445]
			val_target0 = train_target[489:2445]
			train_set0 = np.delete(train_set, range(489,978), axis=0)
			train_target0 = np.delete(train_target, range(489,978), axis=0)
		else:			#oxxxx
			val_set0 = train_set[0:1956]
			val_target0 = train_target[0:1956]
			train_set0 = train_set[489:2445]
			train_target0 = train_target[489:2445]

		for epoch in range(epochs):
			X0, Y0 = train_set0, train_target0
			X, Y = val_set0, val_target0
	
			sess.run(train_step, feed_dict={
				x_data: X0,
				y_target: Y0,
				n_batch: batch_size
			})

			train_loss = sess.run(mse, feed_dict={
				x_data: X0,
				y_target: Y0,
				n_batch: batch_size
			})

			#show graphs on tensorboard
#			summary = sess.run(summaries, feed_dict={
#				x_data: X0,
#				y_target: Y0,
#				n_batch: batch_size 
#			})

#			file_writer.add_summary(summary, epoch) 

			#evaluation with validation data
			val_loss = sess.run(mse, feed_dict={
				x_data: X,
				y_target: Y,
				n_batch: batch_size
			})
#			print(i)
#			print(len(val_loss))

			Loss['train_loss'].append(train_loss)

			if i == 0:
				val_loss = np.average(val_loss[1467:1956]) 
				Loss['val_loss'].append(val_loss)
			elif i == 1:
				val_loss = np.average(val_loss[1467:1956]) 
				Loss['val_loss'].append(val_loss)
			elif i == 2:
				val_loss = np.average(val_loss[978:1467]) 
				Loss['val_loss'].append(val_loss)
			elif i == 3:
				val_loss = np.average(val_loss[0:489]) 
				Loss['val_loss'].append(val_loss)
			else:
				val_loss = np.average(val_loss[0:489]) 
				Loss['val_loss'].append(val_loss)

		s = np.average(Loss['val_loss'])

		#save a best model
		if i > 0 and s < min(score):
			model_path = saver.save(sess, MODEL_DIR + '/nn1.ckpt')
			print('Model saved to:', model_path)

			#to show the graph of train and validation loss 
			plt.clf()
			plt.close()
			plt.figure()
			tl = np.mean(Loss['train_loss'], axis=1)
			plt.plot(range(epoch+1), tl, "k-", markersize=3, label='train_loss')
			plt.plot(range(epoch+1), Loss['val_loss'], "r-", markersize=3, label='val_loss')
			plt.xlabel('epoch')
			plt.ylabel('error')
#			plt.ylim(0,1000)
			plt.legend()
		elif i == 0:
			model_path = saver.save(sess, MODEL_DIR + '/nn1.ckpt')
			print('Model saved to:', model_path)

			#show the graph of train and validation loss 
			plt.figure()
			tl = np.mean(Loss['train_loss'], axis=1)
			plt.plot(range(epoch+1), tl, "k-", markersize=3, label='train_loss')
			plt.plot(range(epoch+1), Loss['val_loss'], "r-", markersize=3, label='val_loss')
			plt.xlabel('epoch')
			plt.ylabel('error')
#			plt.ylim(0,1000)
			plt.legend()
		else:
			pass

		score.append(s)

		Loss['train_loss'].clear()
		Loss['val_loss'].clear()
	
		sess.close()	
		i += 1 	

	n = score.index((min(score)))
	print("minimum of validation loss", n)
#	print(min(sum0))
#	print(sum0)
	print("validation score", np.average(score))		#compare the accuracy of the hyperparameter

	print("-----end time-----")
	print(time.ctime())
	print("------------------")

	plt.show()

if __name__ == '__main__':
    main()
