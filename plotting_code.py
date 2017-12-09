import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pkl

dir_name = "results_1image/"
num_eps = 400

distance = np.zeros(num_eps)
reward = np.zeros(num_eps)
loss = np.zeros(num_eps)
steps = np.zeros(num_eps)
qval = np.zeros(num_eps)

for fname in os.listdir(dir_name):
	if "reward" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps:
			r = pkl.load(open(dir_name+fname, "rb"))
			reward[idx-1] = r[0]
	if "distance" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps:
			d = pkl.load(open(dir_name+fname, "rb"))
			distance[idx-1] = d[0]
	if "loss" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps:
			l = pkl.load(open(dir_name+fname, "rb"))
			loss[idx-1] = l[0]
	if "steps" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps:
			s = pkl.load(open(dir_name+fname, "rb"))
			steps[idx-1] = s[0]
	if "target" in fname:
		idx = int(fname.split("_")[-1])
		if idx > 0 and idx < num_eps:
			q = pkl.load(open(dir_name+fname, "rb"))
			qval[idx-1] = np.sum([np.average(x) for x in q[0]])
			

dir_name_5 = "results/"
num_eps_5 = 400

distance_5 = np.zeros(num_eps_5)
reward_5 = np.zeros(num_eps_5)
loss_5 = np.zeros(num_eps_5)
steps_5 = np.zeros(num_eps_5)
qval_5 = np.zeros(num_eps_5)

for fname in os.listdir(dir_name_5):
	if "reward" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps_5:
			r = pkl.load(open(dir_name_5+fname, "rb"))
			reward_5[idx-1] = r[0]
	if "distance" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps_5:
			d = pkl.load(open(dir_name_5+fname, "rb"))
			distance_5[idx-1] = d[0]
	if "loss" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps_5:
			l = pkl.load(open(dir_name_5+fname, "rb"))
			loss_5[idx-1] = l[0]
	if "steps" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps_5:
			s = pkl.load(open(dir_name_5+fname, "rb"))
			steps_5[idx-1] = s[0]
	if "target" in fname:
		idx = int(fname.split("_")[-1])
		if idx > 0 and idx < num_eps_5:
			q = pkl.load(open(dir_name_5+fname, "rb"))
			qval_5[idx-1] = np.sum([np.average(x) for x in q[0]])

# dir_name_rnn = "results_rnn/"
# num_eps_rnn = 422

# distance_rnn = np.zeros(num_eps_rnn)
# reward_rnn = np.zeros(num_eps_rnn)
# loss_rnn = np.zeros(num_eps_rnn)
# steps_rnn = np.zeros(num_eps_rnn)
# qval_rnn = np.zeros(num_eps_rnn)

# for fname in os.listdir(dir_name_rnn):
# 	if "reward" in fname:
# 		idx = int(fname.split("_")[1])
# 		if idx > 0 and idx < num_eps_rnn:
# 			r = pkl.load(open(dir_name_rnn+fname, "rb"))
# 			reward_rnn[idx-1] = r[0]
# 	if "distance" in fname:
# 		idx = int(fname.split("_")[1])
# 		if idx > 0 and idx < num_eps_rnn:
# 			d = pkl.load(open(dir_name_rnn+fname, "rb"))
# 			distance_rnn[idx-1] = d[0]
# 	if "loss" in fname:
# 		idx = int(fname.split("_")[1])
# 		if idx > 0 and idx < num_eps_rnn:
# 			l = pkl.load(open(dir_name_rnn+fname, "rb"))
# 			loss_rnn[idx-1] = l[0]
# 	if "steps" in fname:
# 		idx = int(fname.split("_")[1])
# 		if idx > 0 and idx < num_eps_rnn:
# 			s = pkl.load(open(dir_name_rnn+fname, "rb"))
# 			steps_rnn[idx-1] = s[0]
# 	if "target" in fname:
# 		idx = int(fname.split("_")[-1])
# 		if idx > 0 and idx < num_eps_rnn:
# 			q = pkl.load(open(dir_name_rnn+fname, "rb"))
# 			qval_rnn[idx-1] = np.sum([np.average(x) for x in q[0]])

plt.plot(range(num_eps), reward[:num_eps], label="1 image", color="b")
plt.plot(range(num_eps), reward_5[:num_eps], label="Last 5 images", color="r")
# plt.plot(range(num_eps_rnn), reward_rnn, label="RNN", color="g")
plt.xlabel("Episode Number", fontsize=25)
plt.ylabel("Reward", fontsize=25)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend()
plt.show()

plt.plot(range(num_eps), distance[:num_eps], label="1 image", color="b")
plt.plot(range(num_eps), distance_5[:num_eps], label="Last 5 images", color="r")
# plt.plot(range(num_eps_rnn), distance_rnn, label="RNN", color="g")
plt.xlabel("Episode Number", fontsize=25)
plt.ylabel("Distance", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend()
plt.show()

plt.plot(range(num_eps), steps[:num_eps], label="1 image", color="b")
plt.plot(range(num_eps), steps_5[:num_eps], label="Last 5 images", color="r")
# plt.plot(range(num_eps_rnn-1), steps_rnn[:-1], label="RNN", color="g")
plt.xlabel("Episode Number", fontsize=25)
plt.ylabel("Steps", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend()
plt.show()
avg_loss = loss/steps
avg_loss_5 = loss_5/steps_5
# avg_loss_rnn = loss_rnn/steps_rnn

plt.plot(range(num_eps), avg_loss[:num_eps], label="1 image", color="b")
plt.plot(range(num_eps), avg_loss_5[:num_eps], label="Last 5 images", color="r")
# plt.plot(range(num_eps_rnn-1), avg_loss_rnn[:-1], label="RNN", color="g")
plt.xlabel("Episode Number", fontsize=25)
plt.ylabel("Average loss per step", fontsize=25)

# plt.plot(range(num_eps_rnn-1), qval[:num_eps_rnn-1], label="Current state", color="b")
# plt.plot(range(num_eps_rnn-1), qval_5[:num_eps_rnn-1], label="Last 5 states", color="r")
# plt.plot(range(num_eps_rnn-1), qval_rnn[:-1], label="RNN", color="g")
# plt.xlabel("Episode Number", fontsize=25)
# plt.ylabel("Average loss per step", fontsize=25)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend()
plt.show()