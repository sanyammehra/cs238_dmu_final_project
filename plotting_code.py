import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pkl

dir_name = "results_original/"
num_eps = 1202

distance = np.zeros(num_eps)
reward = np.zeros(num_eps)
loss = np.zeros(num_eps)
steps = np.zeros(num_eps)

for fname in os.listdir(dir_name):
	if "reward" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0:
			r = pkl.load(open(dir_name+fname, "rb"))
			reward[idx-1] = r[0]
	if "distance" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0:
			d = pkl.load(open(dir_name+fname, "rb"))
			distance[idx-1] = d[0]
	if "loss" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0:
			l = pkl.load(open(dir_name+fname, "rb"))
			loss[idx-1] = l[0]

dir_name_5 = "results_5states_first 305episodes/"
num_eps_5 = 305

distance_5 = np.zeros(num_eps_5)
reward_5 = np.zeros(num_eps_5)
loss_5 = np.zeros(num_eps_5)
steps_5 = np.zeros(num_eps_5)

for fname in os.listdir(dir_name_5):
	if "reward" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps_5:
			r = pkl.load(open(dir_name+fname, "rb"))
			reward_5[idx-1] = r[0]
			print idx, reward[idx-1], reward_5[idx-1]
	if "distance" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps_5:
			d = pkl.load(open(dir_name+fname, "rb"))
			distance_5[idx-1] = d[0]
	if "loss" in fname:
		idx = int(fname.split("_")[1])
		if idx > 0 and idx < num_eps_5:
			l = pkl.load(open(dir_name+fname, "rb"))
			loss_5[idx-1] = l[0]

plt.plot(range(num_eps_5), reward[:num_eps_5], label="Current state", color="b")
# plt.plot(range(num_eps_5), reward_5, label="Last 5 states", color="r")
plt.legend()
plt.show()