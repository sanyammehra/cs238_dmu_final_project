from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

import cPickle as pkl

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    # state_dim = 29  #of sensors input
    # state_dim = 29*5  #of sensors input - last 5 states
    state_dim = 29
    num_states = 5

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 10000 #100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, num_states, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, num_states, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
    # samp = env.observation_space.sample()
    #Now load the weight
    # print("Now we load the weight")
    # try:
    #     actor.model.load_weights("actormodel.h5")
    #     critic.model.load_weights("criticmodel.h5")
    #     actor.target_model.load_weights("actormodel.h5")
    #     critic.target_model.load_weights("criticmodel.h5")
    #     print("Weight load successfully")
    # except:
    #     print("Cannot find the weight")

    # Evaluation Metrics
    reward_arr = []
    loss_arr = []
    target_q_arr = []
    distance_arr = []
    steps_arr = []

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)*5)
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        s = s_t
        for si in range(num_states-1):
            s_t = np.vstack((s_t, s))

        total_reward = 0.
        loss_episode = 0.0
        target_q_episode = []
        distance_episode = 0.0

        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            # a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0], s_t.shape[1]))

            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info, dist = env.step(a_t[0])
            distance_episode += dist

            # s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            # s_t1 = np.hstack((tuple(s_t[-116:]),np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))))
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            s_t1 = np.vstack((s_t[-(num_states-1):], s_t1))

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
            
            qval = np.argmax(np.array(target_q_values),1)
            target_q_episode.append(qval)

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                loss_episode += loss
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        if np.mod(i, 50) == 0:
            if (train_indicator):
                actor.model.save_weights("results/actormodel_"+str(i)+".h5", overwrite=True)
                with open("results/actormodel_"+str(i)+".json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("results/criticmodel_"+str(i)+".h5", overwrite=True)
                with open("results/criticmodel_"+str(i)+".json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        if (train_indicator): 
            pkl.dump(reward_arr, open("results/reward_"+str(i),"wb"))
            pkl.dump(loss_arr, open("results/loss_"+str(i),"wb"))
            pkl.dump(target_q_arr, open("results/target_q_"+str(i),"wb"))
            pkl.dump(distance_arr, open("results/distance_"+str(i),"wb"))
            pkl.dump(steps_arr, open("results/steps_"+str(i),"wb"))

            del reward_arr
            del loss_arr
            del target_q_arr
            del distance_arr
            del steps_arr

            reward_arr = []
            loss_arr = []
            target_q_arr = []
            distance_arr = []
            steps_arr = []
                
        reward_arr.append(total_reward)
        loss_arr.append(loss_episode)
        target_q_arr.append(target_q_episode)
        distance_arr.append(distance_episode)
        steps_arr.append(step)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    pkl.dump(reward_arr, open("results/reward","wb"))
    pkl.dump(loss_arr, open("results/loss","wb"))
    pkl.dump(target_q_arr, open("results/target_q","wb"))
    pkl.dump(distance_arr, open("results/distance","wb"))
    pkl.dump(steps_arr, open("results/steps","wb"))

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
