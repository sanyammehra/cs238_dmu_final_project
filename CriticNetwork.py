import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, GRU,  Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, state_size, num_states, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state, self.states_images = self.create_critic_network(state_size, num_states, action_size)  
        self.target_model, self.target_action, self.target_state, self.target_state_images = self.create_critic_network(state_size, num_states, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions, states_images):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions,
            self.states_images: states_images
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, num_states, action_dim):
        print("Now we build the model")
        S = Input(shape=[num_states, state_size])  
        A = Input(shape=[action_dim],name='action2')
        I = Input(shape = [15,64,64])
        
        conv1 = Conv2D(32, 3,3,activation = 'relu',dim_ordering = 'th', input_shape = (3,64,64), border_mode = 'same')(I)
        mp1 = MaxPooling2D(pool_size = (2,2), dim_ordering = 'th')(conv1)
        conv2 = Conv2D( 16,3,3,activation = 'relu',dim_ordering = 'th', input_shape = (3,64,64), border_mode = 'same')(mp1)
        mp2 = MaxPooling2D(pool_size = (2,2), dim_ordering = 'th')(conv2)
        mp2_flatten = Flatten()(mp2)
        im_h = Dense(32, activation = 'relu')(mp2_flatten)

        x = GRU(32, return_sequences=False, name='gru1')(S)  
        x_merge = merge([im_h,x], mode = 'concat')

        w1 = Dense(HIDDEN1_UNITS, activation='relu')(x_merge)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')    
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)   
        model = Model(input=[S,A,I],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S, I
