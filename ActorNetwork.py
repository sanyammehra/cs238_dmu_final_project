import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, GRU, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, num_states, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state, self.states_images = self.create_actor_network(state_size, num_states, action_size)   
        self.target_model, self.target_weights, self.target_state, self.target_state_images = self.create_actor_network(state_size, num_states, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads, states_images):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads,
            self.states_images: states_images
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, num_states, action_dim):
        print("Now we build the model")
        print("Changed model")
        S = Input(shape=[num_states, state_size]) 
        I = Input(shape = [15, 64,64])
        conv1 = Conv2D(32, 3,3,activation = 'relu',dim_ordering = 'th', input_shape = (3,64,64), border_mode = 'same')(I)
        mp1 = MaxPooling2D(pool_size = (2,2), dim_ordering = 'th')(conv1)
        conv2 = Conv2D( 16,3,3,activation = 'relu',dim_ordering = 'th', input_shape = (3,64,64), border_mode = 'same')(mp1)
        mp2 = MaxPooling2D(pool_size = (2,2), dim_ordering = 'th')(conv2)
        mp2_flatten = Flatten()(mp2)
        im_h = Dense(32, activation = 'relu')(mp2_flatten)
        x = GRU(32, return_sequences=False, name='gru1')(S)
        x_merge = merge([im_h,x], mode = 'concat')
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(x_merge)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)  
        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
        V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(input=[S,I],output=V)
        return model, model.trainable_weights, S, I

