import os
import datetime
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow_probability as tfp
import gym
import numpy as np
import time
import random
from collections import deque
from typing import Tuple, List
from multiprocessing import Manager, Process, Queue
import csv
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
# print(tf.__version__)

class ep_buffer:
    """
    Class that stores the state transition information of an episode
    """

    def __init__(self):
        self.memory = deque()

    def add_transition(self, transition: Tuple) -> None:
        """
        Arguments:
        transition -> Tuple (s, a, s', reward)
        """

        self.memory.append(transition)

    @staticmethod
    def compute_Qsa(rewards, gamma: float) -> List:  # O(n^2)
        """
        Computes the sample value function (ground truth) for every single state action pair of an episode

        Arguments:
        rewards -> object that contain all the rewards from the episode from t = 0 to t = len(rewards)
        gamma -> float, discount factor for the rewards

        Returns:
        Qsa -> List

        """
        Qsa = []
        for i in range(len(rewards)):
            partial_Qsa = 0
            t = 0
            for j in range(i, len(rewards)):

                partial_Qsa += rewards[j] * (gamma ** t)
                t += 1

            Qsa.append(partial_Qsa)

        return Qsa

    def unroll_memory(self, gamma):
        """
        Unrolls the states transitions information so that states , actions, next_states, rewards and Qsa's
        are separeted into different numpy arrays

        Returns:
        states -> numpy array (state dimension, num of state transitions)
        actions -> numpy array (action dimension, num of state transitions)
        next_states -> numpy array (state dimension, num of state transitions)
        rewards -> numpy array (num of state transitions, )
        qsa -> numpy array (num of state transitions, )
        """

        states, actions, next_states, rewards = zip(*self.memory)

        qsa = self.compute_Qsa(rewards, gamma)
        states = np.asarray(states)
        actions = np.asarray(actions)
        next_states = np.asarray(next_states)
        rewards = np.asarray(rewards)
        qsa = np.asarray(qsa, dtype=np.float32).reshape(-1, 1)

        # print(f"States: {states.shape}")
        # print(f"actions: {actions.shape}")
        # print(f"next_states: {next_states.shape}")
        # print(f"rewards: {rewards.shape}")
        # print(f"qsa: {qsa.shape}")
        self.memory = deque()
        return states, actions, next_states, rewards, qsa

    

    
def build_networks(layer_sizes, activations, input):
    num_layers = len(layer_sizes)
    output = keras.layers.Dense(units=layer_sizes[0], activation=activations[0], kernel_initializer='glorot_normal')(input)
    for i in range(1, num_layers):
        output = keras.layers.Dense(units=layer_sizes[i], activation=activations[i], kernel_initializer='glorot_normal')(output)
    
    return output
    
    
def build_model(input, output, name):
    return keras.Model(input, output, name=name)

    

class Agent(ep_buffer):
    def __init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 episode_queue,
                 parameters_queue,
                 current_iter, 
                 name,
                 n_episodes_worker,
                 gradient_steps_per_episode,
                 epsilon,
                 record_statistics):
        ep_buffer.__init__(self)
        

        self.gradient_clipping_actor = gradient_clipping_actor
        self.gradient_clipping_critic = gradient_clipping_critic
        self.trunk_config = trunk_config
        self.actor_mu_config = actor_mu_config
        self.actor_cov_config = actor_cov_config
        self.critic_config = critic_config
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.entropy = entropy
        self.action_space_bounds = action_space_bounds
        self.action_space_size = action_space_size
        self.action_bounds = action_space_bounds
        self.number_iter = number_iter
        self.max_steps = max_steps
        self.n_episodes_per_cycle = n_episodes_per_cycle        
        self.gamma = gamma
        self.env_name = env_name
        self.episode_queue = episode_queue
        self.parameters_queue = parameters_queue
        self.name = name
        self.current_iter = current_iter
        self.record_statistics = record_statistics
        self.gradient_steps_per_episode = gradient_steps_per_episode
        self.epsilon = epsilon
        self.n_episodes_worker = n_episodes_worker
        self.actions_taken = 0
        
    def build_models(self):
        
        self.input = keras.Input(shape=(8), name="state")
        self.trunk = build_networks(**self.trunk_config, input=self.input)
        mu_head = build_networks(**self.actor_mu_config, input=self.trunk)
        cov_head = build_networks(**self.actor_cov_config, input=self.trunk)
        critic = build_networks(**self.critic_config, input=self.input)
        self.actor_mu = build_model(self.input, mu_head, "actor_mu")
        self.actor_cov = build_model(self.input, cov_head, "actor_cov")
        self.critic = build_model(self.input, critic, "critic")
        
        index_last_layer = len(self.actor_cov.layers) -1
        
 
        self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                           "cov": [variable.numpy() for variable in self.actor_cov.get_layer(index=index_last_layer).trainable_variables],
                           "critic": [variable.numpy() for variable in self.critic.trainable_variables]
        
                           }
    
        self.variables =  {"mu": self.actor_mu.trainable_variables,
                           "cov": self.actor_cov.get_layer(index=index_last_layer).trainable_variables,
                           "critic": self.critic.trainable_variables}
        
        if self.name == "Global Agent":
            self.trunk_old = build_networks(**self.trunk_config, input=self.input)
            mu_head_old = build_networks(**self.actor_mu_config, input=self.trunk_old)
            cov_head_old = build_networks(**self.actor_cov_config, input=self.trunk_old)
            self.actor_mu_old = build_model(self.input, mu_head_old, "actor_mu_old")
            self.actor_cov_old = build_model(self.input, cov_head_old, "actor_cov_old")
            
            self.variables_old =  {"mu": self.actor_mu_old.trainable_variables,
                            "cov": self.actor_cov_old.get_layer(index=index_last_layer).trainable_variables}
            
            self.current_parameters_old =  {"mu": [variable.numpy() for variable in self.actor_mu_old.trainable_variables],
                           "cov": [variable.numpy() for variable in self.actor_cov_old.get_layer(index=index_last_layer).trainable_variables],
                           "critic": [variable.numpy() for variable in self.critic.trainable_variables]
                           }

        
    def collect_episodes(self, number_ep, max_steps, render=False):
        total_steps = 0
        total_reward = 0
        
        for ep in range(number_ep):
            prev_observation = self.env.reset()
            steps = 0
            done = False
            while done == False and steps <= max_steps:
                if render:
                    self.env.render()

                action, mu, cov = self.take_action(prev_observation.reshape(1, -1))
                if self.name == "Global Agent":
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.histogram(f"Mu-distribution", mu, self.actions_taken)
                            tf.summary.histogram(f"Cov-distibution", cov, self.actions_taken)
                            self.actions_taken += 1
                            
                action = action.numpy().reshape(self.action_space_size,)
                observation, reward, done, _ = self.env.step(action)
                steps += 1

                if steps == max_steps and not done:
                    value = self.state_value(observation.reshape(1, -1))
                    reward += value.numpy()[0]
                

                self.add_transition((prev_observation, action, observation, reward))
                prev_observation = observation
                total_reward += reward

            total_steps += steps

            self.env.close()
        return total_reward, steps
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 8]),))
    def state_value(self, state):
        value = self.critic(state)
        return value
        
        
        
    @tf.function
    def take_action(self, state):
        # state numpy
        
        
        mu = self.actor_mu(state)
        cov = self.actor_cov(state)
        
        #---START Record values for Average and covariance
      
            #---END Record values for Average and covariance
        
        probability_density_func = tfp.distributions.Normal(mu, cov) #cov
        action = probability_density_func.sample(1)
      
        action = tf.clip_by_value(action, self.action_bounds[0], self.action_bounds[1])

        
      
        return action, mu, cov
    
        
class GlobalAgent(Agent):
    def __init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 episode_queue,
                 parameters_queue,
                 current_iter,
                 name,
                 n_episodes_worker,
                 epsilon,
                 gradient_steps_per_episode,
                 record_statistics,
                 average_reward_queue,
                 number_of_childs,
                 save_checkpoints=False):
        
        Agent.__init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 episode_queue,
                 parameters_queue,
                 current_iter,
                 name,
                 n_episodes_worker,
                 gradient_steps_per_episode,
                 epsilon,
                 record_statistics)
        
        
        self.rewards = deque(maxlen=100)
        self.average_reward_queue = average_reward_queue
        self.number_of_childs = number_of_childs
        self.save_checkpoints = save_checkpoints
      
    def training_loop(self):
        try:
            self.build_models() # Create NN
            print(f"1 iter corresponds to {self.number_of_childs * self.n_episodes_worker} episodes and {self.gradient_steps_per_episode} gradient steps")
            self.env = gym.make(self.env_name)
            #---START Create a summary writer
            if self.record_statistics: 
                self.writer = tf.summary.create_file_writer(f"./summaries/global/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
            self.current_pass = 0
        #---END of Create summary writer
        #---START Load variable weights if self.save_checkpoints is activated
            if self.save_checkpoints:
                try:
                    self.actor_mu.load_weights("./saved_checkpoints/actor_mu/")
                    self.actor_cov.load_weights("./saved_checkpoints/actor_cov/")
                    self.critic.load_weights("./saved_checkpoints/critic/")
                except:
                    print("There was an error")
                    pass
            #---END Load variable weights if self.save_checkpoints is activated
        
            self.iter = 0
            self.gradient_steps = 0 # counts gradient steps instead of iterations, there are n steps per iter
            #--- START Main RL loop
            while self.current_iter.value < self.number_iter:
                
                for i in range(self.number_of_childs):
                    #Put enough parameters for all workers
                    self.parameters_queue.put(self.current_parameters_old, block=True, timeout=30)
            
                    
                #---START Record the values for the weights of policy gradient NN
                if self.record_statistics:
                    with self.writer.as_default():
                        for key, parameters in self.variables.items():
                            for variable in parameters:
                                tf.summary.histogram(f"Params_{self.name}_{str(key)}_{variable.name}", variable, self.iter)
                #---END Record the values for the weights of policy gradient NN 
                            
                #---START collect episodes available from all workers
                for i in range(self.number_of_childs * self.n_episodes_worker):
                    try:
                        episode = self.episode_queue.get(block=True, timeout=30)
                    except Exception as e: 
                        print(f"Error: {e}")
    
                
                    if i == 0:
                        states, actions, next_states, rewards, qsa = episode
                        
                    else:
                        states_temp, actions_temp, next_states_temp, rewards_temp, qsa_temp = episode
                        
                    
                        states = np.vstack((states, states_temp))
                        actions = np.vstack((actions, actions_temp))
            
                        qsa = np.vstack((qsa, qsa_temp))
                #---END collect episodes available from all workers

                    
                #---START gradient descent for actor Nets
                for i in range(self.gradient_steps_per_episode):
                    
                    gradients, entropy = self.train_step_actor(states, actions, qsa)
                    

                    #---START Cliping gradients
                    for key, gradient in gradients.items():
                        gradients[key] = [tf.clip_by_value(value, -self.gradient_clipping_actor, self.gradient_clipping_actor) for value in gradient]
                    #---END Clipping Gradients
                    
                    #---START Record gradient to summaries
                    if self.record_statistics:
                        with self.writer.as_default():
                                for key, gradient_list in gradients.items():
                                    for gradient, variable in zip(gradient_list, self.variables[key]):
                                
                                        tf.summary.histogram(f"Gradients_{self.name}_{str(key)}_{variable.name}", gradient, self.gradient_steps)

                    #---END Record gradient to summaries
                    #---Start Record average entropy for episodes
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"Entropy", entropy, self.gradient_steps)
                    #---End Record average entropy'for episodes
                    
                
                    #---START apply gradients for actor
                    for key, value in gradients.items():
                        self.actor_optimizer.apply_gradients(zip(value, self.variables[key]))
                    #---END apply gradients for actor
                    self.gradient_steps += 1
                #---END gradient descent for actor Nets
            
                #---START gradient descent for critic
                critic_gradient = self.train_step_critic(states, qsa)
                    #---START Gradient Clipping critic
                critic_gradient = [tf.clip_by_value(value, -self.gradient_clipping_critic, self.gradient_clipping_critic) for value in critic_gradient]
                    #---
                #---START Record gradient to summaries
                if self.record_statistics:
                    with self.writer.as_default():
                        for gradient, variable in zip(critic_gradient, self.variables["critic"]):
                            tf.summary.histogram(f"Gradients_{self.name}_critic_{variable.name}", gradient, self.iter)

                    #---END Record gradient to summaries
                    #---START Aplly Critic Gradients
                self.critic_optimizer.apply_gradients(zip(critic_gradient, self.variables["critic"]))
                    #---
                #---END gradient descent for critic
            
            
                            
                index_last_layer = len(self.actor_cov.layers) - 1 #find the index of the last layer 
                
                #---START update self.current_parameter with the parameters resulting from n steps of gradient descent
                self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                            "cov": [variable.numpy() for variable in self.actor_cov.get_layer(index=index_last_layer).trainable_variables],
                            "critic": [variable.numpy() for variable in self.critic.trainable_variables]
                            }
                #---END update self.current_parameter with the parameters resulting from n steps of gradient descent
                
                #---START Update Old policy seting theta_old = theta
                for key, value in self.current_parameters.items():
                    if key != "critic":
                        for n, variable in enumerate(self.variables_old[key]):
                            variable.assign(value[n])
                #---END Update Old policy seting theta_old = theta
                
                #---START update self.current_parameter_old with 
                self.current_parameters_old = self.current_parameters
                
                #---START Observe the value of given state to see convergence
                state = np.array([-0.0073947, 1.4089943, -0.7490139, -0.08562016, 0.00857535, 0.1696628, 0.0, 0.0])
                value = self.state_value(state.reshape(1, -1))
                value = float(value.numpy()[0])
            
                if self.record_statistics:
                    with self.writer.as_default():
                        tf.summary.scalar(f"State_value", value, self.iter)
                #---END Observe the value of given state to see convergence
                
                #---START after n iterations RUN EPISODE and PRINT REWARD
                if self.iter % 100 == 0:
                    rewards_volatile = []
                    steps_ep = []
                    for i in range (10):
                        reward, steps = self.collect_episodes(1, self.max_steps, render=False)
                        rewards_volatile.append(reward)
                        steps_ep.append(steps)
                    av_reward = sum(rewards_volatile) / len(rewards_volatile)
                    max_reward = max(rewards_volatile)
                    min_reward = min(rewards_volatile)
                    average_steps = sum(steps_ep) / len(steps_ep)
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"Averge_Reward", av_reward, self.iter)
                            tf.summary.scalar(f"Max_Reward", max_reward, self.iter)
                            tf.summary.scalar(f"Min_Reward", min_reward, self.iter)
                            tf.summary.scalar(f"Average_Length", average_steps, self.iter)
                    self.rewards.append(av_reward)
            
                
                    print(f"Iter: {self.iter}: Average: {av_reward} -- Max: {max_reward} -- Min {min_reward} ")
                    
                
                #---END after n iterations of the loop run episode and print reward
                
                #---START Save weights at current iter
                if self.save_checkpoints:
                    self.actor_mu.save_weights("./saved_checkpoints/actor_mu/")
                    self.actor_cov.save_weights("./saved_checkpoints/actor_cov/")
                    self.critic.save_weights("./saved_checkpoints/critic/")
                    
                
                                
                
                self.current_iter.value += 1
                self.current_pass += 1
                self.iter += 1
            #--- END Main RL loop
            with open("Running_Log.csv", "a") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow(["Run", self.iter])
                
                for i in range (10):
                    reward, steps = self.collect_episodes(1, self.max_steps, render=False)
                    rewards_volatile.append(reward)
                    steps_ep.append(steps)
                av_reward = sum(rewards_volatile) / len(rewards_volatile)
                max_reward = max(rewards_volatile)
                min_reward = min(rewards_volatile)
                average_steps = sum(steps_ep) / len(steps_ep)
                
                writer.writerow([f"Average: {av_reward} -- Max: {max_reward} -- Min {min_reward}"])
            
            self.average_reward_queue.put(sum(self.rewards) / len(self.rewards), block=True, timeout=30)
        except KeyboardInterrupt:
            #---After all steps are run average out the last 100 rewards and put it on a queue
            print("Wait until summary of partial run is updated to Running_log.csv")
            with open("Running_Log.csv", "a") as file:
                
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow(["Run", self.iter])
                    
                    for i in range (10):
                        reward, steps = self.collect_episodes(1, self.max_steps, render=False)
                        rewards_volatile.append(reward)
                        steps_ep.append(steps)
                    av_reward = sum(rewards_volatile) / len(rewards_volatile)
                    max_reward = max(rewards_volatile)
                    min_reward = min(rewards_volatile)
                    average_steps = sum(steps_ep) / len(steps_ep)
                    
                    writer.writerow([f"Average: {av_reward} -- Max: {max_reward} -- Min {min_reward}"])
            print("Press ctr + C one last time. Summary has be saved!")
            
            self.average_reward_queue.put(sum(self.rewards) / len(self.rewards), block=True, timeout=30)
        
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 8]), tf.TensorSpec(shape=[None, 2]), tf.TensorSpec(shape=[None, 1])))
    def train_step_actor(self, states, actions, Qsa):
        with tf.GradientTape(persistent=True) as tape:
            #---START Actor gradient calculation
            #---START Get the parameters for the Normal dist
            mu = self.actor_mu(states)
            cov = self.actor_cov(states)
            mu_old = tf.stop_gradient(self.actor_mu_old(states))
            cov_old = tf.stop_gradient(self.actor_cov_old(states))
            #---END Get the parameters for the Normal dist
            #---START Advantage function computation and normalization
            advantage_function = Qsa - self.critic(states)
            advantage_function_mean = tf.math.reduce_mean(advantage_function)
            advantage_function_std = tf.math.reduce_std(advantage_function)
            advantage_function = tf.math.divide((advantage_function - advantage_function_mean), (advantage_function_std + 1.0e-5))
            #---END Advantage function computation and normalization
            #---START compute the Normal distributions
            self.probability_density_func = tfp.distributions.Normal(mu, cov)
            self.probability_density_func_old = tfp.distributions.Normal(mu_old, cov_old)
            #---END compute the Normal distributions
            #---Entropy
            entropy = self.probability_density_func.entropy()
            entropy_average = tf.reduce_mean(entropy)
            
            #---
            #---START compute the probability of the actions taken at the current episode
            log_probs = self.probability_density_func.log_prob(actions)
            log_probs_old = tf.stop_gradient(self.probability_density_func_old.log_prob(actions))
            #---END compute the probability of the actions taken at the current episode
            #---START Ensemble Actor loss function
            self.probability_ratio = tf.math.exp(log_probs - log_probs_old)
            cpi = tf.math.multiply(self.probability_ratio, tf.stop_gradient(advantage_function))
            clip = tf.math.minimum(cpi, tf.multiply(tf.clip_by_value(self.probability_ratio, 1 - self.epsilon, 1 + self.epsilon), tf.stop_gradient(advantage_function)))
            actor_loss = -tf.reduce_mean(clip) + entropy
            #---END Ensemble Actor loss function
        
        #---START Compute gradients for average
        gradients_mu = tape.gradient(actor_loss, self.actor_mu.trainable_variables)
        #---
        
        #---START Compute gradients for the covariance
        last_layer_index = len(self.actor_cov.layers) - 1 
        gradients_cov = tape.gradient(actor_loss, self.actor_cov.get_layer(index=last_layer_index).trainable_variables)
        # END Compute gradients for the covariance
        
        gradients = {"mu": gradients_mu,
                     "cov": gradients_cov,}
        #---END Actor gradient calculation
        
          
        return gradients, entropy_average
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 8]), tf.TensorSpec(shape=[None, 1])))
    def train_step_critic(self, states, Qsa):
        with tf.GradientTape(persistent=True) as tape:
           
            critic_cost = tf.losses.mean_squared_error(Qsa, self.critic(states))
        
        gradients_critic = tape.gradient(critic_cost, self.critic.trainable_variables)
        
        return gradients_critic
            
        
class WorkerAgent(Agent):
   
    def __init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 episode_queue,
                 parameters_queue,
                 current_iter,
                 epsilon,
                 n_episodes_worker,
                 gradient_steps_per_episode,
                 name,
                 record_statistics): 
        
        Agent.__init__( self,
                        trunk_config,
                        actor_mu_config,
                        actor_cov_config,
                        critic_config, 
                        actor_optimizer,
                        critic_optimizer,
                        entropy,
                        action_space_bounds,
                        action_space_size,
                        number_iter,
                        max_steps,
                        n_episodes_per_cycle,
                        gamma,
                        env_name,
                        state_space_size,
                        gradient_clipping_actor,
                        gradient_clipping_critic,
                        episode_queue,
                        parameters_queue,
                        current_iter,
                        name,
                        n_episodes_worker,
                        gradient_steps_per_episode,
                        epsilon,
                        record_statistics)
       

        
    def training_loop(self):
        #---START build NETworks for critic and actor
        self.build_models()
        #---
        #---Create gym environment
        self.env = gym.make(self.env_name)
        #---
        self.number_passes = 0 # to count the number of iterations done within worker
        
        rewards_collection = deque(maxlen=100) # stores rewards at every episode
        #---START Main loop 
        while self.current_iter.value < self.number_iter:
            self.iter = self.current_iter.value
            #---Update variables with information coming from gradient descent
            self.update_variables() 
            #---
            #---START Collect n episodes from this worker
            for ep in range(self.n_episodes_worker): #Run more than 1 episode for each each gradient descent step
               
                reward_ep, _ = self.collect_episodes(self.n_episodes_per_cycle, self.max_steps)
                states, actions, next_states, rewards, qsa = self.unroll_memory(self.gamma)
                rollout = (states, actions, next_states, rewards, qsa)

                self.episode_queue.put(rollout)
                rewards_collection.append(reward_ep)
            #---END Collect n episodes from this worker     
           
            self.number_passes += 1


    def update_variables(self):
        #---Get current parameters from queue(put in by the global agent)
        try:
            self.new_params = self.parameters_queue.get(block=True, timeout=30)
        except Exception as e:
            print(e)
        
        #---
        #---START assign the variables of the worker with the variable values from global
        for key, value in self.new_params.items():
            for n, variable in enumerate(self.variables[key]):
                variable.assign(value[n])
        #END assign the variables of the worker with the variable values from global

        
        #---START update variable current_parameters to reflect the information provided by global
        index_last_layer = len(self.actor_cov.layers) - 1 
        self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                        "cov": [variable.numpy() for variable in self.actor_cov.get_layer(index=index_last_layer).trainable_variables],
                        "critic": [variable.numpy() for variable in self.critic.trainable_variables]
    
                        }
        #---END update variable current_parameters to reflect the information provided by global
                
            

trunk_config = {
   
    "layer_sizes": [100, 100],
    "activations": [ "relu", "relu"],
}



mu_head_config = {
    "layer_sizes":[2],
    "activations": ["tanh"]
    }

cov_head_config = {
    "layer_sizes":[2],
    "activations": ["sigmoid"],
  
    }

critic_net_config= {
    "layer_sizes":[100, 64, 1],
    "activations": ["relu", "relu", "linear"],
    }


hyperparameters = { "trunk_config": trunk_config,
                    "actor_mu_config": mu_head_config,
                    "actor_cov_config":cov_head_config, 
                    "critic_config": critic_net_config,
                    "actor_optimizer": tf.keras.optimizers.SGD(learning_rate=0.001),
                    "critic_optimizer": tf.keras.optimizers.SGD(learning_rate=0.01),
                    "entropy":0.05,
                    "gamma":0.999,
                    "gradient_clipping_actor": 0.08, 
                    "gradient_clipping_critic": 0.2, 
                    "gradient_steps_per_episode": 3,
                    "epsilon": 0.2,
                    "n_episodes_worker": 3
                    }

agent_configuration = {

    "action_space_bounds":[-1, 1],
    "action_space_size":2,
    "number_iter":10000,
    "max_steps":500,
    "n_episodes_per_cycle":1,
    "env_name":"LunarLanderContinuous-v2",
    "state_space_size":8,
}



if __name__ == '__main__':
    
    number_of_workers = 2
    params_queue = Manager().Queue(number_of_workers)
    episode_queue = Manager().Queue()
    current_iter = Manager().Value("i", 0)
    average_reward_queue = Queue(1)
    
    
    global_agent = GlobalAgent(**agent_configuration, 
                               **hyperparameters,
                            episode_queue=episode_queue,
                            parameters_queue = params_queue,
                            current_iter=current_iter,
                            name="Global Agent", 
                            record_statistics=True,
                            average_reward_queue=average_reward_queue,
                            number_of_childs=number_of_workers,
                            save_checkpoints=True)
    
    
    workers = [WorkerAgent(**agent_configuration, **hyperparameters, episode_queue=episode_queue, parameters_queue = params_queue, current_iter=current_iter, name=f"Worker_{_}", record_statistics=False) for _ in range(number_of_workers)]
    
    processes = []
    p1 = Process(target=global_agent.training_loop)
    processes.append(p1)
    p1.start()

    for worker in workers:
        
        p = Process(target=worker.training_loop)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    print(average_reward_queue.get())
    print("Simulation Run")
        
        
        
