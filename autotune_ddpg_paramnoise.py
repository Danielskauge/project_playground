import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp
import hyperopt
from param_noise import *


problem = "Pendulum-v1"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, gamma
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, gamma):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, gamma)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor(hidden_layers_shape):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(hidden_layers_shape[0], activation="relu")(inputs)
    out = layers.Dense(hidden_layers_shape[1], activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(hidden_layers_shape):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(hidden_layers_shape[0], activation="relu")(concat)
    out = layers.Dense(hidden_layers_shape[1], activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
    
def policy(state, noise_object,epsilon):
    sampled_actions = tf.squeeze(perturbed_actor(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + epsilon*noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def perturb_actor_parameters():
    """Apply parameter noise to actor model, for exploration"""
    global actor_perturbed
    actor_perturbed.set_weights(actor_model.get_weights())
    params = perturbed_actor.state_dict()
      for name in params:
           if 'ln' in name:
                pass
            param = params[name]
            random = torch.randn(param.shape)
            param += random * param_noise.current_stddev

#find out how these funcitons oppose/compliment each other
@tf.function
def update_perturbed_actor(actor, perturbed_actor, param_noise_stddev):

    for var, perturbed_var in zip(actor.variables, perturbed_actor.variables):
        if var in actor.perturbable_vars:
            perturbed_var.assign(var + tf.random.normal(shape=tf.shape(var), mean=0., stddev=param_noise_stddev))
        else:
            perturbed_var.assign(var)




#held constant through auto hyperparameter optimization
total_episodes = 100

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

hidden_layers_shape = (400,300)

buffer_size = 50000
batch_size = 64

epsilon = 1
min_epsilon = 0.01

noise_std_dev = 0.2

# To store reward history of each episode

critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)


def loss_given_hyperparams(params):
    tau = params['tau']
    actor_lr = params['actor_lr']
    critic_lr = params['critic_lr']
    gamma = params['gamma']
    noise_std_dev = params['noise_std_dev']

    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    buffer = Buffer(buffer_size, batch_size)

    global actor_model
    global perturbed_actor
    global critic_model
    global target_actor
    global target_critic

    actor_model = get_actor(hidden_layers_shape)
    critic_model = get_critic(hidden_layers_shape)

    #add parameter noise to actor
    perturbed_actor = perturb_actor_parameters()

    target_actor = get_actor(hidden_layers_shape)
    target_critic = get_critic(hidden_layers_shape)

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3, adaptation_coefficient=1.05)

    # Takes about 4 min to train
    for ep in range(total_episodes):

        epsilon -= (epsilon/EXPLORE)
        epsilon = np.maximum(min_epsilon,epsilon)
        prev_state = env.reset()
        episodic_reward = 0

        perturbed_actor = perturb_actor_parameters()

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise, epsilon)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            buffer.learn(gamma)

            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = -np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    return avg_reward_list[-1]

def calculate_optimal_params():
    print("Starting hyperparam-optimization")

    params = {'tau': hp.uniform('tau', 0.001, 0.01),
        'critic_lr': hp.uniform('critic_lr',0.0002,0.02),
        'actor_lr': hp.uniform('actor_lr',0.0001,0.01),
        'gamma': hp.uniform('gamma',0.9,0.999),
        'noise_std_dev': hp.uniform('noise_std_dev',0.1,0.4), }

    best = fmin(loss_given_hyperparams, params, algo=hyperopt.rand.suggest, max_evals=100)

    print(best)
    print(hyperopt.space_eval(params, best))

calculate_optimal_params()
