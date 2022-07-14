def adapt_param_noise(self, obs0):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        mean_distance = self.get_mean_distance(obs0).numpy()

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(mean_distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

        self.param_noise.adapt(mean_distance)
        return mean_distance

    @tf.function
    def get_mean_distance(self, obs0):
        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        update_perturbed_actor(self.actor, self.perturbed_adaptive_actor, self.param_noise.current_stddev)

        normalized_obs0 = tf.clip_by_value(normalize(obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        actor_tf = self.actor(normalized_obs0)
        adaptive_actor_tf = self.perturbed_adaptive_actor(normalized_obs0)
        mean_distance = tf.sqrt(tf.reduce_mean(tf.square(actor_tf - adaptive_actor_tf)))
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            update_perturbed_actor(self.actor, self.perturbed_actor, self.param_noise.current_stddev)

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


