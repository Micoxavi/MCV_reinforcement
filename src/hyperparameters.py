"""
Script which contain the hyperparameters for the Pong Enviroment
"""
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from supersuit import frame_stack_v1


def get_hyperparams() -> dict:
    """
    Function to configure the different hyperparameters for the RL Pong task.

    :Return:
        params: Dictionary containing the different parameters:
        {
        batch_size: int
        buffer_size: int
        env_wrapper: stable_baselines3 . common . atari_wrappers . AtariWrapper
        exploration_final_eps: float
        exploration_fraction: float
        frame_stack: int
        gradient_steps: int
        learning_rate: float
        learning_starts: int
        n_timesteps: float
        optimize_memory_usage: float
        policy: str
        target_update_interval: int
        train_freq: int
        }
    """
    env = AtariWrapper(gym.make('PongNoFrameskip-v4'))
    env = frame_stack_v1(env , 4)

    # @exploration_fraction --> the fraction of total timesteps over which the
                              # exploration rate is annealed from 1.0 to "exploration_final_eps".
                              # By default on the introduction is set to 0.1, which means exploration
                              # will be annealed over the first 10% of total timesteps.

    params = {

        # Parameters given by default on the exercise training introduction
        'env': env,
        'batch_size': 32,
        'buffer_size': 100_000,
        'exploration_final_eps': 0.01, # epsilon min
        'exploration_fraction': 0.1,  
        'frame_stack': 4,  # default on AtariWraper
        'gradient_steps': 1,
        'learning_rate': 0.000_1,
        'learning_starts': 100_000,
        'n_timesteps': 10_000_000.0,
        'target_update_interval': 1000,
        'train_freq': 4,

        # Other usefull parameters found on internet
        'discount': 0.99,
        'hidden_size': 512,
        'epsilon_start': 1.0,
        'nb_actions': env.action_space.n
    }
    return params
