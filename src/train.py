"""
Agent train loop script.
"""
import time
import torch
import numpy as np
from memory_buffer import ReplyBuffer
from agent import DQNAgent
from hyperparameters import get_hyperparams
import wandb

wandb.login(key='50315889c64d6cfeba1b57dc714112418a50e134')

def train():
    """
    Training loop function.
    The code iterates through the episodes and then the timesteps 
    within the episodes. The agent stops training based on 
    some stopping condition like a max episode value, a max 
    timestep value, or if the agent's performance reaches a certain level.

    In each episode and at each timestep the agent selects an action, 
    calls env.step(), and stores the sample in the replay buffer. 
    The agent will also train the training neural network and periodically 
    update the target network. When an episode is done the env is reset.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device))

    start_time = time.time()

    params, env = get_hyperparams()

    # start a new wandb run to track this script
    wandb.init(  
        project="RL-Pong_nt_5_000_000",  # set the wandb project where this run will be logged
        config= params,  # track hyperparameters and run metadata
        name='default_params'
    )

    reply_buffer = ReplyBuffer(buffer_size=params['buffer_size'])
    dqn_agent = DQNAgent(input_shape=params['frame_stack'],
                         nb_actions=params['nb_actions'],
                         device=device, params=params, env=env)

    stats_reward_list = []  # Store stats for ploting
    stats_update = 10  # print stats every n episodes
    total_reward = 0
    episodes = 1
    episode_length = 0
    stats_loss = .0
    state = env.reset()
    final_time = time.time()
    total_time = (final_time - start_time) / 3600
    eps_decay = 0.999985

    # epsilon_decay_steps = params['n_timestamps'] * params['exploration_fraction']
    # epsilon_step = (params['epsilon_start'] - params['exploration_final_eps'] /
    #                 epsilon_decay_steps)

    epsilon = params['epsilon_start']

    for time_stamp in range(params['n_timestamps']):
        action = dqn_agent.select_actions(state, epsilon)

        # decay epsilon
        # epsilon -= epsilon_step
        # epsilon = (params['exploration_final_eps'] if epsilon < params['exploration_final_eps']
        #            else epsilon)
        epsilon = max(epsilon* eps_decay , params['exploration_final_eps'])

        # enter action into the env
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        episode_length += 1

        # add experience to buffer
        reply_buffer.add_data((state, next_state, action, reward, float(done)))

        if time_stamp > params['learning_starts']:

            # Start Agent training
            stats_loss += dqn_agent.train(reply_buffer, params['batch_size'], params['discount'])
            # print('training')
            # Update target network every n stats update when the conditions in the
            # update_target_function are met.
            dqn_agent.update_target_network(time_stamp, update_every=stats_update)

        if done:
            state = env.reset()
            stats_reward_list.append((episodes, total_reward, episode_length))
            episodes += 1
            total_reward = 0
            episode_length = 0

            if time_stamp > params['learning_starts'] and episodes % stats_update == 0:
                # Wandb config
                final_time = time.time()
                total_time = (final_time - start_time) / 3600
                print(f'''Episode: {episodes}
                      Timestep: {time_stamp}
                      Total reward: {round(np.mean(stats_reward_list[-stats_update:], axis=0)[1], 1)}
                      Episode length: {round(np.mean(stats_reward_list[-stats_update:], axis=0)[2], 1)}
                      Epsilon: {round(epsilon, 2)}
                      Loss: {round(stats_loss, 4)}
                      Duration {round(total_time, 2)}''')

                stats_loss = .0

                # Save weights
                if episodes % params['save_weights_every'] == 0:
                    torch.save(dqn_agent.train_network.state_dict(),
                               f'C:/Users/Xavi/Documents/MCV_code/MCV_reinforcement/weights/train_network_episode_{episodes}5_000_000.pt')

                    if dqn_agent.target_network:
                        torch.save(dqn_agent.target_network.state_dict(),
                                   f'C:/Users/Xavi/Documents/MCV_code/MCV_reinforcement/weights/target_network_episode_{episodes}5_000_000.pt')

                if time_stamp % params['n_timestamps'] == 0:
                    torch.save(dqn_agent.train_network.state_dict(),
                               'C:/Users/Xavi/Documents/MCV_code/MCV_reinforcement/weights/train_network_episode_final_5_000_000.pt')

                    torch.save(dqn_agent.target_network.state_dict(),
                                   'C:/Users/Xavi/Documents/MCV_code/MCV_reinforcement/weights/target_network_episode_final_5_000_000.pt')

            if len(stats_reward_list) > stats_update and \
                np.mean(stats_reward_list[-stats_update:], axis=0)[1] > 20:
                print(f"""Stopping at episode {episodes} with average reward of
                      {round(np.mean(stats_reward_list[-stats_update:], axis=0)[1], 1)} 
                       in last {stats_update} episodes.""")
                break

        else:
            state = next_state

        wandb.log({"episodes": episodes,
                   "reward": total_reward,
                   "episode_length": episode_length,
                   "epsilon": epsilon,
                   "loss": stats_loss,
                   "time": total_time})
                

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == '__main__':
    train()