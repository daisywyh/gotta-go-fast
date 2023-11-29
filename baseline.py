"""
Setup: setup all of the imports we need to do for this project
"""

# PyTorch imports
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# OpenAI Gym Imports
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# Import NES Emulator for Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

# Import Data Structures
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# Import Numpy
import numpy as np

# Import Time module
import time, datetime

# Import Matplotlib
import matplotlib.pyplot as plt

"""
Definitions:
- Environment: Super Mario Bros NES
- Action Space: < Movement Options >
- State Space: < The World and Mario's position / velocity / characteristics in that World
- Reward: Use pre-made Reward Function

Other Information:
- self.information: {
  "coins"     : (int)   - The number of coins held at this current observation
  "flag_get'  : (bool)  - Whether or not the flag has been reached
  "life"      : (int)   - The number of lives Mario has (life <= 1)
  "score"     : (int)   - The listed in-game score at the current observation
  "stage"     : (int)   - The current stage number. (stage in [1,4])
  "status"    : (str)   - Mario's status (status in {'small', 'big', 'fire'(?)})
  "time"      : (int)   - The remaining in-game time at the current observation
  "world"     : (int)   - The current world number. (world in [1,8])
  "x_pos"     : (int)   - Mario's X position
  "y_pos"     : (int)   - Mario's Y position
}

Define additional parameters for future use:
"""
OLD_ACTION_SPACE = [

  # Singular Buttons
  ["right"],        # WALK to the RIGHT
  ["left"],         # WALK to the LEFT
  ["down"],         # CROUCH / ENTER PIPE
  ["B"],            # SPECIAL in place

  # Jump Heights
  ["A"],                       # JUMP in place
  ["A", "A", "A"],             # MID-JUMP
  ["A", "A", "A", "A", "A"],   # BIGGER Jump

  # Speed Up
  ["right", "B"],        # RUN to the RIGHT
  ["left", "B"],         # RUN to the LEFT

]

ACTION_SPACE = [

  # Singular Buttons
  ["right"],        # WALK to the RIGHT
  ["left"],         # WALK to the LEFT
  ["down"],         # CROUCH / ENTER PIPE
  ["B"],            # SPECIAL in place
  ["A"],            # JUMP in place

]

RIGHT_ONLY = [
  [],                              # NO input  
  ["right"],                       # WALK to the RIGHT
  ["right", "B"],                  # RUN to the RIGHT
  ["right", "A"],                  # JUMP while moving RIGHT
  ["right", "A", "A", "A"],        # BIGGER JUMP while moving RIGHT
  ["right", "A", "A", "A", "A", "A"],   # EVEN BIGGER JUMP while moving RIGHT
]

BASELINE = [
  [],                              # NO input  
  ["right"],                       # WALK to the RIGHT
  ["right", "B"],                  # RUN to the RIGHT
  ["right", "A"],                  # JUMP while moving RIGHT
]

"""
Initialize the Environment
"""
# Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen), else 'rgb'
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)


"""
Initialize the Action Space

Currently limit the action-space to
- 0. walk right (right)
- 1. jump right (right + jump)
"""
# env = JoypadSpace(env, [["right"], ["right", "A"]])
env = JoypadSpace(env, ACTION_SPACE)

"""
Reset the Environment to Initialize

reset() returns the environment to an INITIAL STATE and returns
the initial observation of that environment.

reset() -> (state, information)
"""
env.reset()

"""
Takes one step in the environment, with some action in Actions.

step() returns the next observation (next_state) of the environment, the reward
given action, a boolean representing whether or not a terminal state was reached,
a boolean representing truncation (???), and a dictionary of new information.
"""
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

#####################################################################################
                              # Preprocessor Classes #                           
#####################################################################################
class SkipFrame(gym.Wrapper):
    """
    Idea: We don't need EVERY frame. Therefore, let us cut our computations down by
    skipping some frames.

    This is because every observation is a (3, 240, 256) size array.
    - [0, 255] for R/G/B values
    - 240px X direction
    - 256px Y direction

    Every frame skip makes our computation much more efficient, at the cost of
    information loss.
    """
    def __init__(self, env, skip):
          """Return only every `skip`-th frame"""
          super().__init__(env)
          self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

class GrayScaleObservation(gym.ObservationWrapper):
    """
    For the purposes of our training, we don't derive much information
    from the colors provided in our observations. Therefore, let us
    downsample our observation of color values and convert our observation
    into Black and White.

    This process trims our observation from (3, 240, 256) to (1, 240, 256).
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    """
    Additionally, we likely do not need the entire screen to make
    optimal decisions.

    This class will downsample our observation from (1, 240, 256) to (1, 84, 84)
    around Mario.
    """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)              # Apply SkipFrame wrapper to current observation
env = GrayScaleObservation(env)           # Downscale observation to black and white
env = ResizeObservation(env, shape=84)    # Downscale observation to a 84x84 square.
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

#####################################################################################
                                # Agent Class #                           
#####################################################################################
"""
As an Agent, Mario must be able to perform the following:
- ACT according to the optimal action policy based on the current state (of the environment).
- RECALL previous experiences (defined as (state, action, reward, state')) to update policy.
- LEARN better policies over time.

The required functions are:
- act(state)          : Choose an epsilon-greedy action
- cache(experience)   : Add current experience to memory
- recall()            : Sample experiences from memory
- learn()             : Update our approximate Q function using experiences
"""
class Mario:
    """
    
    """
    def __init__(self, state_dim, action_dim, save_dir, load_chkpt = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # Loading Model from Checkpoint
        self.chkpt_fpath = load_chkpt

        cuda_ = "cuda:0"
        self.device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define memory for RECALL and LEARN
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        # self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=self.device))
        
        self.batch_size = 32

        # Define variables for Mario's DNN learning
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Load previous model if passed in
        if not self.chkpt_fpath is None:
            checkpoint = torch.load(self.chkpt_fpath)
            self.net.load_state_dict(checkpoint['model'])
            self.exploration_rate = checkpoint['exploration_rate']

        self.save_every = 5e5  # num. of experiences between saving Mario Net

        # Discount factor for TD
        self.gamma = 0.9

        # Updating the model
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx


    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x): return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
      
    def td_estimate(self, state, action):
        """
        The current optimal Q-value for a given (state, action).
        """
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        The sum of current reward and the estimated Q-value in (state', action)
        """
        next_state_Q = self.net(next_state, model="online")  # state' 
        best_action = torch.argmax(next_state_Q, axis=1)     # action'
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target):
        """
        Compute loss between td_estimate and td_target and perform backprop to
        update our model.
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        """
        TODO:
        """
        self.net.target.load_state_dict(self.net.online.state_dict())


    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), 
                 exploration_rate=self.exploration_rate
                ),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, checkpoint_fpath, model, optimizer):
        """
        Load model from checkpoint
        """
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])          
        return model, optimizer, checkpoint['epoch']

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

class MarioNet(nn.Module):
    """
    mini CNN structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

########################################################################################################
###                                     Train our Model                                              ###
########################################################################################################

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, load_chkpt="checkpoints/first_run_FULL_action_space/mario_net_6.chkpt")

logger = MetricLogger(save_dir)

episodes = 8000
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
