**Status:** Development (Major updates may be expected)

# Bullet-Safety-Gym

"Bullet-Safety-Gym" is a free and open-source framework to benchmark and assess 
safety specifications in Reinforcement Learning (RL) problems.


## Yet another Gym?

Benchmarks are inevitable in order to assess the scientific progress. In recent 
years, a plethora of benchmarks frameworks has emerged but we felt that the 
following points have not been sufficiently answered yet:

- **Increasing awareness of safety in AI**: The number of publications regarding
safety is rising. While deep RL is making big strides out of its infancy, a 
majority of environments considers only reward maximization and have no explicit 
connection to safety concepts.

+ **Unified repository**: 
Although diverse safety environments have been used in recent works,
repositories that unify and standardize them into one framework are scarce, e.g.
Ray et al. 2019)

+ **Strong Reproducibility**: Many state-of-the-art RL papers do not rely on 
open-source software, which may slow down the pace of research compared to when 
experiments were freely accessible to everyone. For a recent and hot discussion
about reproducibility, see the following ICLR review posted
[on Reddit](
https://www.reddit.com/r/MachineLearning/comments/jssmia/d_an_iclr_submission_is_given_a_clear_rejection/
)

## Implemented Agents

Bullet-Safety-Gym is shipped with the following four agents:

+ **Ball**: A spherical shaped agent which can freely move on the xy-plane. 
+ **Car**: A four-wheeled agent based on MIT's Racecar.
+ **Drone**: An air vehicle based on the AscTec Hummingbird quadrotor. 
+ **Ant**: A four-legged animal with a spherical torso.


Ball | Car | Drone | Ant
--- | ---| ---| ---
![Ball](./docs/figures/agent_ball.png) |![Car Agent](./docs/figures/agent_car.png)|![Drone Agent](./docs/figures/agent_drone.png)|![Ant Agent](./docs/figures/agent_ant.png)



## Tasks

+ **Circle**: 
Agents are expected to move on a circle in clock-wise direction (as proposed 
by Achiam et al. (2017)). The reward is dense and increases by the agent's 
velocity and by the proximity towards the boundary of the circle. Costs are 
received when agent leaves the safety zone defined by the two yellow boundaries.

+ **Gather**
Agents are expected to navigate and collect as many green apples as possible
 while avoiding red bombs (Duan et al. 2016). In contrast to the other tasks, 
 agents in the gather tasks receive only sparse rewards when reaching apples. 
 Costs are also sparse and received when touching bombs (Achiam et al. 2017).

+ **Reach**: Agents are supposed to move towards a goal (Ray et al. 2019). As 
soon the agents enters the goal zone, the goal is re-spawned such that the agent
has to reach a series of goals. Obstacles are placed to hinder the agent from 
trivial solutions. We implemented obstacles with a physical body, into which 
agents can collide and receive costs, and ones without collision shape that 
produce costs for traversing. Rewards are dense and increase for moving closer 
to the goal and a sparse component is obtained when entering the goal zone. 
    


+ **Run**: Agents are rewarded for running through an avenue between two 
safety boundaries (Chow et al. 2019). The boundaries are non-physical 
bodies which can be penetrated without collision but provide costs. 
Additional costs are received when exceeding an agent-specific velocity 
threshold.



Circle | Gather | Reach | Run
--- | ---| ---| ---
![Circle](./docs/figures/task_circle.png) |![Gather](./docs/figures/task_gather.png)|![Reach](./docs/figures/task_reach.png)|![Run](./docs/figures/task_run.png)



# Installation

Here are the (few) steps to follow to get our repository ready to run.

Clone the repository and install the Bullet-Safety-Gym package via pip. Use the 
following three lines:

```
git clone https://github.com/SvenGronauer/Bullet-Safety-Gym.git

cd Bullet-Safety-Gym

pip install -e .
```

## Supported Systems

We currently support Linux and OS X running Python 3.5 or greater.
Windows should also work (but has not been tested yet).

Note: This package has been tested on Mac OS Mojave and Ubuntu (18.04 LTS, 
20.04 LTS), and is probably fine for most recent Mac and Linux operating 
systems. 

## Dependencies 

Bullet-Safety-Gym heavily depends on two packages:

+ [Gym](https://github.com/openai/gym)
+ [PyBullet](https://github.com/bulletphysics/bullet3)

# Getting Started

After the successful installation of the repository, the Bullet-Safety-Gym 
environments can be simply instantiated via `gym.make`. See: 

```
>>> import gym
>>> import bullet_safety_gym
>>> env = gym.make('SafetyCarGather-v0')
```

The functional interface follows the API of the OpenAI Gym (Brockman et al., 
2016) that consists of the three following important functions:

```
>>> observation = env.reset()
>>> random_action = env.action_space.sample()  # usually the action is determined by a policy
>>> next_observation, reward, done, info = env.step(random_action)
```

Besides the reward signal, our environments provide an additional cost signal, 
which is contained in the `info` dictionary:
```
>>> info
{'cost': 1.0}
```

A minimal code for visualizing a uniformly random policy in a GUI, can be seen 
in:

```
import gym
import bullet_safety_gym

env = gym.make('SafetyAntCircle-v0')

while True:
    done = False
    env.render()  # make GUI of PyBullet appear
    x = env.reset()
    while not done:
        random_action = env.action_space.sample()
        x, reward, done, info = env.step(random_action)
```
Note that only calling the render function before the reset function triggers 
visuals.


# List of environments

The environments are named in the following scheme:
```Safety{#agent}{#task}-v0``` where the agent can be any of 
```{Ball, Car, Drone, Ant}``` and the task 
can be of ```{Circle, Gather, Reach, Run,}```.

There exists also a function which returns all available environments of the 
Bullet-Safety-Gym:
```
from bullet_safety_gym import get_bullet_safety_gym_env_list

env_list = get_bullet_safety_gym_env_list()
```

## Reach Environments

+ SafetyBallReach-v0
+ SafetyCarReach-v0
+ SafetyDroneReach-v0
+ SafetyAntReach-v0

## Circle Run Environments
+ SafetyBallCircle-v0
+ SafetyCarCircle-v0
+ SafetyDroneCircle-v0
+ SafetyAntCircle-v0

## Run Environments
+ SafetyBallRun-v0
+ SafetyCarRun-v0
+ SafetyDroneRun-v0
+ SafetyAntRun-v0

## Gather Environments
+ SafetyBallGather-v0
+ SafetyCarGather-v0
+ SafetyDroneGather-v0
+ SafetyAntGather-v0


# References

+ Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Con- strained policy optimization. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 22–31, International Convention Centre, Sydney, Australia, 06– 11 Aug 2017. PMLR.

+ G. Brockman, Vicki Cheung, Ludwig Pettersson, J. Schneider, John Schulman, Jie Tang, and W. Zaremba. Openai gym. ArXiv, abs/1606.01540, 2016.

+ Yinlam Chow, Ofir Nachum, Aleksandra Faust, Mohammad Ghavamzadeh, and Edgar A. Due ́n ̃ez-Guzma ́n. Lyapunov-based safe policy optimization for continuous control. CoRR, abs/1901.10031, 2019.

+ Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learn- ing, volume 48 of Proceedings of Machine Learning Research, pages 1329–1338, New York, New York, USA, 20–22 Jun 2016. PMLR.

+ Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking Safe Exploration in Deep Reinforcement Learning. 2019.
