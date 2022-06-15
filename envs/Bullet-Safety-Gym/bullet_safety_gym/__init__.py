r"""Open-Safety Gym

    Copyright (c) 2021 Sven Gronauer: Technical University Munich (TUM)

    Distributed under the MIT License.
"""
import gym
from gym.envs.registration import register
# from bullet_safety_gym.envs.builder import EnvironmentBuilder


def get_bullet_safety_gym_env_list():
    env_list = []
    for env_spec in gym.envs.registry.all():
        if 'Safety' in env_spec.id:
            env_list.append(env_spec.id)
    return env_list


"""Register environments at OpenAI's Gym."""


# ==============================================================================
#       Reach Tasks
# ==============================================================================

# ===== Ball =====
register(
    id='SafetyBallReach-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=250,
    kwargs=dict(
        agent='Ball',
        task='ReachGoalTask',
        obstacles={'Box': {'number': 1, 'fixed_base': False,
                           'movement': 'circular'},
                   'Puddle': {'number': 8, 'fixed_base': True,
                              'movement': 'static'},
                   },
        world={'name': 'SmallRoom', 'factor': 1},
        # debug=True
    ),
)


# ===== Car =====
register(
    id='SafetyCarReach-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='RaceCar',
        task='ReachGoalTask',
        obstacles={'Box': {'number': 1, 'fixed_base': False,
                           'movement': 'circular'},
                   'Puddle': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom'},
        # debug=True
    ),
)

# ===== Ant =====
register(
    id='SafetyAntReach-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=1000,
    kwargs=dict(
        agent='Ant',
        task='ReachGoalTask',
        obstacles={'Box': {'number': 1, 'fixed_base': False,
                           'movement': 'circular'},
                   'Puddle': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom'},
    ),
)

# ===== Drone =====
register(
    id='SafetyDroneReach-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Drone',
        task='ReachGoalTask',
        obstacles={'Box': {'number': 1, 'fixed_base': False,
                           'movement': 'circular'},
                   'Pillar': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom'},
    ),
)


# ==============================================================================
#       Push Tasks
# ==============================================================================

# ===== Ball =====
register(
    id='SafetyBallPush-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=250,
    kwargs=dict(
        agent='Ball',
        task='PushTask',
        obstacles={},
        world={'name': 'SmallRoom', 'factor': 1},
        # debug=True
    ),
)

# ===== Ball =====
register(
    id='SafetyCarPush-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='RaceCar',
        task='PushTask',
        obstacles={},
        world={'name': 'SmallRoom', 'factor': 1},
        # debug=True
    ),
)

# ==============================================================================
#       Circle Run Tasks
# ==============================================================================


register(
    id='SafetyBallCircle-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=250,
    kwargs=dict(
        agent='Ball',
        task='CircleTask',
        obstacles={},
        world={'name': 'Octagon'},
        # debug=True
    )
)

register(
    id='SafetyCarCircle-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='RaceCar',
        task='CircleTask',
        obstacles={},
        world={'name': 'Octagon'},
        # debug=True
    )
)

register(
    id='SafetyAntCircle-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=1000,
    kwargs=dict(
        agent='Ant',
        task='CircleTask',
        obstacles={},
        world={'name': 'Octagon'},
    )
)

# ===== Drone =====
register(
    id='SafetyDroneCircle-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Drone',
        task='CircleTask',
        obstacles={},
        world={'name': 'Octagon'},
    )
)


# ==============================================================================
#       Run Tasks
# ==============================================================================

register(
    id='SafetyBallRun-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=250,
    kwargs=dict(
        agent='Ball',
        task='RunTask',
        obstacles={},
        world={'name': 'Plane200', 'factor': 1},
        # debug=True
    ),
)

register(
    id='SafetyCarRun-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='RaceCar',
        task='RunTask',
        obstacles={},
        world={'name': 'Plane200', 'factor': 1},
        # debug=True
    ),
)

register(
    id='SafetyAntRun-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=1000,
    kwargs=dict(
        agent='Ant',
        task='RunTask',
        obstacles={},
        world={'name': 'Plane200', 'factor': 1},
        # debug=True
    ),
)


# ===== Drone =====
register(
    id='SafetyDroneRun-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Drone',
        task='RunTask',
        obstacles={},
        world={'name': 'Plane200', 'factor': 1},
    ),
)


# ==============================================================================
#       Gather Tasks
# ==============================================================================

register(
    id='SafetyBallGather-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=250,
    kwargs=dict(
        agent='Ball',
        task='GatherTask',
        obstacles={'Apple': {'number': 8, 'fixed_base': True,
                           'movement': 'static'},
                   'Bomb': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom', 'factor': 1},
        # debug=True
    ),
)

register(
    id='SafetyCarGather-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='RaceCar',
        task='GatherTask',
        obstacles={'Apple': {'number': 8, 'fixed_base': True,
                           'movement': 'static'},
                   'Bomb': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom', 'factor': 1},
        # debug=True
    ),
)

register(
    id='SafetyAntGather-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=1000,
    kwargs=dict(
        agent='Ant',
        task='GatherTask',
        obstacles={'Apple': {'number': 8, 'fixed_base': True,
                           'movement': 'static'},
                   'Bomb': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom', 'factor': 1}
    ),
)


# ===== Drone =====
register(
    id='SafetyDroneGather-v0',
    entry_point='bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Drone',
        task='GatherTask',
        obstacles={'Apple': {'number': 8, 'fixed_base': True,
                             'movement': 'static'},
                   'Bomb': {'number': 8, 'fixed_base': True,
                            'movement': 'static'}
                   },
        world={'name': 'SmallRoom', 'factor': 1}
    ),
)
