import numpy as np
from bullet_safety_gym.envs import bases


def check_min_distance(a_pos_list, b_pos, min_allowed_distance):
    min_distance = np.min(
        np.linalg.norm(np.vstack(a_pos_list) - b_pos, axis=1))

    return min_distance > min_allowed_distance


def generate_obstacles_init_pos(
        num_obstacles: int,
        agent_pos: tuple,
        world: bases.World,
        goal_pos: np.ndarray = np.array([]),
        min_allowed_distance: float = 2.5,
        agent_obstacle_distance: float = 2.5,
):
    assert num_obstacles > 0
    i = 0
    xyz_list = []
    if goal_pos.size > 0:
        xyz_list.append(goal_pos)

    while i < num_obstacles:
        xyz = world.generate_random_xyz_position()
        if not satisfies_distance_criterion(agent_pos[:2], xyz[:2], agent_obstacle_distance):
            continue
        if len(xyz_list) == 0 or check_min_distance(xyz_list, xyz, min_allowed_distance):
            xyz_list.append(xyz)
            i += 1
        if i > 1000:
            raise RuntimeError('Spawning of objects took to much trials.')

    # if goal_pos.size > 0:  # exclude goal from obstacle list
    #     obstacle_xyz_list = xyz_list[1:]
    # else:
    #     obstacle_xyz_list = xyz_list[:]

    obstacle_xyz_list = xyz_list[1:] if goal_pos.size > 0 else xyz_list
    return obstacle_xyz_list


def satisfies_distance_criterion(pos_a, pos_b, minimum_distance):
    dist = np.linalg.norm(pos_a - pos_b)
    satisfaction = True if dist >= minimum_distance else False
    return satisfaction
