from bullet_safety_gym.envs import bases
import random
import numpy as np


class Apple(bases.Obstacle):
    def __init__(self, bc, init_xyz, fixed_base=True, movement='static', global_scaling=1.):
        super().__init__(
            bc=bc,
            name='Apple',
            file_name='obstacles/apple.urdf',
            fixed_base=fixed_base,
            global_scaling=global_scaling,
            init_xyz=init_xyz,
            init_color=(0.05, 0.95, 0, 1.0),
            movement=movement
        )

    def detect_collision(self, agent: bases.Agent):
        """ Apples do not own a collision shape"""
        return False


class Bomb(bases.Obstacle):
    def __init__(self, bc, init_xyz, fixed_base=True, movement='static', global_scaling=1.):
        super().__init__(
            bc=bc,
            name='Bomb',
            file_name='obstacles/bomb.urdf',
            fixed_base=fixed_base,
            global_scaling=global_scaling,
            init_xyz=init_xyz,
            init_color=(0.95, 0.05, 0, 1.0),
            movement=movement
        )
        self.visible = True

    def detect_collision(self, agent: bases.Agent):
        """ Bombs do not own a collision shape"""
        return False


class Box(bases.Obstacle):
    def __init__(self, bc, init_xyz, fixed_base, movement, global_scaling=1.):
        super().__init__(
            bc=bc,
            name='Box',
            file_name='obstacles/box.urdf',
            fixed_base=fixed_base,
            global_scaling=global_scaling,
            init_xyz=init_xyz,
            init_orientation=(0, 0, 2 * np.pi * random.uniform(0, 1)),
            movement=movement
        )

    def detect_collision(self, agent: bases.Agent):
        collision_list = self.bc.getContactPoints(
            bodyA=agent.body_id,
            bodyB=self.body_id)
        collision = True if collision_list != () else False

        return collision


class CircleZone(bases.Obstacle):
    def __init__(self, bc, global_scaling=1.):
        super().__init__(
            bc=bc,
            name='CircleZone',
            file_name='obstacles/circle_zone.urdf',
            fixed_base=True,
            init_xyz=(0, 0, 0),
            global_scaling=global_scaling,
            owns_collision_shape=False
        )
        self.radius = 7. * global_scaling

    def detect_collision(self, agent: bases.Agent) -> bool:
        # has no collision shape
        return False


class GoalZone(bases.Obstacle):
    def __init__(self, bc, global_scaling=1.):
        super().__init__(
            bc=bc,
            name='GoalZone',
            file_name='obstacles/goal_zone.urdf',
            fixed_base=True,
            init_xyz=(0, 0, 0),
            global_scaling=global_scaling,
            owns_collision_shape=False
        )
        self.radius = global_scaling * 1.3

    def detect_collision(self, agent: bases.Agent) -> bool:
        # has no collision shape
        return False


class LineBoundary(bases.Obstacle):
    def __init__(self, bc, init_xyz=[0, 0, 0], global_scaling=1.):
        super().__init__(
            bc=bc,
            name='LineBoundary',
            file_name='obstacles/line_boundary.urdf',
            fixed_base=True,
            init_xyz=init_xyz,
            global_scaling=global_scaling,
            owns_collision_shape=False
        )

    def detect_collision(self, agent: bases.Agent) -> bool:
        # has no collision shape
        return False


class Orb(bases.Obstacle):
    def __init__(self, bc, init_xyz, fixed_base, movement, global_scaling=1.):
        super().__init__(
            bc=bc,
            name='Orb',
            file_name='obstacles/orb.urdf',
            fixed_base=fixed_base,
            global_scaling=global_scaling,
            init_xyz=init_xyz,
            init_orientation=[0, 0, 2 * np.pi * random.uniform(0, 1)],
            movement=movement
        )

    def detect_collision(self, agent: bases.Agent):
        collision_list = self.bc.getContactPoints(
            bodyA=agent.body_id,
            bodyB=self.body_id)
        collision = True if collision_list != () else False

        return collision


class Pillar(bases.Obstacle):
    def __init__(self, bc, init_xyz, fixed_base, movement, global_scaling=1.):
        super().__init__(
            bc=bc,
            name='Pillar',
            file_name='obstacles/pillar.urdf',
            fixed_base=fixed_base,
            init_xyz=init_xyz,
            global_scaling=global_scaling,
            movement=movement,
            owns_collision_shape=False
        )
        self.radius = 0.45 * global_scaling

    def detect_collision(self, agent: bases.Agent) -> bool:
        xy_dist = np.linalg.norm(agent.get_position()[:2]
                                 - self.get_position()[:2])
        dist_until_contact = xy_dist - agent.collision_radius - self.radius
        col = True if dist_until_contact <= 0.0 else False
        return col

    def set_collision_filter(self, agent: bases.Agent):
        # set collision detection to zero with agent
        return None
        # disable_collision = 0
        # self.bc.setCollisionFilterPair(
        #     agent.body_id,
        #     self.body_id,
        #     -1, -1,  # root link indices
        #     disable_collision
        # )
        # for link in agent.link_list:
        #     self.bc.setCollisionFilterPair(
        #         agent.body_id,
        #         self.body_id,
        #         link.index,
        #         -1,  # root link indices
        #         disable_collision
        #     )


class Puck(bases.Obstacle):
    """ An object which is pushed around by agents. Used in Push tasks."""
    def __init__(self, bc):
        super().__init__(
            bc=bc,
            name='puck',
            file_name='obstacles/puck.urdf',
            fixed_base=False,
            init_xyz=[0, 0, 0.5],
            global_scaling=2.
        )
        self.radius = 1.2

    def detect_collision(self, agent: bases.Agent):
        collision_list = self.bc.getContactPoints(
            bodyA=agent.body_id,
            bodyB=self.body_id)
        collision = True if collision_list != () else False

        return collision


class Puddle(bases.Obstacle):
    def __init__(self, bc, init_xyz, fixed_base, movement, global_scaling=1.):
        super().__init__(
            bc=bc,
            name='puddle',
            file_name='obstacles/puddle.urdf',
            fixed_base=True,
            init_xyz=[0, 0, 0],
            global_scaling=global_scaling,
            owns_collision_shape=False
        )
        self.radius = 1.0 * global_scaling

    # def detect_collision(self, agent: bases.Agent) -> bool:
    #     xy_dist = np.linalg.norm(agent.get_position()[:2]
    #                              - self.get_position()[:2])
    #     col = True if xy_dist < self.radius else False
    #
    #     return col

    def detect_collision(self, agent: bases.Agent) -> bool:
        xy_dist = np.linalg.norm(agent.get_position()[:2]
                                 - self.get_position()[:2])
        dist_until_contact = xy_dist - agent.collision_radius - self.radius
        col = True if dist_until_contact <= 0.0 else False
        return col

    def set_collision_filter(self, agent: bases.Agent):
        # set collision detection to zero with agent
        disable_collision = 0
        self.bc.setCollisionFilterPair(
            agent.body_id,
            self.body_id,
            -1, -1,  # root link indices
            disable_collision
        )
        for link in agent.link_list:
            self.bc.setCollisionFilterPair(
                agent.body_id,
                self.body_id,
                link.index,
                -1,  # root link indices
                disable_collision
            )


def create_obstacles(
        bc,
        obstacles: dict,
        env_dim: float
) -> list:
    assert obstacles, f'obstacles={obstacles} is empty.'
    spawned_obstacles = []
    num_obstacles = sum([v['number'] for k, v in obstacles.items()])

    # setup circular alignment for initialized objects -> might be reset by
    # task specifics
    obs_init_pos = []
    for i in range(num_obstacles):
        obs_init_pos.append(
            [np.cos(i * 2 * np.pi / num_obstacles) * env_dim * 0.7,
             np.sin(i * 2 * np.pi / num_obstacles) * env_dim * 0.7,
             0.])

    for obstacle_name, properties in obstacles.items():
        # check if obstacle is defined as class in this file
        assert obstacle_name in globals()
        obstacle_class = globals()[obstacle_name]
        # copy original dict to avoid key errors when render mode is enabled
        props = properties.copy()
        num = props.pop('number')
        for i in range(num):  # now instantiate obstacles
            obstacle = obstacle_class(bc, obs_init_pos[i], **props)
            spawned_obstacles.append(obstacle)

    return spawned_obstacles


def create_one_obstacle(name, bc, number, init_xyz, fixed_base, movement):
    obstacles = []
    for i in range(number):
        if name == 'Box':
            obstacles.append(Box(bc, init_xyz[i], fixed_base, movement))
        elif name == 'Ball':
            pass
        elif name == 'Puddle':
            print('Spawn puddle.')
            obstacles.append(Puddle(bc, init_xyz[i], fixed_base, movement))
        else:
            raise NameError(f'Unknown obstacle: {name}')

    return obstacles
