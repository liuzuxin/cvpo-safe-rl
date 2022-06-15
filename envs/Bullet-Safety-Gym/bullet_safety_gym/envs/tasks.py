import numpy as np
from bullet_safety_gym.envs import env_utils
from bullet_safety_gym.envs import bases, sensors, agents
from bullet_safety_gym.envs.obstacles import GoalZone, LineBoundary, CircleZone, \
    Puck, Apple, Bomb


def angle2pos(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate angle towards a position, e.g. used to determine yaw of agent
    towards the goal zone.

    Returns:
        angle in radians
    """
    assert pos1.shape == pos2.shape
    diff = pos2 - pos1
    diff /= np.linalg.norm(diff)
    # x1: y-coordinates, x2: x-coordinates
    angle = np.arctan2(diff[1], diff[0])
    return angle


class ReachGoalTask(bases.Task):
    def __init__(
            self,
            bc,
            world,
            agent,
            obstacles,
            use_graphics
    ):
        super().__init__(
            bc=bc,
            world=world,
            agent=agent,
            obstacles=obstacles,
            continue_after_goal_achievement=True,
            use_graphics=use_graphics
        )
        # spawn goal zone
        self.goal = GoalZone(bc)
        self.world_name = world
        self.old_dist = self.get_xy_distance()   # used for shaped rewards
        # add sensors to agent depending on the type of obstacles
        self.equip_agent_with_sensors()
        # increase powers of some agents to improve random exploration
        self.agent.upgrade_power()

    def get_xy_distance(self) -> float:
        return np.linalg.norm(
            self.agent.get_position()[:2] - self.goal.get_position()[:2]
        )

    def calculate_cost(self):
        """Determines costs depending on agent and obstacles."""
        number_collisions = self.get_collisions()
        z = self.agent.get_position()[2]
        cs = dict(
            number_collisions=number_collisions,
            cost_collisions=number_collisions,
            # Drone should not leave valid operation space...
            cost_out_of_range=(1. if z > 2 else 0.)
        )
        # sum all costs in one total cost
        cs['cost'] = min(1, sum(v for k, v in cs.items() if k.startswith('cost_')))
        return cs

    def calculate_reward(self):
        """Implements the task's specific reward function, which depends on
        the agent and the surrounding obstacles.

        Note that potential-based reward shaping is applied.
        """
        cur_dist = self.get_xy_distance()
        reward = self.old_dist - cur_dist + 0.01 * self.agent.specific_reward()
        self.old_dist = cur_dist
        return reward

    def get_collisions(self) -> int:
        """ returns number of collisions with obstacles."""
        if len(self.obstacles) == 0:
            return 0
        collision_list = [ob.detect_collision(self.agent)
                          for ob in self.obstacles]
        return sum(collision_list)

    def get_observation(self):
        """Returns a task related observation: distance to goal zone."""
        delta_xyz = self.goal.get_position() - self.agent.get_position()
        # rescale into [-1, +1]
        return delta_xyz[:2] / (2 * self.world.env_dim)

    @property
    def goal_achieved(self):
        achieved = False
        agent_velocity = np.linalg.norm(self.agent.get_linear_velocity())
        if agent_velocity < 5 and self.get_xy_distance() < self.goal.radius:
            achieved = True

        return achieved

    def update_goal(self):
        goal_set = False
        while not goal_set:
            new_goal_pos = self.world.generate_random_xyz_position()

            min_distance = np.linalg.norm(
                    self.agent.get_position()[:2] - new_goal_pos[:2]
            )
            for obstacle in self.obstacles:

                min_distance = min(min_distance, np.linalg.norm(
                    obstacle.get_position()[:2] - new_goal_pos[:2]
                ))
            if min_distance > 1.5:
                self.goal.set_position(new_goal_pos)
                # self.bc.stepSimulation()
                goal_set = True
        self.old_dist = self.get_xy_distance()

    def setup_camera(self) -> None:
        """ Default setting for rendering."""
        self.world.camera.update(
            cam_base_pos=(0, -3, 0),
            cam_dist=1.2*self.world.env_dim,
            cam_yaw=0,
            cam_pitch=-60
        )

    def specific_reset(self) -> None:
        """ Set positions and orientations of agent and obstacles."""

        # set agent and goal positions
        self.agent.specific_reset()
        agent_pos = self.agent.init_xyz
        agent_pos[:2] = self.world.generate_random_xyz_position()[:2]
        goal_pos = agent_pos
        while np.linalg.norm(agent_pos[:2]-goal_pos[:2]) < self.world.body_min_distance:
            goal_pos = self.world.generate_random_xyz_position()
        # adjust the height of agent
        # agent_pos = np.concatenate((agent_pos[:2], [self.agent.init_xyz[2]]))
        self.agent.set_position(agent_pos)
        self.goal.set_position(goal_pos)
        self.old_dist = self.get_xy_distance()

        # set agent orientation towards goal
        yaw = angle2pos(self.agent.get_position(), self.goal.get_position())
        yaw = self.agent.init_rpy[2] + yaw
        # apply random orientation to agent.
        yaw += np.random.uniform(-np.pi, np.pi)
        quaternion = self.bc.getQuaternionFromEuler([0, 0, yaw])
        self.agent.set_orientation(quaternion)

        # reset obstacle positions
        if len(self.obstacles) > 0:
            obs_init_pos = env_utils.generate_obstacles_init_pos(
                num_obstacles=len(self.obstacles),
                agent_pos=self.agent.get_position(),
                goal_pos=self.goal.get_position(),
                world=self.world,
                min_allowed_distance=self.world.body_min_distance,
                agent_obstacle_distance=self.agent_obstacle_distance
            )
            for i in range(len(self.obstacles)):
                self.obstacles[i].set_position(obs_init_pos[i])


class PushTask(bases.Task):
    def __init__(
            self,
            bc,
            world,
            agent,
            obstacles,
            use_graphics,
            sensor='LIDARSensor'
    ):
        super().__init__(
            bc=bc,
            world=world,
            agent=agent,
            obstacles=obstacles,
            continue_after_goal_achievement=True,
            use_graphics=use_graphics
        )
        # spawn goal zone
        self.goal = GoalZone(bc=bc)
        self.puck = Puck(bc=bc)

        self.world_name = world
        self.old_dist = self.get_xy_distance()   # used for shaped rewards

        # add sensor to agent
        if len(self.obstacles) > 0:
            assert hasattr(sensors, sensor), f'Sensor={sensor} not implemented.'
            sensor = getattr(sensors, sensor)(
                bc=bc,
                agent=self.agent,
                obstacles=self.obstacles,
                number_rays=32,
                ray_length=self.world.env_dim/2,
                visualize=self.use_graphics
            )
            self.agent.set_sensor(sensor)

    @property
    def puck_to_goal_xy_distance(self) -> float:
        return np.linalg.norm(
            self.puck.get_position()[:2] - self.goal.get_position()[:2]
        )

    @property
    def agent_to_puck_xy_distance(self) -> float:
        return np.linalg.norm(
            self.puck.get_position()[:2] - self.agent.get_position()[:2]
        )

    def get_xy_distance(self) -> float:
        return np.linalg.norm(
            self.agent.get_position()[:2] - self.goal.get_position()[:2]
        )

    def calculate_cost(self):
        """determine costs depending on agent and obstacles. """
        number_collisions = self.get_collisions()
        cs = dict(
            number_collisions=number_collisions,
            cost_collisions=number_collisions
        )
        # sum all costs in one total cost
        cs['cost'] = sum(v for k, v in cs.items() if k.startswith('cost_'))

        return cs

    def calculate_reward(self):
        """ Apply potential-based shaping to the reward. """
        cur_dist = self.get_xy_distance()
        # reduce agent specific reward such that electricity costs are not
        # higher than moving towards the goal
        reward = self.old_dist - cur_dist + 0.01 * self.agent.specific_reward()
        self.old_dist = cur_dist

        return reward

    def get_collisions(self) -> int:
        """Returns the number of collisions with obstacles that occurred after
        the last simulation step call."""
        if len(self.obstacles) == 0:
            return 0
        collision_list = [ob.detect_collision(self.agent)
                          for ob in self.obstacles]

        return sum(collision_list)

    def get_observation(self):
        """Returns a task related observation: distance to puck,
        distance from puck to goal."""
        puck_to_goal = self.puck.get_position()[:2] - self.agent.get_position()[:2]
        delta_xyz = self.goal.get_position() - self.agent.get_position()
        # rescale into [-2, +2]
        return delta_xyz[:2] / self.world.env_dim

    @property
    def goal_achieved(self):
        achieved = False
        if self.puck_to_goal_xy_distance < self.goal.radius:
            achieved = True
        return achieved

    def update_goal(self):
        goal_set = False
        while not goal_set:
            new_goal_pos = self.world.generate_random_xyz_position()

            min_distance = np.linalg.norm(
                    self.agent.get_position()[:2] - new_goal_pos[:2]
            )
            for obstacle in self.obstacles:

                min_distance = min(min_distance, np.linalg.norm(
                    obstacle.get_position()[:2] - new_goal_pos[:2]
                ))
            if min_distance > 1.5:
                self.goal.set_position(new_goal_pos)
                # self.bc.stepSimulation()
                goal_set = True
        self.old_dist = self.get_xy_distance()

    def setup_camera(self) -> None:
        self.world.camera.update(
            cam_base_pos=(0, -3, 0),
            cam_dist=1.2*self.world.env_dim,
            cam_yaw=0,
            cam_pitch=-60
        )

    def specific_reset(self) -> None:
        """ Set positions and orientations of agent and obstacles."""

        # set agent and goal positions
        self.agent.specific_reset()
        agent_pos = self.world.generate_random_xyz_position()
        goal_pos = agent_pos
        while np.linalg.norm(agent_pos[:2]-goal_pos[:2]) < self.world.body_min_distance:
            goal_pos = self.world.generate_random_xyz_position()
        # adjust the height of agent
        agent_pos = np.concatenate((agent_pos[:2], [self.agent.init_xyz[2]]))
        self.agent.set_position(agent_pos)
        self.goal.set_position(goal_pos)
        self.old_dist = self.get_xy_distance()

        # apply random orientation to agent.
        random_yaw = np.random.uniform(-np.pi, np.pi)
        quaternion = self.bc.getQuaternionFromEuler([0, 0, random_yaw])
        self.agent.set_orientation(quaternion)

        # reset obstacle positions
        if len(self.obstacles) > 0:
            obs_init_pos = env_utils.generate_obstacles_init_pos(
                num_obstacles=len(self.obstacles),
                agent_pos=self.agent.get_position(),
                goal_pos=self.goal.get_position(),
                world=self.world,
                min_allowed_distance=self.world.body_min_distance,
                agent_obstacle_distance=self.agent_obstacle_distance
            )
            for i in range(len(self.obstacles)):
                self.obstacles[i].set_position(obs_init_pos[i])


class CircleTask(bases.Task):
    """ A task where agents have to run as fast as possible within a circular
        zone.
        Rewards are by default shaped.

    """
    def __init__(
            self,
            bc,
            world,
            agent,
            obstacles,
            use_graphics,
    ):
        super().__init__(
            bc=bc,
            world=world,
            agent=agent,
            obstacles=obstacles,
            continue_after_goal_achievement=False,  # no goal present
            use_graphics=use_graphics
        )
        self.old_velocity = 0.0  # used for shaped rewards

        # spawn circle zone
        self.circle = CircleZone(bc)
        # spawn safety boundaries
        self.x_lim = 6.
        self.bound_1 = LineBoundary(bc, init_xyz=[-self.x_lim, 0, 0])
        self.bound_2 = LineBoundary(bc, init_xyz=[self.x_lim, 0, 0])

    def calculate_cost(self, **kwargs):
        """ determine costs depending on agent and obstacles
        """
        costs = {}
        if np.abs(self.agent.get_position()[0]) > self.x_lim:
            costs['cost_outside_bounds'] = 1.
        # sum all costs in one total cost
        costs['cost'] = min(1, sum(v for k, v in costs.items() if k.startswith('cost_')))

        return costs

    def calculate_reward(self):
        """ Returns the reward of an agent running in a circle (clock-wise).
        """
        vel = self.agent.get_linear_velocity()[:2]
        pos = self.agent.get_position()[:2]
        dist = np.linalg.norm(pos)
        # position vector and optimal velocity are orthogonal to each other:
        # optimal reward when position vector and orthogonal velocity
        # point into same direction
        vel_orthogonal = np.array([-vel[1], vel[0]])
        r = 0.1*np.dot(pos, vel_orthogonal)/(1+np.abs(dist-self.circle.radius))
        r += 0.01 * self.agent.specific_reward()
        return r

    def get_collisions(self) -> int:
        """Returns the number of collisions with obstacles that occurred after
        the last simulation step call."""
        return 0  # no obstacles are spawned for Circle tasks

    def get_observation(self) -> np.ndarray:
        """Returns a task related observation: distance from circle boundary.
        Only agent's joint states are relevant, no sensors given."""
        pos = self.agent.get_position()[:2]
        dist = np.linalg.norm(pos)
        return np.array([dist-self.circle.radius, ]) / self.world.env_dim

    @property
    def goal_achieved(self) -> bool:
        # agent runs endlessly
        return False

    def setup_camera(self) -> None:
        # Note: disable planar reflection such that circle zone (with alpha < 1)
        # is nicely rendered
        self.bc.changeVisualShape(0, -1, rgbaColor=[1, 1, 1, 1])
        self.bc.configureDebugVisualizer(
            self.bc.COV_ENABLE_PLANAR_REFLECTION, 0)
        self.world.camera.update(
            cam_base_pos=(0, -3, 0),
            cam_dist=1.2 * self.world.env_dim,
            cam_yaw=0,
            cam_pitch=-60
        )

    def specific_reset(self) -> None:
        """ Reset agent position and set orientation towards desired run
            direction."""
        self.old_velocity = 0.
        self.agent.specific_reset()
        max_dist_to_origin = 4.
        min_dist_to_origin = 2

        agent_pos = np.random.uniform(-max_dist_to_origin, max_dist_to_origin, 2)
        positioning_done = False
        while not positioning_done:
            agent_pos = np.random.uniform(-max_dist_to_origin,
                                          max_dist_to_origin, 2)
            if min_dist_to_origin <= np.linalg.norm(agent_pos) <= max_dist_to_origin:
                positioning_done = True

        # adjust the height of agent
        agent_pos = np.concatenate((agent_pos[:2], [self.agent.init_xyz[2]]))
        self.agent.set_position(agent_pos)

        # set agent orientation in forward run direction
        y = angle2pos(self.agent.get_position(), np.zeros(3)) + np.pi / 2
        y += self.agent.init_rpy[2]
        quaternion = self.bc.getQuaternionFromEuler([0, 0, y])
        self.agent.set_orientation(quaternion)

    def update_goal(self):
        """ nothing to do for the run task."""
        pass


class RunTask(bases.Task):
    """ A task where agents have to run into the x-direction and are penalized
        for exceeding the velocity limit and crossing the safety boundaries.
    """
    def __init__(
            self,
            bc,
            world,
            agent,
            obstacles,
            use_graphics,
    ):
        super().__init__(
            bc=bc,
            world=world,
            agent=agent,
            obstacles=obstacles,
            continue_after_goal_achievement=False,  # no goal present
            use_graphics=use_graphics
        )
        self.old_potential = 0.0  # used for shaped rewards

        # spawn safety boundaries and rotate by 90°
        self.y_lim = 2.
        self.bound_1 = LineBoundary(bc, init_xyz=(11, -self.y_lim, 0))
        self.bound_2 = LineBoundary(bc, init_xyz=(11, self.y_lim, 0))
        quaternion = self.bc.getQuaternionFromEuler([0, 0., 0.5*np.pi])
        self.bound_1.set_orientation(quaternion)
        self.bound_2.set_orientation(quaternion)

    def calculate_cost(self):
        """ determine costs depending on agent and obstacles
        """
        costs = {}
        if np.abs(self.agent.get_position()[1]) > self.y_lim:
            costs['cost_outside_bounds'] = 1.
        if self.agent.velocity_violation:
            costs['cost_velocity_violation'] = 1.
        # sum all costs in one total cost
        costs['cost'] = min(1, sum(v for k, v in costs.items() if k.startswith('cost_')))
        return costs

    def calculate_task_potential(self) -> float:
        """ Return euclidean distance to fictitious target position.
        """
        cur_xy = self.agent.get_position()[:2]
        goal_xy = np.array([1e3, 0])
        return -np.linalg.norm(cur_xy - goal_xy) * 60

    def calculate_reward(self):
        progress = self.calculate_task_potential() - self.old_potential
        self.old_potential = self.calculate_task_potential()
        reward = progress + self.agent.specific_reward()
        return reward

    def get_collisions(self) -> int:
        """Returns the number of collisions with obstacles that occurred after
        the last simulation step call."""
        if len(self.obstacles) == 0:
            collision_list = []
        else:
            collision_list = [ob.detect_collision(self.agent)
                              for ob in self.obstacles]
        return sum(collision_list)

    def get_observation(self) -> np.ndarray:
        # update camera position
        agent_x = self.agent.get_position()[0]
        self.world.camera.update(cam_base_pos=(agent_x+3, 0, 2))
        # no task specific observations...
        return np.array([])

    @property
    def goal_achieved(self) -> bool:
        """agent cannot reach goal: run endlessly"""
        return False

    def setup_camera(self) -> None:
        """ Keep PyBullet's default camera setting."""
        self.world.camera.update(
            cam_base_pos=(3., 0, 2),
            cam_dist=2.5,
            cam_yaw=90,
            cam_pitch=-50
        )

    def specific_reset(self) -> None:
        """ Set positions and orientations of agent and obstacles."""
        self.agent.specific_reset()  # reset joints
        new_pos = self.agent.init_xyz
        new_pos[:2] = np.random.uniform(-0.01, 0.01, 2)
        self.agent.set_position(new_pos)
        self.old_potential = self.calculate_task_potential()

    def update_goal(self) -> None:
        # no goals are present in the run task...
        pass


class GatherTask(bases.Task):
    def __init__(
            self,
            bc,
            world,
            agent,
            obstacles,
            use_graphics
    ):
        super().__init__(
            bc=bc,
            world=world,
            agent=agent,
            obstacles=obstacles,
            continue_after_goal_achievement=False,  # terminate after goal
            use_graphics=use_graphics
        )
        self.agent_obstacle_distance = 0.5  # reduce agent obstacle spacing
        self.apple_reward = 10.
        self.bomb_cost = 1.
        self.dead_agent_reward = -10
        self.detection_distance = 1
        # reduce distance to objects, especially important for more complex
        # agents such as Ant to enable random exploration to collect sparse
        # rewards
        self.agent_obstacle_distance = 1.0  # default value in other tasks: 2.5
        self.obstacle_obstacle_distance = 2.0  # default in other tasks: 2.5
        # add sensors to agent depending on the type of obstacles
        self.equip_agent_with_sensors()

        # increase powers of some agents to increase range of exploration
        self.agent.upgrade_power()

    def calculate_cost(self):
        """determine costs depending on agent and obstacles. """
        info = {}
        c = self.get_collisions() * self.bomb_cost
        z = self.agent.get_position()[2]  # Limit range of Drone agent

        # sum all costs in one total cost
        info['cost_gathered_bombs'] = c
        info['cost_out_of_range'] = 1. if z > 2 else 0.
        # limit cost to be at most 1.0
        info['cost'] = min(1, sum(v for k, v in info.items()
                                  if k.startswith('cost_')))
        return info

    def calculate_reward(self):
        """ Apply potential-based shaping to the reward. """
        r = 0.
        for o in self.obstacles:
            if not isinstance(o, Apple):
                continue  # only consider apples
            xy_diff = o.get_position()[:2] - self.agent.get_position()[:2]
            dist = np.linalg.norm(xy_diff)
            if o.is_visible and dist < self.detection_distance:
                o.update_visuals(make_visible=False)
                r += self.apple_reward
        if not self.agent.alive:
            r = self.dead_agent_reward
        return r

    def equip_agent_with_sensors(self):
        apples = [ob for ob in self.obstacles if isinstance(ob, Apple)]
        bombs = [ob for ob in self.obstacles if isinstance(ob, Bomb)]

        # Pseudo LIDARs for apples and bombs
        for i, obs in enumerate([apples, bombs]):
            sensor = getattr(sensors, 'PseudoLIDARSensor')(
                bc=self.bc,
                agent=self.agent,
                obstacles=obs,
                number_rays=16,
                ray_length=self.world.env_dim,
                visualize=False
            )
            self.agent.add_sensor(sensor)

    def get_collisions(self) -> int:
        """Counts the number of collisions with bombs that occurred after
        the last simulation step call."""
        c = 0
        for o in self.obstacles:
            if not isinstance(o, Bomb):
                continue  # only consider apples
            xy_diff = o.get_position()[:2] - self.agent.get_position()[:2]
            dist = np.linalg.norm(xy_diff)
            # obstacles are only active when they are visible...
            if o.is_visible and dist < self.detection_distance:
                o.update_visuals(make_visible=False)
                c += 1
        return c

    def get_observation(self):
        # no task specific information in Gather
        return []

    @property
    def goal_achieved(self):
        # goal is achieved when all apples are collected
        achieved = False
        available_apples = [o for o in self.obstacles
                            if isinstance(o, Apple) and o.is_visible]
        if len(available_apples) == 0:
            achieved = True
        return achieved

    def update_goal(self):
        # nothing to update
        pass

    def setup_camera(self) -> None:
        self.world.camera.update(
            cam_base_pos=(0, -3, 0),
            cam_dist=1.2*self.world.env_dim,
            cam_yaw=0,
            cam_pitch=-60
        )

    def specific_reset(self) -> None:
        """ Set positions and orientations of agent and obstacles."""

        # first, set agent xy and adjust its height
        self.agent.specific_reset()
        agent_pos = np.zeros(3)
        agent_pos = np.concatenate((agent_pos[:2], [self.agent.init_xyz[2]]))
        self.agent.set_position(agent_pos)

        # second, reset obstacle positions
        if len(self.obstacles) > 0:
            obs_init_pos = env_utils.generate_obstacles_init_pos(
                num_obstacles=len(self.obstacles),
                agent_pos=self.agent.get_position(),
                goal_pos=np.array([]),  # no goal in gather task
                world=self.world,
                min_allowed_distance=self.obstacle_obstacle_distance,
                agent_obstacle_distance=self.agent_obstacle_distance
            )
            for i, ob in enumerate(self.obstacles):
                ob.set_position(obs_init_pos[i])

        # finally, make all collected objects visible again
        [ob.update_visuals(make_visible=True) for ob in self.obstacles]
