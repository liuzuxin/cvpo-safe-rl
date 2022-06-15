r"""Open-Safety Gym

    Copyright (c) 2021 Sven Gronauer: Technical University Munich (TUM)

    Distributed under the MIT License.
"""
import pybullet as pb
import numpy as np
import gym
from pybullet_utils import bullet_client
from bullet_safety_gym.envs.obstacles import create_obstacles
from bullet_safety_gym.envs import bases, worlds, tasks, agents
import os
import pkgutil


def get_physics_parameters(task: str) -> tuple:
    """PyBullet physics simulation parameters depend on the task.

    Parameters
    ----------
    task: str
        Holding the name of a task class.

    Returns
    -------
    tuple
        Holding time_step, frame_skip, number_solver_iterations parameters.

    Raises
    ------
    ValueError
        If no class is found for task name.
    """
    assert hasattr(tasks, task), f'Task={task} not implemented.'
    if task in ['RunTask', 'GatherTask']:
        # the physics parameters are identically to PyBullet locomotion envs
        time_step = 1/40.
        frame_skip = 4
        number_solver_iterations = 5
    elif task in ['CircleTask']:
        time_step = 1/60.
        frame_skip = 6
        number_solver_iterations = 5
    elif task in ['ReachGoalTask', 'PushTask']:
        # avoid frame skip for collision detection: PyBullet returns only
        # collision information of last sub-step => frame_skip == 1
        time_step = 1/10.
        frame_skip = 1
        number_solver_iterations = 5
    else:
        raise ValueError(f'No physics parameters defined for task={task}')
    return time_step, frame_skip, number_solver_iterations



class EnvironmentBuilder(gym.Env):
    """Building class and starting point for all Bullet-Safety-Gym environments.

    To provide easy customization and extension, Bullet-Safety-Gym incorporates a
    modular structure. The EnvironmentBuilder class organizes the world creation
    in the physics simulator with individual obstacles, world, agents, etc.
    The layout information and world bodies is encoded at initialization into a
    dictionary that is passed to EnvironmentBuilder on which the world layout is
    based. This offers the opportunity to the user that it can easily be
    customized.

    This class follows the interface and inherits from the OpenAI Gym and,
    hence, can be used as expected with the methods reset(), step(), render()...

    Example
    -------
    Create a small-sized room and spawn a ball-shaped agent and one box as
    obstacle::

        from bullet_safety_gym.envs.builder import EnvironmentBuilder

        layout=dict(
            agent='Ball',
            task='ReachGoalTask',
            obstacles={'Box': {'number': 1, 'fixed_base': True,
                               'movement': 'static'},
                       },
            world={'name': 'SmallRoom', 'factor': 1},
            debug=True  # use this flag to navigate the ball with your keyboard
        )
        env = EnvironmentBuilder(**layout)
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            agent: str,
            task: str,
            obstacles: dict,
            world: dict,
            graphics=False,
            debug=False
    ):
        self.input_parameters = locals()  # save setting for later reset
        self.use_graphics = graphics
        self.debug = debug
        self.world_name = world['name']
        self.global_scaling = world.get('factor', 1.0)

        # Physics parameters depend on the task
        time_step, frame_skip, num_solver_iter = get_physics_parameters(task)
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.number_solver_iterations = num_solver_iter
        self.dt = self.time_step * self.frame_skip

        # first init PyBullet
        self.bc = self._setup_client_and_physics()
        self.bullet_client_id = self.bc._client
        self.stored_state_id = -1

        self._setup_simulation()

        # Define limits for observation space and action space
        obs_dim = self.get_observation().shape[0]
        act_dim = self.agent.act_dim
        o_lim = 1000 * np.ones((obs_dim, ), dtype=np.float32)
        a_lim = np.ones((act_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-o_lim, o_lim, dtype=np.float32)
        self.action_space = gym.spaces.Box(-a_lim, a_lim, dtype=np.float32)

        # stepping information
        self.iteration = 0

    def _setup_client_and_physics(
            self,
            graphics=False
    ) -> bullet_client.BulletClient:
        """Creates a PyBullet process instance.

        The parameters for the physics simulation are determined by the
        get_physics_parameters() function.

        Parameters
        ----------
        graphics: bool
            If True PyBullet shows graphical user interface with 3D OpenGL
            rendering.

        Returns
        -------
        bc: BulletClient
            The instance of the created PyBullet client process.
        """
        if graphics or self.use_graphics:
            bc = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        # optionally enable EGL for faster headless rendering
        try:
            if os.environ["PYBULLET_EGL"]:
                con_mode = bc.getConnectionInfo()['connectionMethod']
                if con_mode == bc.DIRECT:
                    egl = pkgutil.get_loader('eglRenderer')
                    if egl:
                        bc.loadPlugin(egl.get_filename(),
                                           "_eglRendererPlugin")
                        print('LOADED EGL...')
                    else:
                        bc.loadPlugin("eglRendererPlugin")
        except KeyError:
            # print('Note: could not load egl...')
            pass

        # add bullet_safety_gym/envs/data to the PyBullet data path
        bc.setAdditionalSearchPath(bases.get_data_path())
        # disable GUI debug visuals
        bc.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step * self.frame_skip,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=self.frame_skip)
        bc.setGravity(0, 0, -9.81)
        bc.setDefaultContactERP(0.9)
        return bc

    def _setup_simulation(self) -> None:
        """Create world layout, spawn agent and obstacles.

        Takes the passed parameters from the class instantiation: __init__().
        """
        world = self.input_parameters['world']
        agent = self.input_parameters['agent']
        task = self.input_parameters['task']
        obstacles = self.input_parameters['obstacles']
        # load ground plane and obstacles
        factor = world.get('factor', 1.0)
        self.world = self.get_world(world['name'], factor)
        # call agent class: spawns agent in world and collect joint information
        self.agent = self.get_agent(agent)
        # calculate the number of obstacles
        if obstacles:
            number_obstacles = [v['number'] for k, v in obstacles.items()]
            self.num_obstacles = sum(number_obstacles)
            self.obstacles = create_obstacles(self.bc, obstacles,
                                              env_dim=self.world.env_dim)
        else:
            self.num_obstacles = 0
            self.obstacles = []
        # define task
        self.task = self.get_task(task)
        # setup collision filter for some obstacles
        [ob.set_collision_filter(self.agent) for ob in self.obstacles]

    def close(self):
        if self.bullet_client_id >= 0:
            self.bc.disconnect()
        self.bullet_client_id = -1

    def get_agent(
            self,
            ag: str
    ) -> bases.Agent:
        """Instantiate a particular agent class.

        Parameters
        ----------
        ag: str
            Name of agent class to be instantiated.

        Raises
        ------
        AssertionError
            If no class is found for given agent name.
        """
        assert hasattr(agents, ag), f'Agent={ag} not found.'
        agent_cls = getattr(agents, ag)
        return agent_cls(self.bc, debug=self.debug)

    def get_observation(self) -> np.ndarray:
        """

        Returns
        -------
        array
        """
        agent_obs = self.agent.get_observation()
        task_obs = self.task.get_observation()
        obs = np.concatenate([agent_obs, task_obs])
        return obs

    def get_task(
            self,
            task: str
    ) -> bases.Task:
        """Instantiate a particular task class.

        Parameters
        ----------
        task: str
            Name of task class to be instantiated.

        Raises
        ------
        AssertionError
            If no class is found for task agent name.
        """
        assert hasattr(tasks, task), f'Task={task} not implemented.'
        task = getattr(tasks, task)

        return task(
            bc=self.bc,
            world=self.world,
            agent=self.agent,
            obstacles=self.obstacles,
            use_graphics=self.use_graphics
        )

    def get_world(
            self,
            name: str,
            factor: float
    ) -> bases.World:
        """Instantiate the world including ground plane and arena.

        Parameters
        ----------
        name: str
            Name of world class to be instantiated.
        factor: float
            Linear scaling factor of world.

        Raises
        ------
        AssertionError
            If no class is found for given world name.
        """
        assert hasattr(worlds, name), f'World={name} not found.'
        world = getattr(worlds, name)
        return world(self.bc, global_scaling=factor)

    def step(
            self,
            action: np.ndarray
    ) -> tuple:
        """Step the simulation's dynamics once forward.

        This method follows the interface of the OpenAI Gym.

        Parameters
        ----------
        action: array
            Holding the control commands for the agent.

        Returns
        -------
        observation (object)
            Agent's observation of the current environment
        reward (float)
            Amount of reward returned after previous action
        done (bool)
            Whether the episode has ended, handled by the time wrapper
        info (dict)
            contains auxiliary diagnostic information such as the cost signal
        """
        action = np.squeeze(action)
        self.iteration += 1
        self.agent.apply_action(action)
        for obstacle in self.obstacles:
            obstacle.apply_movement()

        # loop and detect collisions
        # Use manual sub-stepping since PyBullet checks only the last sub-step
        self.bc.stepSimulation()
        # collecting information after Sim stepping is crucial to detect
        # collisions or determine reward costs (e.g. electricity costs)
        self.agent.collect_information_after_step()

        r = self.task.calculate_reward()
        info = self.task.calculate_cost()
        # update agent visuals when costs are received
        if info.get('cost', 0) > 0:
            self.agent.violates_constraints(True)
        else:
            self.agent.violates_constraints(False)
        done = not self.agent.alive
        if self.task.goal_achieved:
            if self.task.continue_after_goal_achievement:
                r += 5.0  # add sparse reward
                self.task.update_goal()
            else:
                done = True
        next_obs = self.get_observation()
        return next_obs, r, done, info

    def render(
            self,
            mode='human'
    ) -> np.ndarray:
        """Show PyBullet GUI visualization.

        Render function triggers the PyBullet GUI visualization.
        Camera settings are managed by Task class.

        Note: For successful rendering call env.render() before env.reset()

        Parameters
        ----------
        mode: str

        Returns
        -------
        array
            holding RBG image of environment if mode == 'rgb_array'
        """
        if mode == 'human':
            # close direct connection to physics server and
            # create new instance of physics with GUI visuals
            if not self.use_graphics:
                self.bc.disconnect()
                self.use_graphics = True
                self.bc = self._setup_client_and_physics(graphics=True)
                self._setup_simulation()
        if mode != "rgb_array":
            return np.array([])
        else:
            view_matrix = self.bc.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.world.camera.cam_base_pos,
                distance=self.world.camera.cam_dist,
                yaw=self.world.camera.cam_yaw,
                pitch=self.world.camera.cam_pitch,
                roll=0,
                upAxisIndex=2
            )
            w = float(self.world.camera.render_width)
            h = self.world.camera.render_height
            proj_matrix = self.bc.computeProjectionMatrixFOV(
                fov=60,
                aspect=w / h,
                nearVal=0.1,
                farVal=100.0
            )
            (_, _, px, _, _) = self.bc.getCameraImage(
                width=self.world.camera.render_width,
                height=self.world.camera.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pb.ER_BULLET_HARDWARE_OPENGL)

            new_shape = (self.world.camera.render_height,
                         self.world.camera.render_width,
                         -1)
            rgb_array = np.reshape(np.array(px), new_shape)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        This function is called after agent encountered terminal state.

        Returns
        -------
        array
            holding the observation of the initial state
        """
        # disable rendering before resetting
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        if self.stored_state_id >= 0:
            self.bc.restoreState(self.stored_state_id)
        self.iteration = 0
        self.task.specific_reset()
        # Restoring a saved state circumvents the necessity to load all bodies
        # again..
        if self.stored_state_id < 0:
            self.stored_state_id = self.bc.saveState()
        # now enable rendering again
        self.bc.stepSimulation()
        if self.use_graphics:
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)
        return self.get_observation()
