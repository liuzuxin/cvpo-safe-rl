import pybullet as pb
import numpy as np
from bullet_safety_gym.envs import bases


class Ball(bases.Agent):
    """ A spherical agent that moves on the (x,y)-plane.

    The ball is moved via external forces that are applied in world coordinates.
    Observations are in R^7 and actions are in R^2.
    """
    def __init__(
            self,
            bc,
            init_xyz=(0, 0, .5),  # ball diameter is 0.5
            debug=False
    ):
        super().__init__(
            bc,
            'base_link',
            'robots/ball/ball.urdf',
            act_dim=2,
            obs_dim=7,
            init_xyz=init_xyz,
            fixed_base=False,
            global_scaling=1,
            collision_radius=0.5,  # ball has 0.5 diameter
            self_collision=False,
            velocity_constraint=2.5,
            max_force=3.5,
            max_velocity=0,  # irrelevant parameter (external force controlled)
            debug=debug
        )
        self.radius = 0.5
        self.size_violation_shape = self.global_scaling * 1.25 * self.radius
        self.last_taken_action = np.zeros(self.act_dim)

    def add_sensor(self, sensor):
        """ A sensor is added to the agent by a task.
            e.g. the goal reach tasks adds a sensor to detect obstacles.
        """
        # Avoid rotation of sensor with ball agent
        sensor.rotate_with_agent = False
        super().add_sensor(sensor)

    @property
    def alive(self) -> bool:
        """Returns "False" if the agent died, "True" otherwise.
        Ball agent has no termination criterion."""
        return True

    def apply_action(self, action):
        # check validity of action and clip into range [-1, +1]
        self.last_taken_action = super().apply_action(action)
        x, y = self.last_taken_action
        sphere_pos = self.get_position()
        # over-write actions with keyboard inputs
        if self.debug:
            keys = self.bc.getKeyboardEvents()
            x = 0
            y = 0
            for k, v in keys.items():
                if k == pb.B3G_UP_ARROW and (v & pb.KEY_IS_DOWN):
                    y += 1
                if k == pb.B3G_LEFT_ARROW and (v & pb.KEY_IS_DOWN):
                    x += -1
                if k == pb.B3G_DOWN_ARROW and (v & pb.KEY_IS_DOWN):
                    y += -1
                if k == pb.B3G_RIGHT_ARROW and (v & pb.KEY_IS_DOWN):
                    x += 1

        self.apply_external_force(
            force=np.array([x, y, 0.]) * self.max_force,
            link_id=-1,
            position=sphere_pos,
            frame=pb.WORLD_FRAME
        )

    def get_linear_velocity(self) -> np.ndarray:
        """ over-write Agent class since Ball owns only one body."""
        return self.get_state()[3:6]

    def agent_specific_observation(self) -> np.ndarray:
        """ State of ball is of shape (7,) """
        xyz = 0.1 * self.get_position()
        xyz_dot = 0.2 * self.get_linear_velocity()
        rpy_dot = 0.1 * self.get_angular_velocity()
        obs = np.concatenate((xyz[:2], xyz_dot[:2], rpy_dot))
        return obs

    def get_orientation(self) -> np.ndarray:
        """ over-write Agent class since Ball owns only one body."""
        return self.get_state()[6:9]

    def get_position(self) -> np.ndarray:
        """ over-write Agent class since Ball owns only one body."""
        return self.get_state()[:3]

    def get_quaternion(self):
        xyz, abcd = self.bc.getBasePositionAndOrientation(self.body_id)
        return abcd

    def specific_reset(self):
        """ Reset only agent specifics such as motor joints. Do not set position
            or orientation since this is handled by task.specific_reset()."""
        self.set_position(self.init_xyz)

    def specific_reward(self) -> float:
        """ Some agents exhibit additional rewards besides the task objective,
            e.g. electricity costs, speed costs, etc. """
        return -0.5 * np.linalg.norm(self.last_taken_action)


class RaceCar(bases.Agent):
    """MIT Race Car.

    Designed and developed by the Massachusetts Institute of Technology as
    open-source platform. See: https://mit-racecar.github.io/

    A four-wheeled agent implemented with a simplified control scheme based on
    the target wheel velocity for all four wheels and the target steering angle.
    Observations are in R^7 and actions are in R^2.
    """
    def __init__(
            self, 
            bc, 
            init_xyz=(0, 0, 0.2), 
            debug=False):
        super().__init__(
            bc,
            'base_link',
            'robots/racecar/racecar_differential.urdf',
            act_dim=2,
            obs_dim=7,
            init_xyz=init_xyz,
            init_orientation=(0., 0., np.pi),
            fixed_base=False,
            global_scaling=3,
            self_collision=False,
            collision_radius=0.35,
            velocity_constraint=1.5,
            max_force=20,  # PyBullet default value: 20
            max_velocity=0,  # irrelevant parameter (force-controlled agent)
            violation_shape_factor=0.35,
            debug=debug
        )
        self.steering_links = [0, 2]
        self.motorized_wheels = [8, 15]
        self.speed_multiplier = 40.  # PyBullet default value: 20
        self.steering_multiplier = 0.5  # PyBullet default value: 0.5
        self._set_car_constraints()

    @property
    def alive(self) -> bool:
        """Returns "False" if the agent died, "True" otherwise.
        Car agent has no termination criterion."""
        return True

    def _set_car_constraints(self):
        """This code is copied from the PyBullet repository:
        https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/racecar.py#L16
        """
        car = self.body_id
        for wheel in range(self.bc.getNumJoints(car)):
            self.bc.setJointMotorControl2(car,
                                          wheel,
                                          self.bc.VELOCITY_CONTROL,
                                          targetVelocity=0,
                                          force=0)
            self.bc.getJointInfo(car, wheel)

        c = self.bc.createConstraint(car,
                                     9,
                                     car,
                                     11,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self.bc.createConstraint(car,
                                     10,
                                     car,
                                     13,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self.bc.createConstraint(car,
                                     9,
                                     car,
                                     13,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self.bc.createConstraint(car,
                                     16,
                                     car,
                                     18,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self.bc.createConstraint(car,
                                     16,
                                     car,
                                     19,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self.bc.createConstraint(car,
                                     17,
                                     car,
                                     19,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self.bc.createConstraint(car,
                                     1,
                                     car,
                                     18,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = self.bc.createConstraint(car,
                                     3,
                                     car,
                                     19,
                                     jointType=self.bc.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self.bc.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    def agent_specific_observation(self):
        """ State of cart is of shape (6,) """
        xy = 0.1 * self.get_position()[:2]  # re-scale
        xy_dot = self.get_linear_velocity()[:2]
        yaw = self.get_orientation()[2]
        yaw_dot = 0.1 * self.get_angular_velocity()[2]
        yaw_info = [np.sin(yaw), np.cos(yaw), yaw_dot]
        return np.concatenate((xy, xy_dot, yaw_info))

    def apply_action(self, motorCommands):
        targetVelocity = motorCommands[0] * self.speed_multiplier
        steeringAngle = motorCommands[1] * self.steering_multiplier

        if self.debug:  # over-write passed motor commands
            action = np.zeros_like(motorCommands)
            keys = self.bc.getKeyboardEvents()
            for k, v in keys.items():
                if k == self.bc.B3G_RIGHT_ARROW and (v & self.bc.KEY_IS_DOWN):
                    action += np.array([0, 1])
                if k == self.bc.B3G_LEFT_ARROW and (v & self.bc.KEY_IS_DOWN):
                    action += np.array([0, -1])
                if k == self.bc.B3G_UP_ARROW and (v & self.bc.KEY_IS_DOWN):
                    action += np.array([-1, 0])
                if k == self.bc.B3G_DOWN_ARROW and (v & self.bc.KEY_IS_DOWN):
                    action += np.array([1, 0])
            targetVelocity = action[0] * self.speed_multiplier
            steeringAngle = action[1] * self.steering_multiplier

        for motor in self.motorized_wheels:
            self.bc.setJointMotorControl2(self.body_id,
                                        motor,
                                        self.bc.VELOCITY_CONTROL,
                                        targetVelocity=targetVelocity,
                                        force=self.max_force)
        for steer in self.steering_links:
            self.bc.setJointMotorControl2(self.body_id,
                                        steer,
                                        self.bc.POSITION_CONTROL,
                                        targetPosition=steeringAngle)

    def add_sensor(self, sensor):
        """ Racing Car needs an adjustment of sensor heights."""
        super().add_sensor(sensor)
        self.sensors[-1].set_offset((0.2, 0, 0.3))  # default: (0.2, 0, 0.3)

    def specific_reset(self):
        """ reset motor joints."""
        for j in self.motor_list:
            j.set_state(np.random.uniform(low=-0.1, high=0.1), 0)

    def specific_reward(self) -> float:
        """ Some agents exhibit additional rewards besides the task objective,
            e.g. electricity costs, speed costs, etc. """
        return 0.0

    def upgrade_power(self):
        """Some tasks require higher agent powers to encourage exploratory
        actions."""
        self.max_force = 40.
        self.speed_multiplier = 80


class MJCFAgent(bases.Agent):
    def __init__(
        self,
        bc,
        name,
        file_name,
        foot_list,
        max_force,
        **kwargs
    ):
        super().__init__(
            bc,
            name,
            file_name,
            fixed_base=False,
            global_scaling=1.0,
            self_collision=True,
            max_velocity=5,
            max_force=max_force,
            **kwargs
        )
        self.foot_list = foot_list
        self.feet_contact = np.array([0.0 for _ in self.foot_list],
                                     dtype=np.float32)
        self.feet = [self.link_dict[f] for f in self.foot_list]
        
        # track movement costs
        self.feet_collision_reward = 0.0
        self.joints_at_limit_reward = 0.0
        self.action_reward = 0.0
        self.last_taken_action = None

        # factors
        self.electricity_factor = -2.0  # cost for using motors
        self.stall_torque_factor = -0.1  # cost for running electric current
        self.joints_at_limit_factor = -0.1  # discourage stuck joints
        
        # increase power of all MJCF agents' motors
        for m in self.motor_list:
            m.power_coefficient = 100.
            # print(f'Joint: {m.name} lower: {m.lowerLimit} lower: {m.upperLimit}')

    def apply_action(self, action):
        # check validity of action and clip into range [-1, +1]
        valid_action = super().apply_action(action)
        self.last_taken_action = action
        for n, j in enumerate(self.motor_list):
            tq = float(self.max_force * j.power_coefficient * valid_action[n])
            j.set_torque(tq)

    def collect_information_after_step(self):
        """ some agents need to update internals after pybullet.stepSimulation()
        call, e.g. update collision information or feet contact infos."""

        # ===== Update feet information: contacts and costs
        ground_ids = {(0, -1)}  # (body_id, link_index) tuple
        for i, f in enumerate(self.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # check contact between ground and foot by comparing sets
            self.feet_contact[i] = 1.0 if ground_ids & contact_ids else 0.0
        # self.feet_collision_reward = sum(self.feet_contact)
        self.feet_collision_reward = 0.0

        # ===== Update joint costs
        j = np.array(
            [j.get_relative_position() for j in self.motor_list],
            dtype=np.float32).flatten()
        non_zeros = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        self.joints_at_limit_reward = self.joints_at_limit_factor * non_zeros

        # ===== Action costs: joint electricity, torque ...
        a, j_speeds = self.last_taken_action, j[1::2]
        e_cost = self.electricity_factor * float(np.abs(a * j_speeds).mean())
        stall_torque = self.stall_torque_factor * float(np.square(a).mean())
        self.action_reward = e_cost + stall_torque

    def agent_specific_observation(self) -> np.ndarray:
        js = np.array(
            [j.get_relative_position() for j in self.motor_list],
            dtype=np.float32).flatten()
        obs = np.concatenate([
            0.1 * self.get_position(),
            0.3 * self.get_linear_velocity(),
            # Use quaternion instead of sin/cos of rpy...
            self.get_quaternion(),
            # np.cos(self.get_orientation()),  # rpy
            # np.sin(self.get_orientation()),  # rpy
            0.15 * self.get_angular_velocity(),
            0.5 * js,
            self.feet_contact
        ])
        # no clipping performed in obs!
        return obs

    def specific_reset(self) -> None:
        for j in self.motor_list:
            j.reset_position_and_disable_motor(
                np.random.uniform(low=-0.1, high=0.1), 0)
            j.power_coefficient = 100.
        self.feet_contact = np.array([0.0 for _ in self.foot_list],
                                     dtype=np.float32)

    def specific_reward(self) -> float:
        """ Some agents exhibit additional rewards besides the task objective,
            e.g. electricity costs, speed costs, etc. """
        alive_reward = 1.0 if self.alive else -1
        r = sum([self.action_reward,
                 self.feet_collision_reward,
                 self.joints_at_limit_reward,
                 alive_reward
                 ])
        return r


class Ant(MJCFAgent):
    foot_list = ['front_left_foot', 'front_right_foot',
                 'left_back_foot', 'right_back_foot']

    def __init__(
            self,
            bc,
            **kwargs
    ):
        super().__init__(
            bc=bc,
            name='torso',
            file_name='robots/mujoco/ant.xml',
            obs_dim=33,
            act_dim=8,  # number of actuators
            collision_radius=0.25,
            init_xyz=(0, 0, 0.75),
            max_force=2.5,
            velocity_constraint=1.5,
            foot_list=Ant.foot_list,
            **kwargs
        )
        self.radius = 0.25
        self.size_violation_shape = self.global_scaling * 2.5 * self.radius

    def add_sensor(self, sensor):
        """ A sensor is added to the agent by a task.
            e.g. the goal reach tasks adds a sensor to detect obstacles.
        """
        # Avoid rotation of sensor with Ant agent
        sensor.rotate_with_agent = False
        super().add_sensor(sensor)

    @property
    def alive(self):
        """0.25 is central sphere radius, die if it touches the ground
        """
        return True if self.get_position()[2] > 0.26 else False

    def specific_reset(self) -> None:
        """ Improved spawning behavior of Ant agent.

            Ankle joints are set to 90° position which enables uniform random
            policies better exploration and occasionally generates forward
            movements.
        """
        for i, j in enumerate(self.motor_list):
            noise = np.random.uniform(low=-0.1, high=0.1)
            pos = noise if i % 2 == 0 else np.pi/2 + noise
            if i == 3 or i == 5:
                pos *= -1
            j.reset_position_and_disable_motor(pos, 0)
            j.power_coefficient = 100.
        self.feet_contact = np.array([0.0 for _ in self.foot_list],
                                     dtype=np.float32)

    def upgrade_power(self):
        """Some tasks require higher agent powers to encourage exploratory
        actions."""
        self.max_force = 10.


class Drone(bases.Agent):
    """Drone agent based on the AscTec Hummingbird.

    URDF file, parameters and calculations taken from:
        - Brian Delhaisse (PyRoboLearn Repository)
    Meshes from:
        - https://github.com/ethz-asl/rotors_simulator/tree/master/rotors_description/meshes
    """
    def __init__(self,
                 bc,
                 init_xyz=(0, 0, .17),  # initially touches ground plane
                 debug=False):
        # from URDF file..
        self.radius = 0.14
        self.diameter = 2. * self.radius
        self.area = np.pi * self.radius**2

        # motors 1 and 3 rotate in CCW direction, and motors 2 and 4 CW
        # Note that CCW = +1, CW = -1
        self.propeller_directions = np.array([+1, -1, +1, -1])
        self.propeller_pitch = 4.7 * 0.0254
        # some constants
        self.k1 = 1./3.29546
        self.k2 = 1.5
        self.gravity = 9.81
        self.air_density = 1.225
        self.mass = 1.420

        super().__init__(
            bc,
            'base_link',
            'robots/quadcopter/quadcopter.urdf',
            act_dim=4,
            obs_dim=17,
            init_xyz=init_xyz,
            init_orientation=(0., 0., 0.),
            fixed_base=False,
            collision_radius=0.5,
            # make drone twice as large. Not that this does not change its mass!
            global_scaling=2.,
            self_collision=False,
            velocity_constraint=1.5,
            max_force=23.444,  # Maximum force used for RPM control of motors
            max_velocity=770,  # rad/sec
            violation_shape_factor=0.35,
            debug=debug
        )
        # collect link info
        self.rotor_ground_contact = np.zeros_like(self.link_list)
        self.ground_collision_penalty = 0.0

        # joint velocity (rad/s) to hover on position
        v = self.get_stationary_joint_velocity()
        self.hover_velocities = self.propeller_directions * v
        # maximum angular velocity of motors is:
        # (1 + self.action_range) * hover_velocities
        self.action_range = 0.1
        self.last_action = np.zeros(self.act_dim)
        # By default, PyBullet clamps angular velocities to 100 rad/s
        self.max_joint_velocity = 1000
        for m in self.motor_list:
            self.bc.changeDynamics(self.body_id, m.index,
                                   maxJointVelocity=self.max_joint_velocity)

    def agent_specific_observation(self):
        """ State of cart is of shape (6,) """
        xyz = 0.1 * self.get_position()  # re-scale
        vel = 0.2 * self.get_linear_velocity()
        quat = pb.getQuaternionFromEuler(self.get_orientation())
        rpy_dot = 0.1 * self.get_angular_velocity()
        rotor_speeds = np.array([m.get_velocity() for m in self.motor_list])
        normed_rotor_speeds = 2.5e-3 * (rotor_speeds - self.hover_velocities)
        return np.concatenate((xyz, vel, quat, rpy_dot, normed_rotor_speeds))

    def apply_action(self, motorCommands):
        """ motorCommands are target RMPs of motors."""
        # check validity of action and clip into range [-1, +1]
        clipped_action = super().apply_action(motorCommands)
        xyz, abcd = self.bc.getBasePositionAndOrientation(self.body_id)
        xyz_dot, rpy_dot = self.bc.getBaseVelocity(self.body_id)

        # matrix of size 3x3
        R = np.array(self.bc.getMatrixFromQuaternion(abcd)).reshape((3, 3))

        multiplier = (1 + self.action_range * clipped_action)
        target_velocities = self.hover_velocities * multiplier

        # PyBullet can not simulate air, calculate thrust force of the given
        # joints, and apply it on the link
        for motor, d, v in zip(self.motor_list,
                               self.propeller_directions,
                               target_velocities):
            motor.set_velocity(v)
            linear_velocity = np.array(xyz_dot)  #  Cartesian world linear velocity
            propeller_up_vec = R.dot(np.array([0., 0., 1.]))
            v0 = linear_velocity.dot(propeller_up_vec)
            # compute thrust
            f = self.calculate_thrust_force(
                v * d,
                self.area,
                self.propeller_pitch,
                v0
            )
            # apply force in the simulation
            self.apply_external_force(
                force=np.array([0, 0, f]),
                link_id=motor.index,
                position=(0., 0., 0.),
                frame=pb.LINK_FRAME
            )

    def add_sensor(self, sensor) -> None:
        """ Cart needs to adjust height of sensors."""
        super().add_sensor(sensor)
        self.sensors[-1].set_offset((0, 0, 0.))

    @property
    def alive(self):
        roll, pitch = self.get_orientation()[:2]
        d = 0.5 * np.pi
        return True if abs(pitch) <= d and abs(roll) <= d else False

    def calculate_thrust_force(
            self,
            angular_speed: float,  # [rad/s]
            area: float,
            propeller_pitch: float,
            v0: float = 0,
            air_density: float = 1.225
    ) -> float:
        """Determine thrust force based on current rotor speed. """
        tmp = angular_speed / (2*np.pi) * propeller_pitch
        diameter = (4. * area / np.pi)**0.5
        return air_density * area * (tmp**2 - tmp*v0) * (self.k1 * diameter / propeller_pitch)**self.k2

    def collect_information_after_step(self):
        """Determine if Drone agent touches the ground plane."""
        ground_ids = {(0, -1)}  # (body_id, link_index) tuple
        for i, rotor in enumerate(self.link_list):
            contact_ids = set((x[2], x[4]) for x in rotor.contact_list())
            # check contact between ground and drone link by comparing sets
            link_has_ground_contact = bool(ground_ids & contact_ids)
            self.rotor_ground_contact[i] = float(link_has_ground_contact)
            # if link_has_ground_contact:
            #     print(f'{rotor.name} contacts ground...')
        self.ground_collision_penalty = np.sum(self.rotor_ground_contact)

    def get_stationary_joint_velocity(self) -> float:
        fg = self.mass * self.gravity / 4.
        p = self.propeller_pitch
        j_vel_stationary = (2*np.pi / p) * (fg / (self.air_density * self.area) * (p / (self.k1 * self.diameter))**self.k2)**0.5
        return j_vel_stationary

    def get_stationary_rpm(self) -> float:
        """ Default Stationary RPM: 4407.2195"""
        fg = self.mass * self.gravity / 4.
        p = self.propeller_pitch
        rmp_stat = (60 / p) * (fg / (self.air_density * self.area)
                               * (p / (self.k1 * self.diameter))**self.k2)**0.5
        return rmp_stat

    def specific_reset(self) -> None:
        """ reset motor joints."""
        for j in self.motor_list:
            j.set_state(np.random.uniform(low=-0.1, high=0.1), 0)

    def specific_reward(self) -> float:
        """ Some agents exhibit additional rewards besides the task objective,
            e.g. electricity costs, speed costs, etc. """
        alive_reward = 1.0 if self.alive else -10.0
        xyz_dot, rpy_dot = self.bc.getBaseVelocity(self.body_id)
        # punish fast rotations
        spin_penalty = 0.01 * np.linalg.norm(rpy_dot)**2
        # print('ground_collision_penalty:', self.ground_collision_penalty)
        r = alive_reward - spin_penalty - self.ground_collision_penalty
        return r
