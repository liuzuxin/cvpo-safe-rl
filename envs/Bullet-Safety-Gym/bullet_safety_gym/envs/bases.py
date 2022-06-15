""" Base Classes for the Bullet-Safety-Gym Environments

    Author:             Sven Gronauer (sven.gronauer@tum.de)
    Created:            25.06.2020
    Last major update:  18.12.2020

    Many parts are inspired by the PyBullet Repository, credits to Erwin Coumans
"""
import os
import pybullet as pb
import numpy as np
import abc
from bullet_safety_gym.envs import sensors
from pybullet_utils import bullet_client
import time


def get_data_path() -> str:
    """ Returns the path to the files located in envs/data."""
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    return data_path


class Joint(object):
    """A convenient wrapper to access joint information of the Bullet simulator.
    Many parts of this class are inspired by the PyBullet repository.

    Important Note (from PyBullet documentation):
        "By default, each revolute joint and prismatic joint is motorized using
        a velocity motor. You can disable those default motor by using a maximum
        force of 0. This will let you perform torque control." """
    types = {
        "FIXED": 4,
        "GEAR": 6,
        "PLANAR": 3,
        "POINT2POINT": 5,
        "PRISMATIC": 1,
        "REVOLUTE": 0,  # Bullet converts joints of type continuous to revolute
        "SPHERICAL": 2
    }

    def __init__(
            self,
            bc: bullet_client.BulletClient,
            body_id: int,
            joint_index: int,
            power: float = 1.0,
            max_force=None,
            max_velocity=None,
            debug=False
    ):
        self.bc = bc
        self.body_id = body_id
        self.index = joint_index

        jointInfo = self.bc.getJointInfo(body_id, joint_index)
        self.name = jointInfo[1].decode("utf-8")
        t = jointInfo[2]
        self.type = list(Joint.types.keys())[
            list(Joint.types.values()).index(t)]
        self.damping = jointInfo[6]
        self.friction = jointInfo[7]
        self.lower_limit = jointInfo[8]
        self.upper_limit = jointInfo[9]
        self.max_force = max_force or jointInfo[10] or 1000
        self.max_velocity = max_velocity or jointInfo[11] or 1.0
        self.child_link_name = jointInfo[12]
        self.parent_link_index = jointInfo[16]
        self.power_coefficient = power
        self.has_torque_sensor = False
        self.init_motor()
        # self.enable_torque_control()
        # self.enable_torque_sensor()
        if debug:
            self.print_information()

    @property
    def controllable(self) -> True:
        """

        """


        """ Check if torque can be applied to joint.

        Note:
            Bullet only supports 1-DOF motorized joints at the moment:
            sliding joint or revolute joints.
        """
        controllable_types = ['PRISMATIC', 'REVOLUTE']
        controllable = True if self.type in controllable_types else False
        return controllable

    def disable_motor(
            self
    ) -> None:
        """Makes this joint freely movable."""
        self.bc.setJointMotorControl2(
            self.body_id,
            self.index,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=0,
            targetVelocity=0,
            positionGain=0.1,
            velocityGain=0.1,
            force=0
        )

    def disable_torque_sensor(self) -> None:
        """Disable the torque sensor of this joint."""
        self.bc.enableJointForceTorqueSensor(
            self.body_id,
            self.index,
            0
        )
        self.has_torque_sensor = False

    def enable_torque_control(self):
        """Prismatic and revolute joints are spawned in PyBullet by default in 
        velocity control mode. Disable that for torque control. Disable these
        joint defaults by using a maximum force of 0, which allows you to use
        the torque control mode.
        """
        if self.type in ['PRISMATIC', 'REVOLUTE']:
            self.bc.setJointMotorControl2(
                self.body_id,
                self.index,
                self.bc.VELOCITY_CONTROL,
                force=0
            )

    def enable_torque_sensor(self) -> None:
        """Enable the torque sensor for joint."""
        if not self.has_torque_sensor:
            self.bc.enableJointForceTorqueSensor(
                self.body_id,
                self.index,
                1
            )
        self.has_torque_sensor = True

    def get_relative_position(self) -> tuple:
        """Returns the 2-dim vector (position, velocity) of this joint normed
        into the range [-1, +1]."""
        x, x_dot = self.get_state()
        pos_mid = 0.5 * (self.lower_limit + self.upper_limit)
        diff = self.upper_limit - self.lower_limit
        return 2 * (x - pos_mid) / diff, 0.1 * x_dot

    def get_state(self) -> np.ndarray:
        """Returns the 2-dim vector (position, velocity) of this joint."""
        x, vx, *_ = self.bc.getJointState(
            self.body_id,
            self.index)
        return np.array([x, vx])

    def get_position(self) -> float:
        """ Get position of joint in [rad]."""
        return self.get_state()[0]

    def get_velocity(self) -> float:
        """ Get angular of joint in [rad/s]."""
        return self.get_state()[1]

    def init_motor(self) -> None:
        """Initialize joints in position control mode. The zero force makes them
        freely movable."""
        self.bc.setJointMotorControl2(
            self.body_id,
            self.index,
            pb.POSITION_CONTROL,
            positionGain=0.1,
            velocityGain=0.1,
            force=0
        )

    def print_information(self) -> None:
        """Debug print joint information to console."""
        print('---' * 20)
        print(f' Joint name: \t \t{self.name}')
        print(f' Joint type: \t \t{self.type}')
        print(f' Joint index: \t \t{self.index}')
        print(f' Body index: \t \t{self.body_id}')
        print(f' maximum force: \t \t{self.max_force}')

    def reset_position_and_disable_motor(self, position, velocity):
        self.set_state(position=position,
                       velocity=velocity)
        self.disable_motor()

    def set_position(
            self,
            position: float  # in rad
    ) -> None:
        """Switches joint into position control mode and sets target position.
        """
        self.bc.setJointMotorControl2(
            self.body_id,
            self.index,
            pb.POSITION_CONTROL,
            targetPosition=position,
            force=self.max_force
        )

    def set_state(
            self,
            position: float,  # in rad
            velocity: float  # in rad/s
    ):
        """Set joint to target position and velocity.
        Note: this overrides the physics simulation."""
        self.bc.resetJointState(
            self.body_id,
            self.index,
            targetValue=position,
            targetVelocity=velocity
        )

    def get_torque(self) -> float:
        """Measures the applied force to the joint."""
        msg = f'Torque sensor not enabled for {self.name}'
        assert self.has_torque_sensor, msg
        *_, applied_torque = self.bc.getJointState(
            self.body_id,
            self.index)
        return applied_torque

    def set_torque(
            self,
            torque: float
    ) -> None:
        """Switches joint into torque control mode and sets target torque."""
        self.bc.setJointMotorControl2(
            bodyIndex=self.body_id,
            jointIndex=self.index,
            controlMode=pb.TORQUE_CONTROL,
            force=torque
        )

    def set_velocity(
            self,
            velocity: float
    ) -> None:
        """Switches joint into velocity control mode and sets target velocity.
        """
        self.bc.setJointMotorControl2(
            self.body_id,
            self.index,
            pb.VELOCITY_CONTROL,
            targetVelocity=velocity,
            force=self.max_force
        )


class Link(object):
    """A convenient wrapper to access link information of the Bullet simulator.
    Many parts of this class are inspired by the PyBullet repository.
    """
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            name: bytearray,
            body_id: int,
            link_index: int
    ):
        self.bc = bc
        self.name = name.decode("utf-8")
        self.body_id = body_id
        self.index = link_index
        self.init_position = self.get_position()
        self.init_orientation = self.get_orientation()
        self.init_state = self.get_state()

    def contact_list(self):
        """ Determines the contact points between this link and all other bodies
        of the simulation. Note that pb.getContactPoints() returns the contact
        points computed during the most recent call to pb.stepSimulation().
        """
        return self.bc.getContactPoints(
            self.body_id, -1,
            self.index, -1)

    def get_state(self) -> np.ndarray:
        """Returns the state of the link as 12-dim vector."""
        if self.index == -1:
            # if this link is the base link of robot
            xyz, abcd = self.bc.getBasePositionAndOrientation(self.body_id)
            # Cartesian world velocity
            xyz_dot, rpy_dot = self.bc.getBaseVelocity(self.body_id)
        else:
            # if this link is not the base link of robot
            xyz, abcd, _, _, _, _, xyz_dot, rpy_dot = self.bc.getLinkState(
                self.body_id, self.index, computeLinkVelocity=1)
        # Convert quaternion [a, b, c, d] to Euler [roll, pitch, yaw]
        rpy = pb.getEulerFromQuaternion(abcd)
        return np.concatenate((xyz, xyz_dot, rpy, rpy_dot))

    def get_orientation(self) -> np.ndarray:
        """Returns this link's current orientation as Euler angles (roll, pitch,
         yaw)."""
        return self.get_state()[6:9]

    def get_position(self) -> np.ndarray:
        """Returns this link's position in world coordinates."""
        return self.get_state()[0:3]

    def get_angular_velocity(self) -> np.ndarray:
        """Returns this link's rotational speed in [rad/s]."""
        return self.get_state()[9:12]

    def get_linear_velocity(self) -> np.ndarray:
        """Returns this link's linear velocity in Cartesian world-space
        coordinates."""
        return self.get_state()[3:6]

    def get_quaternion(self) -> np.ndarray:
        """Return this link's current orientation as Quaternion (a, b, c, d)."""
        if self.index == -1:
            # if link is the base link of robot
            _, abcd = self.bc.getBasePositionAndOrientation(self.body_id)
        else:
            _, abcd, _, _, _, _, xyz_dot, rpy_dot = self.bc.getLinkState(
                self.body_id, self.index, computeLinkVelocity=1)
        return np.array(abcd)

    def print_information(self) -> None:
        """Debug print link information to user console."""
        print('---' * 20)
        print(f' Link name: \t \t{self.name}')
        print(f' Link index: \t \t{self.index}')
        print(f' Body index: \t \t{self.body_id}')

    def reset_position(self, position: float) -> None:
        """Reset the position of the link. This overrides physics simulation and
        should be called at episode restart."""
        assert self.index < 0, 'reset_pose() is only implemented for base link.'
        self.bc.resetBasePositionAndOrientation(self.body_id, position,
                                                self.get_orientation())

    def reset_orientation(self, orientation):
        """Reset the orientation of the link. This overrides physics simulation
        and should be called at episode reset."""
        assert self.index < 0, 'reset_pose() is only implemented for base link.'
        self.bc.resetBasePositionAndOrientation(
            self.body_id,
            self.get_position(),
            orientation)

    def reset_velocity(self,
                       linear_velocity=[0, 0, 0],
                       angular_velocity=[0, 0, 0]
                       ) -> None:
        """Reset the linear and angular velocities of the link. This overrides
        physics simulation and should be called at episode restart."""
        self.bc.resetBaseVelocity(
            self.body_id,
            linear_velocity,
            angular_velocity)

    def reset_pose(self,
                   position: np.ndarray,
                   orientation: np.ndarray
                   ):
        """Reset the position and orientation of the link. This overrides the
        physics simulation and should be called at episode restart."""
        assert self.index < 0, 'reset_pose() is only implemented for base link.'
        self.bc.resetBasePositionAndOrientation(
            self.body_id,
            position,
            orientation)


class Body(object):
    """ Base class for physical bodies in simulation, e.g. robots from URDF file
        Bodies hold links and joints (if any). The base link of a body has the
        body_id -1.
    """

    def __init__(
            self,
            bc: bullet_client.BulletClient,
            name: str,
            file_name: str,
            init_color: tuple = (1., 1., 1, 1.0),
            init_xyz: tuple = (0., 0., 0.),
            init_orientation: tuple = (0., 0., 0.),
            fixed_base=False,
            global_scaling=1,
            self_collision=False,
            verbose=False,
            debug=False
    ):
        assert len(init_orientation) == 3, 'init_orientation expects (r,p,y)'
        assert len(init_xyz) == 3
        self.bc = bc
        self.name = name
        self.file_name = file_name
        self.fixed_base = 1 if fixed_base else 0
        self.file_name_path = os.path.join(get_data_path(), self.file_name)
        self.init_xyz = np.array(init_xyz)
        self.init_color = np.array(init_color)
        self.init_orientation = pb.getQuaternionFromEuler(init_orientation)
        self.init_rpy = init_orientation
        self.global_scaling = global_scaling
        self.self_collision = self_collision
        self.visible = True
        self.verbose = verbose
        self.debug = debug

        self.body_id = self._load_body_asset()

        # collect information about loaded object
        self.num_joints = self.bc.getNumJoints(self.body_id)

    def _load_body_asset(self) -> int:
        """Load file from disk and read body information. Expects file format:
            - URDF
            - XML

        Returns
        -------
            body_id of loaded body
        """
        assert os.path.exists(self.file_name_path), \
            f'Did not find {self.file_name} at: {get_data_path()}'

        file_type = self.file_name.split('.')[-1]
        if file_type == 'urdf':
            flags = 0  # zero means no flags are set
            if self.self_collision:
                flags = pb.URDF_USE_SELF_COLLISION  # | pb.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
            body_id = self.bc.loadURDF(
                self.file_name_path,
                self.init_xyz,
                self.init_orientation,
                globalScaling=self.global_scaling,
                useFixedBase=self.fixed_base,
                flags=flags
            )
        elif file_type == 'xml':
            msg = f'global scaling is not supported for MJCF files!'
            assert self.global_scaling == 1, msg
            if self.self_collision:
                body_id = self.bc.loadMJCF(
                    self.file_name_path,
                    flags=self.bc.URDF_USE_SELF_COLLISION |
                          self.bc.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
            else:
                body_id = self.bc.loadMJCF(
                    self.file_name_path
                )
            # loadMJCF returns tuple instead of integer
            if isinstance(body_id, tuple):
                body_id = body_id[0]
        else:
            raise ValueError('Expected file of type URDF or XML.')
        msg = f'Loading object {self.name} from {self.file_name_path} failed.'
        assert body_id >= 0, msg
        return body_id

    def apply_external_force(
            self,
            force=np.zeros(3),
            link_id=-1,
            position=None,
            frame=pb.WORLD_FRAME
    ) -> None:
        """Set force to given link of body. Note that applied external forces 
        are set to zero after PyBullet simulation step. 
        """
        assert force.ndim == 1 and force.shape == (3,)
        self.bc.applyExternalForce(
            self.body_id,
            link_id,
            force,
            posObj=position,
            flags=frame
        )

    def get_state(self):
        """Returns position and velocity of the base link."""
        xyz, abcd = self.bc.getBasePositionAndOrientation(self.body_id)
        xyz_dot, rpy_dot = self.bc.getBaseVelocity(self.body_id)
        # Quaternion (a, b, c, d) to Euler (roll pitch yaw)
        rpy = pb.getEulerFromQuaternion(abcd)
        return np.concatenate([xyz, xyz_dot, rpy, rpy_dot])

    def get_orientation(self) -> np.ndarray:
        """Returns orientation of base link in Euler coordinates:
        np.array([r, p, y]) roll, pitch, yaw
        """
        return self.get_state()[6:9]

    def get_position(self) -> np.ndarray:
        """Returns position of base link in Cartesian coordinates:
        np.array([x, y, z]
        """
        return self.get_state()[:3]

    def get_angular_velocity(self):
        """ returns angular velocity of base link"""
        return self.get_state()[9:12]

    def get_linear_velocity(self):
        """ returns linear velocity (xyz_dot) of base link"""
        return self.get_state()[3:6]

    @property
    def is_visible(self):
        """ Flag indicating if body is visualized in renderer."""
        return self.visible

    def set_mass(self, mass):
        self.bc.changeDynamics(
            self.body_id,
            linkIndex=-1,
            mass=mass
        )

    def set_position(self,
                     position
                     ) -> None:
        """Reset the base of the object at the specified position in world space
        coordinates [X,Y,Z].
        """
        quaternion_orient = pb.getQuaternionFromEuler(self.get_orientation())
        self.bc.resetBasePositionAndOrientation(
            self.body_id,
            position,
            quaternion_orient)

    def set_orientation(self,
                        orientation: tuple
                        ) -> None:
        """Reset body at the specified orientation as world space quaternion
        [X,Y,Z,W] """
        assert len(orientation) == 4, 'expecting quaternion'
        self.bc.resetBasePositionAndOrientation(
            self.body_id,
            self.get_position(),
            orientation)

    def set_velocity(self,
                     linear_velocity=[0, 0, 0],
                     angular_velocity=[0, 0, 0]
                     ) -> None:
        self.bc.resetBaseVelocity(
            self.body_id,
            linear_velocity,
            angular_velocity)

    def print_information(self):
        print('=' * 35)
        print('Robot name:', self.name)
        print('Loaded body_id:', self.body_id, 'at pos:', self.init_xyz)
        print('Number of joints:', self.num_joints)
        print('=' * 35, '\n')


class Agent(Body):
    """An agent embodies multiple links and joints. This class provides an
    interface to exert control commands (so-called actions) to the agent and
    retain information such as rewards, costs, contact information, etc.
    """
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            name,
            file_name,
            act_dim,
            collision_radius: float,
            obs_dim,
            init_xyz,
            max_force: float,
            velocity_constraint: float,
            init_orientation: tuple = (0., 0., 0.),
            fixed_base=False,
            global_scaling=1,
            violation_shape_factor=1.5,
            self_collision=True,
            max_velocity=None,
            debug=False,
            **kwargs
    ):
        Body.__init__(
            self,
            bc=bc,
            name=name,
            file_name=file_name,
            init_xyz=init_xyz,
            init_orientation=init_orientation,
            fixed_base=fixed_base,
            global_scaling=global_scaling,
            self_collision=self_collision,
            debug=debug,
            **kwargs
        )
        self.velocity_constraint = velocity_constraint

        # space information
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        part_name, robot_name = self.bc.getBodyInfo(self.body_id)
        self.robot_name = robot_name.decode("utf8")
        print('Agent name:', self.robot_name) if debug else None
        self._violates_constraints = False
        self.violation_shape = None
        self.violation_shape_factor = violation_shape_factor
        self.size_violation_shape = self.global_scaling * violation_shape_factor
        self.collision_radius = collision_radius

        # Note: number of joints is equal to the number of links
        self.num_joints = self.bc.getNumJoints(self.body_id)
        self.max_force = max_force
        self.max_velocity = max_velocity
        self.joint_dict = {}
        self.joint_list = []
        self.link_dict = {}
        self.link_list = []
        self.motor_list = []
        self.motor_dict = {}
        self.root_link = None
        self._collect_information()

        # sensor information: a sensor is added by a task (e.g. Reach goal)
        self.sensors = []

        # create collision shapes (for cost violation display)
        self.violation_shape = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=self.size_violation_shape,
            rgbaColor=[1, 0, 0, 0.001],  # alpha = 0 renders shadows
            specularColor=[0, 0, 0],
        )
        self.violation_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.violation_shape,
            basePosition=self.get_position()
        )

    def _collect_information(self):
        """Collect joint information about the robot during initialization."""
        for i in range(self.num_joints):
            joint = Joint(
                bc=self.bc,
                body_id=self.body_id,
                joint_index=i,
                max_force=self.max_force,
                max_velocity=self.max_velocity
            )
            self.joint_list.append(joint)
            self.joint_dict[joint.name] = joint
            if joint.controllable and not joint.name.startswith('jointfix') \
                    and not joint.name.startswith('ignore'):
                self.motor_list.append(joint)
                self.motor_dict[joint.name] = joint
            else:
                joint.disable_motor()
            link = Link(
                bc=self.bc,
                name=joint.child_link_name,
                body_id=self.body_id,
                link_index=i
            )
            self.link_list.append(link)
            self.link_dict[link.name] = link

            # if link matches the name of the whole robot, then take it as root
            if joint.child_link_name == self.robot_name:
                self.root_link = link

            if i == 0 and self.root_link is None:
                link = Link(
                    bc=self.bc,
                    name=self.robot_name.encode('utf-8'),
                    body_id=self.body_id,
                    link_index=-1
                )
                print('Root Link:', link.name) if self.debug else None
                self.root_link = link
                self.link_list.append(link)
                self.link_dict[link.name] = link

    def add_sensor(self, sensor) -> None:
        """A sensor is added to the agent by a task, e.g. the goal reach tasks
        adds a sensor to detect obstacles.
        """
        assert isinstance(sensor, sensors.Sensor)
        self.sensors.append(sensor)

    @abc.abstractmethod
    def agent_specific_observation(self) -> np.ndarray:
        """Each agent owns individual state information, e.g. joint positions
         and velocities. Some information might be also irrelevant for certain
         agents, e.g. Ball agent is spherical so its orientation does not
         matter to us."""
        raise NotImplementedError

    @property
    def alive(self) -> bool:
        """Returns "False" if the agent died, "True" otherwise."""
        raise NotImplementedError

    def apply_action(
            self,
            action: np.ndarray
    ) -> np.ndarray:
        """Exert control inputs on the agent's joints. This parent class
        performs a clipping of actions into the interval [-1, +1]."""
        assert np.isfinite(action).all()
        clipped_action = np.clip(action, -1.0, +1.0)
        return clipped_action

    def collect_information_after_step(self) -> None:
        """Some agents need to update internals after the pb.stepSimulation()
        call, e.g. update collision information or feet contact infos."""
        pass

    def get_linear_velocity(self) -> np.ndarray:
        return self.root_link.get_linear_velocity()

    def get_observation(self):
        """Returns agent specific observation plus sensor data."""
        obs = self.agent_specific_observation()
        sensor_obs = np.array([s.get_observation()
                               for s in self.sensors]).flatten()
        obs = np.concatenate([obs, sensor_obs])
        return obs

    def get_orientation(self) -> np.ndarray:
        return self.root_link.get_orientation()

    def get_position(self) -> np.ndarray:
        """ Returns (roll, pitch, yaw) array."""
        return self.root_link.get_position()

    def get_quaternion(self):
        """ Returns quaternion (4-dim vector) array."""
        return self.root_link.get_quaternion()

    @property
    def has_sensor(self) -> bool:
        return True if len(self.sensors) > 0 else False

    def set_sensor(self, sensor: sensors.Sensor, index: int):
        """ Set sensor to agent's sensor list
        """
        assert isinstance(sensor, sensors.Sensor)
        assert index < len(self.sensors)
        self.sensors[index] = sensor

    @abc.abstractmethod
    def specific_reset(self) -> None:
        """ Reset only agent specifics such as motor joints. Do not set position
            or orientation since this is handled by task.specific_reset()."""
        raise NotImplementedError

    @abc.abstractmethod
    def specific_reward(self) -> float:
        """ Some agents exhibit additional rewards besides the task objective,
            e.g. electricity costs, speed costs, etc. """
        return 0.0

    def upgrade_power(self):
        """Some tasks require higher agent powers to encourage exploratory
        actions."""
        pass

    def violates_constraints(
            self,
            does_violate_constraint
    ) -> None:
        """Displays a red sphere which indicates the receiving of costs when
        enable is True, else deactivate visual shape.
        """
        # update position
        self.bc.resetBasePositionAndOrientation(
            self.violation_body_id,
            self.get_position(),
            [0, 0, 0, 1])

        if self._violates_constraints:
            if not does_violate_constraint:
                # make collision visual transparent
                self.bc.changeVisualShape(
                    self.violation_body_id,
                    -1,
                    rgbaColor=[1, 0, 0, 0.0]
                )
        else:
            if does_violate_constraint:
                # display constraint violation visual shape
                self.bc.changeVisualShape(
                    self.violation_body_id,
                    -1,
                    rgbaColor=[1, 0, 0, 0.5]
                )
        self._violates_constraints = does_violate_constraint

    @property
    def velocity_violation(self) -> bool:
        """ Check if agents exceeds speed constraint."""
        vel = np.linalg.norm(self.get_linear_velocity()[:2])
        vel_constraint = True if vel > self.velocity_constraint else False
        return vel_constraint


class Obstacle(Body):
    """ Obstacles are bodies which do not own controllable joints and are mostly
        simple geometric objects. Obstacles are manipulated by applying external
        forces.
    """

    def __init__(self,
                 bc: bullet_client.BulletClient,
                 name,
                 file_name,
                 fixed_base=False,
                 global_scaling=1.,
                 init_xyz=(0., 0., 0.),
                 init_orientation=(0., 0., 0),
                 init_color=(1., 1., 1, 1.0),
                 movement='static',
                 owns_collision_shape=True
                 ):
        super().__init__(
            bc=bc,
            name=name,
            file_name=file_name,
            init_xyz=init_xyz,
            fixed_base=fixed_base,
            global_scaling=global_scaling,
            init_color=init_color,
            init_orientation=init_orientation
        )
        self.owns_collision_shape = owns_collision_shape
        self.movement = movement
        # use offset such that objects exhibit different movement patterns
        self.movement_offset = np.random.uniform(0, 2 * np.pi)
        if movement.lower() != 'static':
            self.constraint = self.bc.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=self.bc.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0])

    def apply_movement(self):
        if self.movement == 'circular':
            r = 0.7
            vel_factor = 1
            t = self.movement_offset
            circle_vec = np.array([np.sin(vel_factor * (time.time() + t)),
                                   np.cos(vel_factor * (time.time() + t)), 0])
            target_pos = np.array(self.init_xyz) + r * circle_vec
            self.bc.changeConstraint(self.constraint,
                                     target_pos)

    @abc.abstractmethod
    def detect_collision(self, agent: Agent) -> bool:
        raise NotImplementedError

    # TODO: collision detection is done with manual loops
    def set_collision_filter(self, agent: Agent) -> None:
        """some child classes might require special collision shape
        treatment, e.g. puddle obstacles."""
        return None

    def set_position(self,
                     position
                     ) -> None:
        """Reset the base of the object at the specified position in world space
        coordinates [X,Y,Z].
        """
        super().set_position(position=position)
        # also reset constraints if body is movable
        self.init_xyz = position  # over-write init xyz for correct movements
        self.apply_movement()

    def update_visuals(self, make_visible: bool):
        if self.is_visible and not make_visible:
            # make obstacle invisible
            self.visible = False
            self.bc.changeVisualShape(
                self.body_id,
                -1,
                rgbaColor=[1, 1, 1, 0.0])
        if not self.is_visible and make_visible:
            # make object visible again
            self.visible = True
            self.bc.changeVisualShape(
                self.body_id,
                -1,
                rgbaColor=self.init_color)


class Task(abc.ABC):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            world,
            agent: Agent,
            obstacles: list,
            continue_after_goal_achievement: bool,
            use_graphics: bool,
            agent_obstacle_distance: float = 2.5
    ):
        self.agent = agent
        self.agent_obstacle_distance = agent_obstacle_distance
        self.bc = bc
        self.continue_after_goal_achievement = continue_after_goal_achievement
        self.obstacles = obstacles
        self.world = world
        self.use_graphics = use_graphics
        self.setup_camera()

    @abc.abstractmethod
    def calculate_cost(self):
        """Implements the task's specific cost function."""
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_reward(self):
        """Implements the task's specific reward function, which depends on
        the agent and the surrounding obstacles."""
        raise NotImplementedError

    def equip_agent_with_sensors(self):
        """Agents must be equipped with sensors to detect obstacles. If physical
        obstacles (owning collision shapes) are present, a LIDAR sensor is
        added, otherwise a Pseudo LIDAR is used to detect non-physical obstacles(
        owning only visual but no collision shape).
        """
        # check if objects have collision shapes
        collision_shapes = [ob.owns_collision_shape for ob in self.obstacles]
        if any(collision_shapes):
            # add LIDAR sensor to agent if obstacles with collision shapes are
            # present
            sensor = getattr(sensors, 'LIDARSensor')(
                bc=self.bc,
                agent=self.agent,
                obstacles=self.obstacles,
                number_rays=24,
                ray_length=self.world.env_dim / 2,
                visualize=self.use_graphics
            )
            self.agent.add_sensor(sensor)
        # check if at least one obstacle without collision shape is present
        if len(collision_shapes) > 0 and False in collision_shapes:
            sensor = getattr(sensors, 'PseudoLIDARSensor')(
                bc=self.bc,
                agent=self.agent,
                obstacles=[o for o in self.obstacles if
                           not o.owns_collision_shape],
                number_rays=24,
                ray_length=self.world.env_dim / 2,
                visualize=False  # Pseudo rays are not rendered in GUI
            )
            self.agent.add_sensor(sensor)

    @abc.abstractmethod
    def get_collisions(self) -> int:
        """Returns the number of collisions that occurred after the last
        simulation step call."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self):
        """Returns a task related observation. May be empty for some tasks."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def goal_achieved(self) -> bool:
        """Check if task specific goal is achieved."""
        raise NotImplementedError

    @abc.abstractmethod
    def setup_camera(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_reset(self) -> None:
        """ Set positions and orientations of agent and obstacles."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_goal(self):
        raise NotImplementedError


class World:
    def __init__(self,
                 bc: bullet_client.BulletClient,
                 global_scaling: float,
                 env_dim: float,
                 body_min_distance: float = 2.5):
        self.bc = bc
        self.global_scaling = global_scaling
        self.body_min_distance = body_min_distance

        # setup render settings: default setup is suitable for the run tasks
        self.camera = Camera(
            bc=bc,
            cam_base_pos=(-3, 0, 0),
            cam_dist=16 * global_scaling,
            cam_yaw=0,
            cam_pitch=-89
        )
        self.env_dim = env_dim

    @abc.abstractmethod
    def generate_random_xyz_position(self):
        pass


class Camera:
    def __init__(self,
                 bc: bullet_client.BulletClient,
                 cam_base_pos: tuple,
                 cam_dist: float,
                 cam_yaw: float,
                 cam_pitch: float,
                 render_width: int = 480,
                 render_height: int = 360
                 ):
        # setup render settings
        self.bc = bc
        self.cam_base_pos = cam_base_pos
        self.cam_dist = cam_dist
        self.cam_yaw = cam_yaw
        self.cam_pitch = cam_pitch
        self.render_width = render_width
        self.render_height = render_height

    def update(self,
               cam_base_pos=None,
               cam_dist=None,
               cam_yaw=None,
               cam_pitch=None) -> None:
        if cam_base_pos:
            self.cam_base_pos = cam_base_pos
        if cam_dist:
            self.cam_dist = cam_dist
        if cam_yaw:
            self.cam_yaw = cam_yaw
        if cam_pitch:
            self.cam_pitch = cam_pitch
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=self.cam_base_pos,
            cameraDistance=self.cam_dist,
            cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch
        )
