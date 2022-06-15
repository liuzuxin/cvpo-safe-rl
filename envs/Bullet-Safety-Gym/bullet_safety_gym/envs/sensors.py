import pybullet as pb
import math
import abc
import numpy as np


class Sensor(abc.ABC):
    """ Baseclass for sensor units."""
    def __init__(
            self,
            bc,
            offset: list,
            agent,
            obstacles: list,
            coordinate_system: int,
            rotate_with_agent: bool,
            visualize: bool
    ):
        self.agent = agent
        self.bc = bc
        self.coordinate_system = coordinate_system
        self.obstacles = obstacles
        self.offset = np.array(offset)
        self.rotate_with_agent = rotate_with_agent
        self.visualize = visualize  # indicates if rendering is enabled

    def get_observation(self) -> np.ndarray:
        """Synonym method for measure()."""
        return self.measure()

    @abc.abstractmethod
    def measure(self, *args, **kwargs) -> np.ndarray:
        """ Collect information about nearby and detectable objects/bodies."""
        raise NotImplementedError

    def set_offset(self, offset: list) -> None:
        """ By default, the sensor is placed at agent's root link position.
            However, sometimes it is useful to adjust the position by an offset.
        """
        assert np.array(offset).shape == (3, )
        self.offset = np.array(offset)

    # @property
    # def type(self):
    #     return self.__class__

    @property
    @abc.abstractmethod
    def shape(self) -> tuple:
        """ Get the sensor dimension as vector."""
        raise NotImplementedError


class LIDARSensor(Sensor):
    """ A sensor that performs radial ray casts to collect intersection
        information about nearby obstacles.

        Note: until now, only the world frame coordination system is supported.

    """
    supported_frames = [pb.WORLD_FRAME]

    def __init__(
            self,
            bc,
            agent,
            number_rays,
            ray_length,
            obstacles,
            offset=(0, 0, 0),
            coordinate_system=pb.WORLD_FRAME,
            hit_color=(0.95, 0.1, 0),
            miss_color=(0.25, 0.95, 0.05),
            rotate_with_agent=True,  # Rotate LIDAR rays with agent
            visualize=True
    ):

        assert pb.MAX_RAY_INTERSECTION_BATCH_SIZE > number_rays
        assert coordinate_system in LIDARSensor.supported_frames

        super().__init__(
            bc=bc,
            agent=agent,
            obstacles=obstacles,
            offset=offset,
            coordinate_system=coordinate_system,
            rotate_with_agent=rotate_with_agent,
            visualize=visualize)

        self.number_rays = number_rays
        self.ray_length = ray_length
        self.ray_width = 0.15

        # visualization parameters
        self.replace_lines = True
        self.hit_color = hit_color
        self.miss_color = miss_color

        # collect measurement information
        self.rays_starting_points = []
        self.rays_end_points = []
        self.ray_ids = self.init_rays()

    def init_rays(self) -> list:
        """ Spawn ray visuals in simulation and collect their body IDs.
            Note: Rays are spawned clock-wise beginning at 12 clock position.
        """
        from_position = np.array([0, 0, 0]) + self.offset
        ray_ids = []
        for i in range(self.number_rays):
            if self.replace_lines and self.visualize:
                end_point = [1, 1, 1]
                ray_ids.append(
                    self.bc.addUserDebugLine(from_position, end_point,
                                        self.miss_color, lineWidth=self.ray_width))
            else:
                ray_ids.append(-1)

        return ray_ids

    def set_ray_positions(self, from_position=None, shift=0.):
        if from_position is None:
            from_position = self.agent.get_position()
        assert from_position.shape == (3, ), f'Got shape={from_position.shape}'
        self.rays_starting_points = []
        self.rays_end_points = []

        if self.rotate_with_agent:
            abcd = self.agent.get_quaternion()
            R = np.array(self.bc.getMatrixFromQuaternion(abcd)).reshape((3, 3))
            start_pos = from_position + R @ self.offset
        else:
            start_pos = from_position + self.offset
        for i in range(self.number_rays):
            self.rays_starting_points.append(start_pos)
            angle = (2. * math.pi * float(i)) / self.number_rays + shift
            dx = self.ray_length * math.sin(angle)
            dy = self.ray_length * math.cos(angle)
            dz = 0.
            if self.rotate_with_agent:
                # Rotate LIDAR rays with agent
                rotated_delta_xyz = R @ np.array([dx, dy, dz])
                end_point = start_pos + rotated_delta_xyz
            else:
                # Keep rays parallel to the ground -> no rotation with agent
                end_point = start_pos + np.array([dx, dy, dz])
            self.rays_end_points.append(end_point)

    @property
    def shape(self) -> tuple:
        return (self.number_rays, )

    def render(self, data) -> None:
        """ Display and update ray visuals."""
        if not self.visualize:
            # Do not draw debug lines when visuals are not rendered
            return

        for i in range(self.number_rays):
            hitObjectUid = data[i][0]

            if hitObjectUid < 0:  # no object intersection
                # hitPosition = [0, 0, 0]
                self.bc.addUserDebugLine(
                    self.rays_starting_points[i],
                    self.rays_end_points[i],
                    self.miss_color,
                    lineWidth=self.ray_width,
                    replaceItemUniqueId=self.ray_ids[i])
            else:
                hitPosition = data[i][3]
                self.bc.addUserDebugLine(
                    self.rays_starting_points[i],
                    hitPosition,
                    self.hit_color,
                    lineWidth=self.ray_width ,
                    replaceItemUniqueId=self.ray_ids[i])

    def measure(self, from_position=None) -> np.ndarray:
        """
            origin_position: list holding 3 entries: [x, y, z]
        """
        self.set_ray_positions(from_position)  # if self.visualize else None
        # Detect distances to close bodies via ray casting (sent as batch)
        results = self.bc.rayTestBatch(
            self.rays_starting_points,
            self.rays_end_points,
            # parentObjectUniqueId=self.agent.body_id
        )
        if not self.replace_lines:
            self.bc.removeAllUserDebugItems()
        self.render(data=results)

        # distances to obstacle in range [0, 1]
        # 1: close to sensor
        # 0: not in reach of sensor
        distances = [1.0 - d[2] for d in results]

        return np.array(distances)


class PseudoLIDARSensor(LIDARSensor):
    """ A sensor that loops over all obstacles in the simulation and divides
    the measured distances into bins.

    Note: this sensor class does not use ray casting.
    """
    supported_frames = [pb.WORLD_FRAME]  # only world frame supported yet.

    def __init__(
            self,
            bc,
            agent,
            obstacles,
            number_rays,
            ray_length,
            offset=(0, 0, 0),
            coordinate_system=pb.WORLD_FRAME,
            visualize=True
    ):
        assert number_rays > 0
        assert coordinate_system in PseudoLIDARSensor.supported_frames
        super().__init__(
            bc=bc,
            agent=agent,
            number_rays=number_rays,
            obstacles=obstacles,
            offset=offset,
            ray_length=ray_length,
            coordinate_system=coordinate_system,
            hit_color=(0.95, 0.1, 0),
            miss_color=(0.25, 0.85, 0.85),
            rotate_with_agent=False,  # Pseudo rays are not rotated with agent..
            visualize=visualize
        )
        self.number_rays = number_rays
        self.ray_length = ray_length
        self.ray_width = 0.15

    def calculate_angle_and_dist_to(self, obstacle) -> tuple:
        """determines angle between agent and obstacles based on world frame
        coordinates."""
        x, y = self.agent.get_position()[:2] - obstacle.get_position()[:2]
        angle = np.arctan2(x, y)  # in [-3.14, +3.14]
        # make angle in [0, 2*pi] beginning from 6 o'clock position and counting
        # in clock-wise direction
        angle += np.pi

        return angle, np.linalg.norm([x, y])

    def measure(self, from_position=None, *args, **kwargs) -> np.ndarray:
        """ Returns distances to obstacles in range [0, 1]
            1: close to sensor
            0: not in reach of sensor
        """
        # shift the rays of Pseudo LIDAR for visual correctness
        shift = 2 * np.pi / (2 * self.number_rays)
        if self.visualize:
            self.set_ray_positions(from_position, shift=shift)

        bins = np.zeros(self.number_rays)
        bin_size = 2 * np.pi / self.number_rays
        for ob in self.obstacles:
            angle, dist = self.calculate_angle_and_dist_to(ob)
            if dist <= self.ray_length:  # only regard when in reach of sensor
                b = int(angle // bin_size)  # determine which bin obs belongs
                hit_distance = 1 - dist / self.ray_length
                # update bin if current obstacles is closer than prior
                if bins[b] < hit_distance:
                    bins[b] = hit_distance
        # display rays in simulation
        self.render(bins) if self.visualize else None
        return bins

    def render(self, hit_distances: np.ndarray) -> None:
        """display rays in simulation"""
        assert self.visualize
        for i in range(self.number_rays):
            color = self.hit_color if hit_distances[i] > 0 else self.miss_color
            # adjust length of ray when object hit
            if hit_distances[i] > 0:
                diff = self.rays_end_points[i] - self.rays_starting_points[i]
                start = self.rays_starting_points[i]
                self.rays_end_points[i] = (1 - hit_distances[i]) * diff + start
            self.bc.addUserDebugLine(
                self.rays_starting_points[i],
                self.rays_end_points[i],
                color,
                lineWidth=self.ray_width,
                replaceItemUniqueId=self.ray_ids[i])
