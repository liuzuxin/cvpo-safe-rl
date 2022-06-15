import numpy as np
from bullet_safety_gym.envs import bases


class Plane(bases.World):
    def __init__(self, bc, file_name, env_dim, global_scaling=1.):
        super().__init__(
            bc=bc,
            global_scaling=global_scaling,
            env_dim=env_dim
        )
        file_name_path = "plane/" + file_name
        self.plane_id = i = self.bc.loadURDF(file_name_path)
        self.bc.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
        # To enable reflections and plane's alpha must be < 1
        self.bc.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_PLANAR_REFLECTION,1)

    def generate_random_xyz_position(self, max_xy=None):
        if max_xy is None:
            max_xy = int(0.7 * self.env_dim)
        pos = np.concatenate((np.random.uniform(-max_xy, max_xy, 2), [0]))
        return pos


class Plane10(Plane):
    def __init__(self, bc, global_scaling=1., env_dim=10):
        super().__init__(
            bc=bc,
            file_name='plane10.urdf',
            global_scaling=global_scaling,
            env_dim=env_dim
        )


class Plane15(Plane):
    def __init__(self, bc, global_scaling=1., env_dim=15):
        super().__init__(
            bc=bc,
            file_name='plane15.urdf',
            global_scaling=global_scaling,
            env_dim=env_dim
        )


class Plane20(Plane):
    def __init__(self, bc, global_scaling=1., env_dim=20):
        super().__init__(
            bc=bc,
            file_name='plane20.urdf',
            global_scaling=global_scaling,
            env_dim=env_dim
        )


class Plane100(Plane):
    def __init__(self, bc, global_scaling=1., env_dim=100):
        super().__init__(
            bc=bc,
            file_name='plane100.urdf',
            global_scaling=global_scaling,
            env_dim=env_dim
        )


class Plane200(Plane):
    def __init__(self, bc, global_scaling=1., env_dim=250):
        super().__init__(
            bc=bc,
            file_name='plane250.urdf',
            global_scaling=global_scaling,
            env_dim=env_dim
        )


class SmallRoom(Plane10):
    def __init__(self, bc, global_scaling=1, env_dim=10):
        super().__init__(bc, global_scaling, env_dim)
        self.room_id = self.bc.loadURDF("obstacles/room_20x20.urdf",
                                        globalScaling=global_scaling,
                                        useFixedBase=True)


class MediumRoom(Plane15):
    def __init__(self, bc, global_scaling=1, env_dim=15):
        super().__init__(bc, global_scaling, env_dim)
        self.room_id = self.bc.loadURDF("obstacles/room_30x30.urdf",
                                        globalScaling=global_scaling,
                                        useFixedBase=True)


class LargeRoom(Plane20):
    def __init__(self, bc, global_scaling=1, env_dim=20):
        super().__init__(bc, global_scaling, env_dim)
        print('Large Room:', env_dim)
        self.room_id = self.bc.loadURDF("obstacles/room_40x40.urdf",
                                        globalScaling=global_scaling,
                                        useFixedBase=True)


class Octagon(Plane15):
    def __init__(self, bc, global_scaling=1, env_dim=10):
        super().__init__(bc, global_scaling, env_dim)
        self.walls = [] 
        yaw_list = np.linspace(0, 7/8 * 2 * np.pi, num=8)
        for yaw in yaw_list:
            base_orientation = self.bc.getQuaternionFromEuler([0, 0, yaw])
            self.walls.append(self.bc.loadURDF(
                "obstacles/octagon_wall.urdf",
                globalScaling=global_scaling,
                useFixedBase=True,
                baseOrientation=base_orientation)
            )

    def generate_random_xyz_position(self):
        dim_factor = 0.7
        angle = np.random.uniform() * 2 * np.pi
        radius = self.env_dim * np.sqrt(np.random.uniform()) * dim_factor
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos = np.array([x, y, 0])
        return pos
