#!/usr/bin/env python

import unittest
import pybullet as pb
from pybullet_utils import bullet_client
import inspect


class TestAgents(unittest.TestCase):
    @classmethod
    def create_agent(cls, agent_cls):
        bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        agent = agent_cls(
            bc=bc
        )
        print(f'Spawn {agent.name} at init pos:', agent.get_position())
        return agent

    def test_locomotion_agents(self):
        # first import module
        from bullet_safety_gym.envs import agents
        # then check all implemented classes...
        for name, agent_cls in inspect.getmembers(agents):
            if inspect.isclass(agent_cls):
                if name == 'MJCFAgent':
                    continue  # avoid abstract classes
                print('Spawn:', name)
                ag = self.create_agent(agent_cls)
                # check validity of observation and action space
                self.assertTrue(ag.act_dim)
                x = ag.get_observation()
                msg = f'Received: {x.shape[0]} expected: {ag.obs_dim}'
                self.assertTrue(ag.obs_dim == x.shape[0], msg)


if __name__ == '__main__':
    unittest.main()
