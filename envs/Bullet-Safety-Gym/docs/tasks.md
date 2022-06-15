# Tasks


- Run: Agents must run as fast as possible into positive x-direction. The y-axis is limited with safety boundaries which are non-physical objects which can be penetrated without collision.
    - Rewards: increase with the velocity of the agent towards the positive x-axis.
    - Costs: are received when the agent exceeds a given speed limit or leaves a pre-defined safety zone.
- Circle: Agents are incetivized to move along a circle in clock-wise direction. (proposed in [1])
    - Rewards: The reward is maximized when the agents move on the circle in clock-wise direction as fast as possible. 
    $r(s) = \frac{v^T [-y, x]}{1+3|r_{agent}-r_{circle}|}$
    - Costs: are obtained when agent leaves the safety zone denoted by the two boundaries, i.e. $c(s) = \mathbb{1}[|x| > x_{lim}]$ where x is the position on the x-axis.
- Gather: Agents spawn in a room and are expected to collect as many apples as possible while avoiding bombs.
    - Rewards: are sparse and positive rewards are perceived by agent when coming close to the apples
    - Costs are likewise sparse and only received when agent comes close to bombs
- Reach: The agents are supposed to move towards a goal zone. As soon the agents come within the goal zone, the goal is spawn such that the agent has to achieve a series of goal positions in order to maximize its reward.
    - Rewards: shaped rewards based euclidean distance when moving closer to the goal ; sparse reward when agent within goal zone
    - Costs: agent receive costs when hitting physical obstacles or walking over non-collision bodies
