class Planner:
    """Agent that plans actions using the world model."""

    def __init__(self, world_model):
        self.world_model = world_model

    def plan(self, observation):
        """Return next action given current observation."""
        raise NotImplementedError
