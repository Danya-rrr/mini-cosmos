class WorldModel:
    """Compact world model for road scene prediction in CARLA."""

    def __init__(self):
        # TODO: define networks, latent space etc.
        pass

    def encode(self, observation):
        """Encode observation into latent representation."""
        raise NotImplementedError

    def predict(self, latent_state, action):
        """Predict next latent state given current state and action."""
        raise NotImplementedError

    def decode(self, latent_state):
        """Decode latent state back to observation space (optional)."""
        raise NotImplementedError
