class TrainingLoop:
    """Handles training of the world model."""

    def __init__(self, world_model, data_loader, optimizer):
        self.model = world_model
        self.data = data_loader
        self.opt = optimizer

    def train_step(self):
        raise NotImplementedError

    def train(self, epochs=1):
        for epoch in range(epochs):
            self.train_step()
