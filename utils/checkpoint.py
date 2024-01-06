class State:
    def __init__(self, model, optim, scheduler):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.epoch, self.iteration = 0, 0