from easydict import EasyDict


class BaseAgent:
    def __init__(self, config):
        if isinstance(config, EasyDict) is False:
            config = EasyDict(config)
        self.config = config

    def run(self):
        """
        The main operator
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        """
        raise NotImplementedError
