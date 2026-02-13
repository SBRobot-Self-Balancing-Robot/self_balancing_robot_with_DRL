import typing as T

class BaseControl:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError

    def step(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError
    
    def update(self, *args, **kwargs) -> T.Any:
        """ This method is called at every step to update the control parameters (e.g. heading, speed) based on the current state and/or time. """
        raise NotImplementedError
    
    def generate_random(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError
    
    def error(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError
    
    