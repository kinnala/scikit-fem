from typing import Callable


class Form:

    def __init__(self, form: Callable):
        self.form = form

    def __call__(self, *args):
        return self.assemble(self.kernel(*args))

    def kernel(self):
        raise NotImplementedError

    def assemble(self):
        raise NotImplementedError
