from typing import Tuple
from .optimizer import _params_t, Optimizer

class Test_OP(Otimizer):
    def __init__(self, params: _params_t, lr: float=...,epsilon:float=...,step:float=...,race:float=...) -> None: ...
