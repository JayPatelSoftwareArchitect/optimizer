import torch
import math
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
import numpy as np
def test_op(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         step_: float,
         lr: float,
         epsilon: float,
         race: float):
    r"""Functional API that performs experiment with dynamic optimizing
    """

    for i, param in enumerate(params):
        grads[i] = torch.clamp(grads[i], -2.7183, 2.7183) #Between -e and +e
        grad = grads[i]
        
        if grad is None:
            continue

        exp_avg = torch.clamp(exp_avgs[i], -2.7183, 2.7183) #Between -e and +e
        exp_avg_sq = torch.clamp(exp_avg_sqs[i], -2.7183, 2.7183) #Between -e and +e
        step = torch.mean(torch.angle(grad)).item() #mean|angle(gradient)|
        if step == 0.0:
            bias_correction0 = 0.0 #race condition
        else:
            bias_correction0 = 1 - math.cos(step) 
        
        grad = grad.add(param, alpha=epsilon) #tweeking epsilon
        
        step_size = lr #/ (math.cos(step)) #calculating step size
        # Decay the first and second moment running average coefficient #based on adam moving average
        exp_avg.mul_(bias_correction0).add_(grad, alpha=1 - bias_correction0)
        exp_avg_sq.mul_(bias_correction0).addcmul_(grad, grad, value=1 - step_)
        if bias_correction0 == 0.0:
            bias_correction0 = race - 1
        denom = (exp_avg_sq.sqrt() / bias_correction0).add_(epsilon)

        param.addcdiv_(exp_avg, denom, value=-step_size)
        # Based on this approch, the loss get's decresed drastically. Starts very high.
        # training loss 2658.5133657789565
        # training loss 451.3037321710388
        # training loss 99.9511641255726
        # training loss 9.853981019621635
        # training loss 6.713869736194611
        # training loss 5.840474301636219
        # training loss 2.4905565305948256
        # training loss 4.694802947074175
        # training loss 3.4495118505954743
        # training loss 4.577337420463562
        # training loss 5.225532374978066
        # training loss 3.178789172887802
        # training loss 2.7605492379665373
        # training loss 2.7612814675569535
        # training loss 2.8135490095615387

class Test_OP(Optimizer):
    r"""Implements algorithm.
    """

    def __init__(self, params, lr=1e-3,epsilon=1e-3,step=1e-3, race=0.3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, epsilon=epsilon, step_=step, race=race)
        super(Test_OP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Test_OP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('test_op', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avgs_seq = []
            max_exp_avgs_seq = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avgs'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avgs_seq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['max_exp_avgs_seq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avgs'])
                    exp_avgs_seq.append(state['exp_avgs_seq'])
                    max_exp_avgs_seq.append(state['max_exp_avgs_seq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
                    

            test_op(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avgs_seq,
                   max_exp_avgs_seq,
                   state_steps,
                   step_=group['step_'],
                   lr=group['lr'],
                   epsilon=group['epsilon'],
                   race=group['race']
                )
        return loss
