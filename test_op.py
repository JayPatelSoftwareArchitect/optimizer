import torch
import math
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
import numpy as np
def test_op(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         state_steps,
         step_: float,
         lr: float,
         epsilon: float,
         race: float,
         avg_loss):
    r"""Functional API that performs experiment with dynamic optimizing
    """
    #g(i) = min(max(g(i)​,min_value),max_value) min_value = -1 max_value = 1 
    
    for i, param in enumerate(params):
        grads[i] = torch.clamp(grads[i], -1, 1) 
        grad = grads[i]
        if grad is None:
            continue
        exp_avg = exp_avgs[i] 
        exp_avg_sq = exp_avg_sqs[i]
        step = np.mean(np.angle(grad.numpy())) #mean|angle(gradient)|
        if step == 0.0:
            bias_correction0 = step_ ** torch.log(race) 
        else:
            bias_correction0 = 1 - math.sin(step) 

        grad = grad.add(param, alpha=epsilon) #tweeking epsilon
        step_size = lr * avg_loss  #calculating step size
        
        exp_avg.mul_(bias_correction0).add_(grad, alpha=1 - bias_correction0)
        exp_avg_sq.mul_(bias_correction0).addcmul_(grad, grad, value=1 - step_)
        exp_avg = torch.clamp(exp_avgs[i], -1, 1) 
        exp_avg_sq = torch.clamp(exp_avg_sqs[i], -1, 1) 
        
        denom = (exp_avg_sq.sqrt() / bias_correction0).add_(epsilon)
        param.addcdiv_(exp_avg, denom, value=-step_size)
    
class Test_OP(Optimizer):
    r"""Implements algorithm.
    """

    def __init__(self, params, lr=0.001, epsilon=1e-2, step=5e-3, race=0.07):
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon : {}".format(epsilon))
        if not 0.0 <= step:
            raise ValueError("Invalid step : {}".format(step))
        if not 0.0 <= race:
            raise ValueError("Invalid epsilon : {}".format(race))    

        defaults = dict(lr=lr, epsilon=epsilon, step_=step, race=race)
        super(Test_OP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Test_OP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('test_op', False)
    @torch.no_grad()
    def reset_after_epoch(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Lazy state initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avgs'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avgs_seq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['avg_loss'] = None

    @torch.no_grad()
    def step(self, loss=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        isAvgLossSet = False

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avgs_seq = []
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
                        state['avg_loss'] = None

                    exp_avgs.append(state['exp_avgs'])
                    exp_avgs_seq.append(state['exp_avgs_seq'])
                    
                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
            if isAvgLossSet == False and loss is not None:
                group['race'] = loss
                isAvgLossSet = True 
                if state['avg_loss'] == None:
                    state['avg_loss'] = loss
                else :
                    state['avg_loss'] = torch.mean(torch.stack([state['avg_loss'], loss]))

            test_op(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avgs_seq,
                   state_steps,
                   step_=group['step_'],
                   lr=group['lr'],
                   epsilon=group['epsilon'],
                   race=group['race'],
                   avg_loss=state['avg_loss']
                )
      