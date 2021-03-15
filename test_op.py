import torch
import math
from torch.optim.optimizer import Optimizer

def test_op(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         lr: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = torch.mean(grad).item()

        bias_correction0 = 1 - abs(math.cos(step))
        
        grad = grad.add(param)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(bias_correction0).add_(grad, alpha=1 - bias_correction0)
        exp_avg_sq.mul_(bias_correction0).addcmul_(grad, grad, value=1 - bias_correction0)
   
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction0)).add_(1e-8)

        param.addcdiv_(exp_avg, denom, value=-bias_correction0)



class Test_OP(Optimizer):
    r"""Implements algorithm.
    """

    def __init__(self, params, lr=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
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
                   group['lr']
                )
        return loss
