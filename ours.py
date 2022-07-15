try:
    import torch
except ImportError:
    raise ImportError("PyTorch is not installed, impossible to import `alig.th.AliG`")
import numpy as np
import copy
class Ours(torch.optim.Optimizer):

    def __init__(self,params,max_lr=None,weight_decay= 0,):
        if max_lr is not None and max_lr <= 0.0:
            raise ValueError("Invalid max_lr: {}".format(max_lr))
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict( max_lr=max_lr,weight_decay=weight_decay,step_size=None,)
        super(Ours, self).__init__(params, defaults)
    def compute_step_size(self, loss):
        # compute squared norm of gradient
        grad_sqrd_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_sqrd_norm += p.grad.data.norm() ** 2
        self.step_size_unclipped = float(2*loss / (grad_sqrd_norm + 1e-5))
        # compute effective step-size (clipped)
        for group in self.param_groups:
            if group["max_lr"] is not None:
                group["step_size"] = min(self.step_size_unclipped, group["max_lr"])
            else:
                group["step_size"] = self.step_size_unclipped

        # average step size for monitoring
        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))
        print(self.step_size)

    @torch.autograd.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            step_size = group["step_size"]
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['sum'] = torch.zeros_like(p.grad.data)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)

                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.addcdiv_(-step_size, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)  # 更新核心部分
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.add_(-step_size, grad/std)
        return loss
