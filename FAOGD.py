import torch

class FAOGD(torch.optim.Optimizer):

    def __init__(self, params, max_lr=None, momentum=0, projection_fn=None, eps=1e-5, adjusted_momentum=False):
        if max_lr is not None and max_lr <= 0.0:
            raise ValueError("Invalid max_lr: {}".format(max_lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(max_lr=max_lr, momentum=momentum, step_size=None)
        super(FAOGD, self).__init__(params_list, defaults)

        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.eps = eps

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['std'] = torch.zeros_like(p.data).detach()

        if self.adjusted_momentum:
            self.apply_momentum = self.apply_momentum_adjusted
        else:
            self.apply_momentum = self.apply_momentum_standard

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def compute_step_size(self, loss):
  
        grad_sqrd_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_sqrd_norm += p.grad.data.norm() ** 2
        self.step_size_unclipped = float(loss / (grad_sqrd_norm + self.eps))

        for group in self.param_groups:
            if group["max_lr"] is not None:
                group["step_size"] = min(self.step_size_unclipped, group["max_lr"])
            else:
                group["step_size"] = self.step_size_unclipped


        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))


    @torch.autograd.no_grad()
    def step(self, closure):
        loss = closure()

        self.compute_step_size(loss)

        for group in self.param_groups:
            step_size = group["step_size"]
            momentum = group["momentum"]

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                state['std'].addcmul_(1, grad, grad)
                std = state['std'].sqrt().add_(1e-10)
                p.add_(-step_size, p.grad / std)

                if momentum:
                    self.apply_momentum(p, std, step_size, momentum)

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def apply_momentum_standard(self, p, std, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(-step_size, p.grad / std)
        p.add_(momentum, buffer)

    @torch.autograd.no_grad()
    def apply_momentum_adjusted(self, p, std, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).sub_(p.grad / std)
        p.add_(step_size * momentum, buffer)
