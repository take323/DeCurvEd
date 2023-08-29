import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal

__all__ = ["CNF", "SequentialFlow"]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, support_sets_mask, target_shift_magnitudes, logpx=None, inds=None, integration_times=None):
        """
        x: original latent codes
        support_set_mask: index k
        target_shift_magnitudes: epsilon
        """
        if inds is None:
            inds_reverse = range(len(self.chain) - 1, -1, -1)
            inds_forward = range(len(self.chain))

        # Map latent codes to Cartesianized latent space.
        for i in inds_forward:
            x = self.chain[i](x, None, integration_times)
        # Shift latent codes in Cartesianized latent space.
        x_new = x + support_sets_mask * target_shift_magnitudes.reshape(-1, 1)

        # Map latent codes to original latent space.
        for i in inds_reverse:
            x_new, logpx = self.chain[i](x_new, logpx, integration_times, True)
        return x_new, logpx

    def manipulate(self, x, support_sets_mask, inds=None, integration_times=None):
        if inds is None:
            inds_reverse = range(len(self.chain) - 1, -1, -1)
            inds_forward = range(len(self.chain))

        # map to cartesianized latent space
        for i in inds_forward:
            x = self.chain[i](x, None, integration_times)
        # shift in cartesianized latent space
        x_new = x
        for i in range(len(support_sets_mask)):
            x_new[i+1] = x_new[i] + support_sets_mask[i]

        # map to original latent space
        for i in inds_reverse:
            x_new = self.chain[i](x_new, None, integration_times, True)
        return x_new


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(CNF, self).__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
            print("Training T :", self.T)
        else:
            print('not training T :', self.T)

        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, x, logpx=None, integration_times=None, reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x.device)
        else:
            _logpx = logpx

        states = (x, _logpx)

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.tensor(0.0).to(x.device), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x.device)
            else:
                integration_times = torch.tensor([0., self.T], requires_grad=False).to(x.device)

        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x.device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x.device),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]

        if logpx is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
