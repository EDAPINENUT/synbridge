from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.stats import norm
import torch.distributed as dist

def get_schedule_sampler(config, diffusion) -> "ScheduleSampler":
    name = config.name
    if name == "uniform":
        return UniformSampler(diffusion)    
    elif name == "real-uniform":
        return RealUniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif name == "lognormal":
        return LogNormalSampler(diffusion)  
    elif name == "continuous-uniform":
        return ContinuousUniformSampler(diffusion, getattr(config, 'min_t', 0.01))
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class ContinuousUniformSampler:
    def __init__(self, diffusion, min_t=0.01, max_t=0.99):
        self.diffusion = diffusion
        self.min_t = min_t
        self.max_t = max_t

    def sample(self, batch_size, device):
        ts = torch.rand(batch_size).to(device) * (self.max_t - self.min_t) + self.min_t
        return ts, torch.ones_like(ts)

class RealUniformSampler:
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.sigma_max = diffusion.sigma_max
        self.sigma_min = diffusion.sigma_min

    def sample(self, batch_size, device):
        ts = torch.rand(batch_size).to(device) *(self.sigma_max - self.sigma_min) + self.sigma_min
        return ts, torch.ones_like(ts)


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class LogNormalSampler:
    def __init__(self, diffusion, p_mean=-1.2, p_std=1.2, even=False):
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        # self.mult = diffusion.sigma_max / 80
        
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def sample(self, bs, device):
        if self.even:
            # buckets = [1/G]
            start_i, end_i = self.rank * bs, (self.rank + 1) * bs
            global_batch_size = self.size * bs
            locs = (torch.arange(start_i, end_i) + torch.rand(bs)) / global_batch_size
            log_sigmas = torch.tensor(self.inv_cdf(locs), dtype=torch.float32, device=device)
        else:
            log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        
        sigmas = torch.exp(log_sigmas) #* self.mult
        weights = torch.ones_like(sigmas)
        return sigmas, weights
