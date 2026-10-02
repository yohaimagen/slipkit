"""Configuration for the verified serial CPU linear backend."""
import os
import math


class AltarConfigBuilder:
    def __init__(self, n_parameters, n_observations, case_dir, chains=1024,
                 steps=1000, tasks=1, output_dir='results', output_freq=1,
                 seed=17, prior='gaussian', initial_scaling=.1, acceptance_weight=8/9, rejection_weight=1/9):
        for name, value in dict(n_parameters=n_parameters, n_observations=n_observations,
                                chains=chains, steps=steps, output_freq=output_freq).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if tasks != 1:
            raise ValueError('Only serial CPU linear execution is validated (tasks=1).')
        if prior not in ('gaussian', 'uniform'):
            raise ValueError('Supported native priors are gaussian and uniform.')
        if not isinstance(seed, int) or seed < 0:
            raise ValueError('seed must be a nonnegative integer.')
        if not math.isfinite(initial_scaling) or not .01 <= initial_scaling <= 1:
            raise ValueError('initial_scaling must lie in [0.01,1].')
        if not all(math.isfinite(v) and v >= 0 for v in (acceptance_weight, rejection_weight)) or not .01 <= rejection_weight <= acceptance_weight+rejection_weight <= 1:
            raise ValueError('Proposal weights must yield scales in [0.01,1].')
        self.initial_scaling, self.acceptance_weight, self.rejection_weight = initial_scaling, acceptance_weight, rejection_weight
        self.n_parameters, self.n_observations = n_parameters, n_observations
        self.case_dir, self.output_dir = os.path.abspath(case_dir), output_dir
        if any(c in str(path) for path in (self.case_dir, self.output_dir) for c in '\n\r'):
            raise ValueError('Configuration paths cannot contain newlines.')
        self.chains, self.steps, self.output_freq = chains, steps, output_freq
        self.seed, self.prior = seed, prior

    def build(self):
        lines = ['linear:', '    model:', f'        case = {self.case_dir}',
                 f'        parameters = {self.n_parameters}', f'        observations = {self.n_observations}']
        for name in ('prep', 'prior'):
            lines += [f'        {name} = altar.distributions.{self.prior}', f'        {name}:',
                      '            parameters = {linear.model.parameters}']
            lines += (['            mean = 0', '            sigma = 1'] if self.prior == 'gaussian'
                      else ['            support = (0, 1)'])
        lines += [f'    rng.seed = {self.seed}', '    controller:', '        sampler:',
                  f'            scaling = {self.initial_scaling}', f'            acceptanceWeight = {self.acceptance_weight}',
                  f'            rejectionWeight = {self.rejection_weight}', '        archiver:',
                  f'            output_dir = {self.output_dir}', f'            output_freq = {self.output_freq}',
                  '    job.tasks = 1', '    job.gpus = 0', f'    job.chains = {self.chains}',
                  f'    job.steps = {self.steps}', '']
        return '\n'.join(lines)

    def save(self, filepath):
        with open(filepath, 'w') as stream:
            stream.write(self.build())
        return os.path.abspath(filepath)
