from torch.optim.lr_scheduler import _LRScheduler


def polylr(optim, mdsteps, end_lr, power):
    '''
    Return PolynomialLRDecay scheduler object.

    Parameterts
    -----------
    optimizer : Optimizer
        optimizer object.

    mdsteps : int
        max steps for decay lr.

    end_lr : float
        last value for lr, no decay after that value.

    power : float
        decay value.
    '''
    return PolynomialLRDecay(optim, mdsteps, end_lr, power)


class PolynomialLRDecay(_LRScheduler):
    '''
    Class used to create custom PolynomialLRDecay scheduler.

    Attributes
    ----------
    max_decay_steps : int
        max steps for decay lr.

    end_learning_rate : float
        last value for lr, no decay after that value.

    power : float
        decay value.

    last_step : int
        current last step number.

    Methods
    -------
    get_lr()
        calculate current lr.

    step(step : int, optional)
        do scheduler step, change lr.
    '''

    def __init__(self, optimizer, max_decay_steps, end_learning_rate, power):
        '''
        Parameters
        ----------
        optimizer : Any
            optimizer object.

        max_decay_steps : int
            max steps for decay lr.

        end_learning_rate : float
            last value for lr, no decay after that value.

        power : float
            decay value.
        '''
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        '''
        Calculate current lr.
        '''
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        end_lr = self.end_learning_rate
        mult_val = ((1 - self.last_step / self.max_decay_steps)
                    ** (self.power))
        res_lst = [(base_lr - self.end_learning_rate) *
                   mult_val + end_lr for base_lr in self.base_lrs]

        return res_lst

    def step(self, step=None):
        '''
        Do scheduler step, change lr.

        Parameters
        ----------
        step : int, optional
            current step value.
        '''
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1

        if self.last_step <= self.max_decay_steps:
            end_lr = self.end_learning_rate
            mult_val = ((1 - self.last_step / self.max_decay_steps)
                        ** (self.power))
            decay_lrs = [(base_lr - self.end_learning_rate) * mult_val +
                         end_lr for base_lr in self.base_lrs]

            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
