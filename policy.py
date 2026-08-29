'''
Controller used for all learning algorithms
'''

import flax.linen as nn

class Policy(nn.Module):
    action_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)

        x = nn.Dense(self.action_dim)(x)
        x = nn.softmax(x)

        return x