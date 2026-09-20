import torch
import math

from torch import nn


######################################################################################
class RSNorm(nn.Module):
    def __init__(self,
                 input_size : int, 
                 *args, 
                 **kwargs):

        super(RSNorm, self).__init__()
        self.input_size     = input_size
        self.mu             = torch.zeros(1, input_size)
        self.var            = torch.ones(1, input_size)
        self.count          = 0
        self.eps            = 1e-6

######################################################################################
    def update_mean_and_var(self, obs):

        assert obs.shape[0] == 1, "We need to update when we transition"

        obs = obs.view(-1)
        delta = obs - self.mu
        self.count += 1


        self.mu = self.mu + (1/self.count) * delta    
        sqred_var = torch.pow(self.var, 2) + (1/self.count) * (torch.pow(delta, 2) - torch.pow(self.var, 2))  
        self.var = torch.sqrt(sqred_var)

######################################################################################
    def normalize_obs(self, obs):

        bs = obs.shape[0]
        rpt_mu = torch.repeat_interleave(self.mu, bs, dim=0)
        rpt_var = torch.repeat_interleave(self.var, bs, dim=0)


        normed_obs = (obs - rpt_mu) / torch.sqrt(torch.pow(rpt_var, 2) + self.eps)
        return normed_obs

######################################################################################
class SimbaInputBlock(nn.Module):
    def __init__(self,
                 input_size : int,
                 action_size : int,
                 hidden_dim : int, 
                 constant : int,
                 normalizer : RSNorm,
                 *args, 
                 **kwargs):
        super(SimbaInputBlock, self).__init__()

        self.normalizer     = normalizer 
        self.action_size    = action_size
        self.constant       = torch.ones(1, 1) * constant
        self.scale_init     = math.sqrt(2 / hidden_dim)
        self.scale_scale    = math.sqrt(2 / hidden_dim)
        self.scale_vector   = nn.Parameter(torch.ones(1, hidden_dim) * self.scale_scale) 

        self.linear         = nn.Linear(input_size + 1, hidden_dim, bias=False)

######################################################################################
    def forward(self, x):
        # Equation 9 and 10

        bs = x.shape[0]
        x, action = torch.split(x, [x.shape[-1] - self.action_size, self.action_size], dim=-1)

        x = self.normalizer.normalize_obs(x)
        x = torch.concat([x, action], dim=-1)
        rpt_c = torch.repeat_interleave(self.constant, bs, dim=0)
        x = torch.concat([x, rpt_c], dim=-1)
        x = nn.functional.normalize(x, dim=-1)
        x = self.linear(x)
        actual_scale = self.scale_vector * (self.scale_init / self.scale_scale)
        x = actual_scale * x
        x = nn.functional.normalize(x, dim=-1)

        return x

######################################################################################
class SimbaEncodingBlock(nn.Module):
    def __init__(self,
                 hidden_dim : int, 
                 number_of_blocks : int,
                 *args, 
                 **kwargs):
        super(SimbaEncodingBlock, self).__init__()

        self.linear_1 = nn.Linear(hidden_dim, hidden_dim*4, bias=False)
        self.linear_2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

        self.scale_init     = math.sqrt(2 / (hidden_dim * 4))
        self.scale_scale    = math.sqrt(2 / (hidden_dim * 4))
        self.scale_vector   = nn.Parameter(torch.ones(1, hidden_dim * 4) * self.scale_scale)

        self.ones           = torch.ones(1, hidden_dim)
        self.alpha_init     = 1 / (number_of_blocks + 1)
        self.alpha_scale    = 1 / math.sqrt(hidden_dim)
        self.alphas         = nn.Parameter(torch.ones(1, hidden_dim) * self.alpha_scale)

######################################################################################
    def forward(self, x):

        input_x = x.clone()
        x = self.linear_1(x)
        actual_scale = self.scale_vector * (self.scale_init / self.scale_scale)
        x = actual_scale * x
        x = nn.functional.relu(x)
        x = self.linear_2(x)
        x = nn.functional.normalize(x, dim=-1)

        # LERP
        actual_alphas = self.alphas * (self.alpha_init / self.alpha_scale)
        one_minus_alpha_h = (self.ones - actual_alphas) * input_x
        alpha_h = actual_alphas * x
        x = nn.functional.normalize(one_minus_alpha_h + alpha_h, dim=-1)

        return x

######################################################################################
class SimbaCritic(nn.Module):

    def __init__(self, 
                 state_dim : int, 
                 **kwargs):
        super(SimbaCritic, self).__init__()
        self.state_dim          = state_dim
        self.hidden_dim         = 512
        self.number_of_block    = 2
        self.output_dim         = 512
        # TODO: this is hard-coded now but we really need to make it a parameter
        self.action_size        = 2

        self.normalizer = RSNorm(
            input_size=state_dim - self.action_size,
            hidden_dim=512
            )

        self.input_encoder = SimbaInputBlock(
            input_size=state_dim,
            action_size=self.action_size,
            hidden_dim=self.hidden_dim,
            constant=3,
            normalizer=self.normalizer
        )

        self.trunk = nn.ModuleList([
            SimbaEncodingBlock(hidden_dim=self.hidden_dim, number_of_blocks=self.number_of_block)
            for _ in range(self.number_of_block)
            ])

        self.logits_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.logits_2 = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.scaler_init = math.sqrt(2 / self.hidden_dim)
        self.scaler_scale = math.sqrt(2 / self.hidden_dim)
        self.scaler_vector = nn.Parameter(torch.ones(1, self.hidden_dim) * self.scaler_scale)

######################################################################################
    def forward(self, x):
        emb = self.input_encoder(x)
        for l in self.trunk:
            emb = l(emb)

        emb = self.logits_1(emb)
        actual_scaler = self.scaler_vector * (self.scaler_init / self.scaler_scale)
        emb = actual_scaler * emb
        emb = self.logits_2(emb)

        return emb
        

######################################################################################

if __name__ == "__main__":
    bs = 32
    input_size = 8
    action_size = 2

    observations = torch.randn(bs, input_size)
    actions = torch.randn(bs, action_size)

    critic_network = SimbaCritic(
        state_dim=input_size + action_size,
    )

    for ob in observations:
        critic_network.normalizer.update_mean_and_var(ob.view(1, -1))

    inputs = torch.concat([observations, actions], dim=-1)
    logits = critic_network(inputs)
    import ipdb; ipdb.set_trace()
    
