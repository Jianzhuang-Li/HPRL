import torch.nn as nn
import torch
import os
import numpy as np
from spe.skpl import MultiLayerLinear
from spe.distribution import MutivariateGuassion
from spe.general_utils import AttrDict
from spe.skpl import SkillPriorMdl, LSTMDecoder
from spe.general_utils import no_batchnorm_update
# for test
from config.config import settings
from data.eplus.eplus_dataset import EplusSpirlSplitDataset

class Policy(nn.Module):
    """Base Policy class.
    """
    def __init__(self, action_dim=None, obs_dim=None):
        super().__init__()
        self._rollout_mode = False
        self._is_train = True
        self._action_dim = action_dim
        self._observation_dim = obs_dim
        self.net = self._build_network()

    def forward(self, obs):
        output_dis = self._compute_action_dist(obs)
        # resample actions
        action = output_dis.rsample()
        log_prob = output_dis.log_prob(action)

        return AttrDict(action = action, \
                        log_prob = log_prob, \
                        dist = output_dis)

    def _compute_action_dist(self, obs):
        raise NotImplementedError

    def _build_network(self):
        raise NotImplementedError
    
    def reset(self):
        pass

    def switch_to_val(self):
        self._is_train = False

    def switch_to_train(self):
        self._is_train = True

    def switch_to_rollout(self):
        self._rollout_mode = True

    def switch_to_non_rollout(self):
        self._rollout_mode = False
    
    @property
    def action_dim(self):
        assert self._action_dim is not None, \
              "\'self.action_dim\' is None, set it in child class."
        return self._action_dim
    
    @property
    def observation_dim(self):
        assert self._observation_dim is not None, \
            "\'self.observation\' is None, set it in child class."
        return self._observation_dim

class MLPPolicy(Policy):
    """Multi-layers Linear policy.
    """
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_layers:int):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.observation_dim = input_dim
        self.num_layers = num_layers
        super().__init__(action_dim=action_dim, obs_dim=input_dim)

    def _build_network(self):
        return MultiLayerLinear(input_dim=self.observation_dim,
                                hidden_dim=self.hidden_dim,
                                output_dim=self.action_dim * 2,
                                layers_num=self.num_layers)
    
    def _compute_action_dist(self, obs):
        return MutivariateGuassion(self.net(obs))
    
class DecoderModelPolicy(Policy):
    """Initializes policy network with pretrained skill prior model.
    """
    def __init__(self, model_params):
        """Policy based on the decoder of SkillPriorMdl Model.

        Args:
            model_params (AttriDict): parameters of decoder policy:
                1) latent_dim: the dim of z.
                2) cond_input: as input with z to lstm.
                2) hidden_size: hidden size of LSTM decoder.
                3) num_layers: layers of lstm.
                4) output_size: the action dim.
                5) device: 'cuda' or cpu.
        """
        self.steps_since_hl, self.last_z = np.Inf, None
        self.model_params = model_params
        self.latent_dim = model_params.latent_dim
        self.cond_dim = model_params.cond_dim
        self.hidden_size = model_params.hidden_size
        self.num_layers =model_params.num_layers
        self.output_size = model_params.output_size
        self.device = model_params.device    
        self.load_weight = model_params.load_weight
        self.weight_path = model_params.weight_path
        self.initial_log_sigma = -50
        super().__init__(obs_dim=self.latent_dim + self.cond_dim, action_dim=self.output_size)
        

    def _compute_action_dist(self, obs):
        assert len(obs.shape) == 2
        split_obs = self._split_obs(obs)
        if obs.shape[0] == 1:
            # rollout
            if self.steps_since_hl > self.horizon - 1:
                self.last_z = split_obs.z
                self.steps_since_hl = 0
            act = self.net(self.last_z, split_obs.cond_input, 1)
            self.steps_since_hl += 1
        else:
            # batch update
            act = self.net(split_obs.z, split_obs.cond_input, 1) 
            # [10, 9, 2] [2] -> [10, 9, 2] repeat
        return MutivariateGuassion(mu=act, log_sigma=self._log_sigma[None].unsqueeze(0).repeat(act.shape[0], act.shape[1], 1))

    def _build_network(self):
        net = LSTMDecoder(input_size=self.latent_dim + self.cond_dim,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               output_size=self.output_size)
        if self.load_weight:
            assert os.path.exists(path=self.weight_path), f"path: {self.weight_path} not exist."
            weights_list = os.listdir(self.weight_path)
            assert "decoder.pkl" in weights_list, f"weight: decoder.pkl not exist."
            net.load_state_dict(torch.load(os.path.join(self.weight_path, "decoder.pkl")))
        self._log_sigma = torch.tensor(self.initial_log_sigma * np.ones(self.action_dim, dtype=np.float32), device=self.device, requires_grad=True)
        return net
    
    def _split_obs(self, obs):
        assert obs.shape[1] == self.latent_dim + self.cond_dim
        return AttrDict(
            cond_input = obs[:, : -self.latent_dim],
            z = obs[:, -self.latent_dim:]
        )
    
    @property
    def horizon(self):
        return self.n_rollout_steps
    

class SkillPriorPolicy(Policy):

    def __init__(self, prior_net_params, policy_params, skill_prior_params):
        """ Policy with pretrained prior network.

        Args:
            prior_net_params (AttriDict): Parametes for pre-trained skill prior network.
                It must match the original SkillPriorMdl and haas the following keys:
                1) prior_input_dim: the observation dim defaultly.
                2) prior_hidden_dim
                3) q_embed_dim: the dim of z.
                4) prior_net_layers
                5) prior_weight_path: The path to load prior weights.
            policy_param (AttriDict): Parameters for policy network and need the following arguments:
                1) policy_model: If it is None, we use the same model with prior network.
                2) policy_model_params: Parametes for policy model.
                3) policy_weight_path
                4) load_weight (bool): if use pre_trained policy weights.
            skill_prior_params (AttriDict): parameters from this model:
                1) reverse_KL (bool): if use reverse kl divergence.
                2) max_divergence_range: clamp range for kl(policy_dist||prior_dist).
        """
        self.prior_net_params = prior_net_params
        self.policy_params = policy_params
        self.skill_prior_params = skill_prior_params
        # paramters for prior network
        self.prior_input_dim = prior_net_params.prior_input_dim
        self.prior_hidden_dim = prior_net_params.prior_hidden_dim
        self.q_embed_dim = prior_net_params.q_embed_dim
        self.prior_net_layers = prior_net_params.prior_net_layers
        self.prior_weight_path = prior_net_params.prior_weight_path

        # paramters for policy network
        self.policy_weight_path = policy_params.policy_weight_path
        self.policy_model = policy_params.policy_model
        self.policy_model_params = policy_params.policy_model_params
        self.load_weight = policy_params.load_weight

        # skill prior policy model parameters
        self.reverse_KL = skill_prior_params.reverse_KL
        self.max_divergence_range = skill_prior_params.max_divergence_range

        # pre-trained prior network
        action_dim = self.q_embed_dim
        obs_dim = self.policy_model_params.obs_dim if self.policy_model_params is not None else self.prior_input_dim
        super().__init__(action_dim, obs_dim)
        self.prior_network = self._build_prior_net(self.prior_net_params)
        self.prior_network.load_state_dict(torch.load(os.path.join(self.prior_weight_path, "prior.pkl")))
        

    def _build_network(self):
        """build policy network

        Returns:
            nn.Module: policy network.
        """
        if self.policy_model is not None:
            net = self.policy_model(self.policy_params)
        else:
            net = self._build_prior_net(self.prior_net_params)
        if self.load_weight:
            if self.policy_model is not None:
                net.load_weight(self.policy_weight_path)
            else:
                net.load_state_dict(torch.load(os.path.join(self.policy_weight_path, "prior.pkl")))
        return net

    def _build_prior_net(self, param):
        net = MultiLayerLinear(
            input_dim=param.prior_input_dim,
            hidden_dim=param.prior_hidden_dim,
            output_dim=param.q_embed_dim * 2,
            layers_num=param.prior_net_layers
        )
        return net
    
    def forward(self, obs):
        policy_output = super().forward(obs)
        if not self._rollout_mode:
            raw_prior_divergence, policy_output.prior_dist = self._compute_prior_divergence(policy_output, obs)
            policy_output.prior_divergence = self._clamp_divergence(raw_prior_divergence)
        return policy_output

    def _clamp_divergence(self, divergence):
        return torch.clamp(divergence, -self.max_divergence_range, self.max_divergence_range)
    
    def _compute_prior_divergence(self, policy_output, obs):
        with no_batchnorm_update(self.prior_network):
            prior_dist = MutivariateGuassion(self.prior_network(obs)).detach()
            div = self._analytic_divergence(policy_output, prior_dist)
        return div, prior_dist
    
    def _compute_action_dist(self, obs):
        return MutivariateGuassion(self.net(obs))

    def _analytic_divergence(self, policy_output, prior_dist):
        assert isinstance(prior_dist, MutivariateGuassion)
        assert isinstance(policy_output.dist, MutivariateGuassion)
        if self.reverse_KL:
            return prior_dist.kl_divergence(policy_output.dist).sum(dim=-1)
        else:
            return policy_output.dist.kl_divergence(prior_dist).sum(dim=-1)
    
    def sample_rand(self, obs):
        with torch.no_grad():
            with no_batchnorm_update(self.prior_network):
                prior_dist = MutivariateGuassion(self.prior_network(obs))
        action = prior_dist.sample()
        # TODO tanh_squash output
        return action
        

if __name__ == "__main__":
    prior_net_params = AttrDict(
        prior_input_dim = settings.SkillPrior.prior_input_dim,
        prior_hidden_dim = settings.SkillPrior.prior_hidden_dim,
        q_embed_dim = settings.SkillPrior.q_embed_dim,
        prior_net_layers = settings.SkillPrior.prior_net_layers,
        prior_weight_path = "./weights/2024-11-11_22-27"
    )

    policy_params = AttrDict(
        policy_model = None,
        policy_model_params = None,
        policy_weight_path = "./weights/2024-11-11_22-27",
        load_weight = True
    )

    skill__prior_params = AttrDict(
        reverse_KL = True,
        max_divergence_range = 10
    )

    data_conf = AttrDict()
    data_conf.dataset_spec = AttrDict(subseq_len = 10)
    data_conf.device = "cuda"
    SKP_model = SkillPriorPolicy(prior_net_params= prior_net_params, \
                                  policy_params=policy_params, skill_prior_params=skill__prior_params).to("cuda")
    

    decoder_model_params = AttrDict(
        latent_dim = settings.SkillPrior.q_embed_dim,
        cond_dim = settings.SkillPrior.prior_input_dim,
        hidden_size = settings.SkillPrior.nz_mid_lstm,
        num_layers = settings.SkillPrior.q_lstm_layers,
        output_size = 2,
        device = 'cuda',
        load_weight = True,
        weight_path = "./weights/2024-11-11_22-27",
    )

    decoder_policy = DecoderModelPolicy(decoder_model_params).to('cuda')


    dataset = EplusSpirlSplitDataset("", data_conf=data_conf, phase="val")

    data = dataset[0]
    state = torch.Tensor(data['states']).to("cuda")
    SKP_model.switch_to_non_rollout()
   
    policy_output = SKP_model(state)

    low_input = torch.cat([policy_output["action"], state], dim = -1)
    print(low_input.shape)

    low_output = decoder_policy(low_input)
    print(low_output)
