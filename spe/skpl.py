from contextlib import contextmanager
import torch
import os
import logging
from torch.nn.modules import ReLU
from spe.distribution import Gaussion, MutivariateGuassion
from config.config import settings
from spe.general_utils import AttrDict
from spe.general_utils import TensorModule
from spe.general_utils import WeightSaver

class MultiLayerLinear(torch.nn.Module):

    def __init__(self, 
                input_dim,
                hidden_dim,
                output_dim,
                layers_num,
                hidden_normal=False,
                active= torch.nn.ReLU
            ):
        super(MultiLayerLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.active = active
        self.hidden_normal = hidden_normal

        self.net = torch.nn.Sequential()
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(torch.nn.ReLU())
        for _ in range(self.layers_num):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(torch.nn.ReLU())
        self.net.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, input):
        return self.net(input)


class MultiGussionDistri(MultiLayerLinear):

    def __init__(self, ladent_dim, input_dim, hidden_dim, output_dim, layers_num, hidden_normal=False, active=torch.nn.ReLU):
        super().__init__(input_dim, hidden_dim, output_dim, layers_num, hidden_normal, active)
        self.ladent_dim = ladent_dim
        self.mu_layer = torch.nn.Linear(output_dim, ladent_dim)
        prior_out_dim = (ladent_dim + 1) * ladent_dim // 2
        self.log_sigma_layer = torch.nn.Linear(output_dim, prior_out_dim)

    def forword(self, input):
        result = super().forword(input)
        mu = self.mu_layer(result)
        log_sigma = self.log_sigma_layer(result)
        output_dim = self.ladent_dim
        L = torch.zeros(input.size(0), output_dim, output_dim).to(input.device)
        idx = 0
        for i in range(output_dim):
            for j in range(i + 1):
                L[:, i, j] = log_sigma[:, idx]
                idx += 1
        cov = torch.bmm(L, L.transpose(1, 2))
        mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        return mvn.sample()


class LSTMEncoder(torch.nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, action, cond_input=None):
        if cond_input is not None:
            encode_input = torch.cat((cond_input, action), dim=-1)
        else:
            encode_input = action
        h0 = torch.zeros(self.num_layers, encode_input.size(0), self.hidden_size).to(encode_input.device)
        c0 = torch.zeros(self.num_layers, encode_input.size(0), self.hidden_size).to(encode_input.device)
        
        # forward lstm
        out, (hn, cn) = self.lstm(encode_input, (h0, c0))
        last_out = hn[-1,:,:]

        # Decode the hidden state of the last time step
        encoded = self.fc(last_out)
        return encoded
    
class LSTMDecoder(LSTMEncoder):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 ):
        super().__init__(input_size, hidden_size, output_size, num_layers)

    def forward(self, z, cond_input, seq_length):
        if cond_input is not None:
            decode_input = torch.cat((cond_input, z), -1)
        else:
            decode_input = z
        x = decode_input.unsqueeze(1).repeat(1, seq_length, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        recon_x = self.fc(out)
       
        return recon_x


class SkillPriorMdl(torch.nn.Module):
    """Skill embeding + prior model for HPRL algorithm.
    """
    def __init__(self, params):
        """Skill Prior Model parameters.

        Args:
            model_params (AttriDict): parameters dictionary:
            (1) encoder_params:
                - cond_encode (bool): if use condition encode. -> (true)
                - cond_encode_dim (int) -> (15)
                - lstm_hidden_dim_enc (int) -> (64)
                - lstm_layers_enc (int) -> (1)
            (2) decoder_params:
                - cond_decode (bool): if use condition decode. -> (true)
                - cond_decode_dim (int) -> (15)
                - lstm_hidden_dim_dec (int) -> (64)
                - lstm_layers_dec (int) -> (1)
                - rec_mse_weight: reconstruction mse loss weight. -> (1)
                - n_rollout_stpes: rollout steps.
            (3) prior_params:
                - prior_input_size (int) -> (15)
                - hidden_dim_pr (int) -> (128)
                - num_layers (int) -> (6)
            (4) beta_params:
                - kl_div_weight: the const alpha. -> (0.005)
                - target_kl: the target kl diverigence used to updata alpha. -> (0.18)
                - optimize_beta: if update the temperature coefficient. -> (true)
                - lr: learning rate if use optimize beta. -> (3e-4)
            (5) model_params:
                - device: 'cuda' or 'cpu'. -> ('cuda')
                - action_dim: action dim of the environment. -> (2)
                - state_dim: state dim of the environmet. -> (15)
                - z_size: the skill space dim -> (10)
        """
        super(SkillPriorMdl, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.params = params
        
        self._default_params(params)
        self._sample_prior = False
        self.q = self._build_inference_network()
        self.decoder = self._build_reconstruction_network()
        self.prior = self._build_prior_network()
        self._build_beta_network()

    def _default_params(self, new_params):
         # model params
        self.model_params = AttrDict(
            device = 'cuda',
            action_dim = 2,
            state_dim = 15,
            z_size = 10,
            skill_len = 9
        )
        # inference parameters
        self.encoder_params = AttrDict(
            cond_encode = True,
            cond_encode_dim = self.model_params.state_dim,
            lstm_hidden_dim_enc = 128,
            lstm_layers_enc = 1
        )
        if 'encoder_params' in new_params.keys():
            self.encoder_params.update(new_params.encoder_params)
        # decoder paramters
        self.decoder_params = AttrDict(
            cond_decode = True,
            cond_decode_dim = self.model_params.state_dim,
            lstm_hidden_dim_dec = 128,
            lstm_layers_dec = 1,
            rec_mse_weight = 1.0,
            n_rollout_steps = self.model_params.skill_len
        )
        if 'decoder_params' in new_params.keys():
            self.decoder_params.update(new_params.decoder_params)
        # prior_params
        self.prior_params = AttrDict(
            prior_input_size = self.model_params.state_dim,
            hidden_dim_pr = 128,
            num_layers = 6
        )
        if 'prior_params' in new_params.keys():
            self.prior_params.update(new_params.prior_params)
        # beta params
        self.beta_params = AttrDict(
            kl_div_weight = 0.0005,
            target_kl = 0.18,
            optimize_beta = False,
            lr = 3e-4
        )
        if 'beta_params' in new_params.keys():
            self.beta_params.update(new_params.beta_params)
       
        if 'model_params' in new_params.keys():
            self.model_params.update(new_params.model_params)
    

    def _build_inference_network(self):
        self.cond_encode = self.encoder_params.cond_encode
        self.cond_encode_dim = self.encoder_params.cond_encode_dim
        self.infer_input_size = self.model_params.action_dim
        if self.cond_encode:
            self.infer_input_size += self.cond_encode_dim
        return LSTMEncoder(
            input_size = self.infer_input_size,
            hidden_size = self.encoder_params.lstm_hidden_dim_enc,
            output_size = self.model_params.z_size * 2,  # for compute mu and sigma individually.
            num_layers = self.encoder_params.lstm_layers_enc
        )
    
    def _build_reconstruction_network(self):
        self.n_rollout_steps = self.decoder_params.n_rollout_steps
        self.rec_mse_weight = self.decoder_params.rec_mse_weight
        self.decoder_input_size = self.model_params.z_size
        self.cond_decode = self.decoder_params.cond_decode
        self.cond_decode_dim = self.decoder_params.cond_decode_dim
        if self.cond_decode:
            self.decoder_input_size += self.cond_decode_dim
        return LSTMDecoder(
            input_size=self.decoder_input_size,
            hidden_size=self.decoder_params.lstm_hidden_dim_dec,
            output_size=self.model_params.action_dim,
            num_layers=self.decoder_params.lstm_layers_dec
        )
        
    def _build_prior_network(self):
        self.prior_input_size = self.prior_params.prior_input_size
        return MultiLayerLinear(
            input_dim= self.prior_input_size,
            hidden_dim=self.prior_params.hidden_dim_pr,
            output_dim=self.model_params.z_size * 2,
            layers_num=self.prior_params.num_layers
        )
    
    def _build_beta_network(self):
        self.optimize_beta = self.beta_params.optimize_beta
        if self.optimize_beta:
            self._log_beta = TensorModule(torch.zeros(1, requires_grad=True, device=self.model_params.device))
            self._beta_opt = torch.optim.Adam(params=self._log_beta.parameters(), lr=self.beta_params.lr, betas=(0.9, 0.99))
            self._target_kl = self.beta_params.target_kl
            assert self._target_kl is not None
        else:
            self.kl_div_weight = self.beta_params.kl_div_weight
            assert self.kl_div_weight is not None

    def forward(self, inputs, use_learned_prior=False):
        """
        Args:
            inputs (_type_): _description_
            use_learned_prior (bool, optional): _description_. Defaults to False.
        """
        output = AttrDict()
        # Run inference
        output.q = self._run_inference(inputs)

        # Compute (fixed) prior
        output.p = self._get_fex_prior(output.q)

        # Infer learned skill prior
        output.q_hat = self._compute_learned_prior(self._learned_prior_input(inputs))
        if use_learned_prior:
             self.p = output.q_hat

        # Sample latent variable
        output.z = output.q.sample()
        output.z_q = output.z.clone()

        # decode
        output.reconstruction = self.decode(output.z, \
            cond_inputs = self._get_seq_dec(inputs), \
            steps = self.n_rollout_steps)
        
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        losses.rec_loss = self._compute_rec_loss(model_output, inputs, self.rec_mse_weight)

        # KL loss
        #TODO beta
        losses.kl_loss = self._compute_kl_loss(model_output, self.beta)

        # learned skill prior net loss
        losses.q_hat_loss = self._compute_prior_loss(model_output, 1)

        # update beta
        if self.optimize_beta:
            self._update_beta(losses.kl_loss.value)

        total_loss = self._compute_total_loss(losses=losses)

        return total_loss, losses


    def _run_inference(self, inputs: AttrDict):
        """Run inference with state sequence conditioning.

        Args:
            inputs (AttriDict): AttriDict with states and actions.
        """
        return MutivariateGuassion(self.q(action=inputs.actions, cond_input=self._get_seq_enc(inputs)))

    def _get_seq_enc(self, inputs: AttrDict):
        """Get condition inputs from inpput, overwrite it to use different condtion inputs.
        Make sure the condition input dim is same to 'self.cond_encode_dim'. 
        We use observation sequences as the condition encode inputs by default.

        Args:
            inputs (AttrDict): offline data with action, observations, rewards, etc.

        Returns:
            torch.Tensor/None: condition encode inputs.
        """
        if not self.cond_encode:
            return None
        return inputs.states[:, :-1]
    
    def _get_seq_dec(self, inputs:AttrDict):
        """Get sequence decoder condition inputs. We use the first observation by default.

        Args:
            inputs (AttrDict): offline data with action, observations, rewards, etc.
        """
        if not self.cond_decode:
            return None
        return inputs.states[:, 0]
    
    def _learned_prior_input(self, inputs):
        """Get the prior network inputs, overwrite it to use different prior inputs.
        By default, we the first observation of a const length sequence.

        Args:
            inputs (AttrDict): offine data with actions, obnservations, rewards, etc.

        Returns:
            Torch.Tensor: prior network inputs.
        """
        return inputs.states[:, 0]

    def _compute_learned_prior(self, inputs):
        """ Compute skill prior distribution.
        """
        return MutivariateGuassion(self.prior(inputs))

    def _regression_targets(self, inputs):
        return inputs.actions
    
    def _compute_rec_loss(self, model_output, inputs, weight):
        loss = AttrDict()
        loss.weight = weight
        distri = Gaussion(model_output.reconstruction, torch.zeros_like(model_output.reconstruction))
        loss.value = distri.nll(self._regression_targets(inputs)).mean()
        return loss

    def _compute_kl_loss(self, model_output, weight):
        loss = AttrDict()
        loss.weight = weight
        loss.value = model_output.q.kl_divergence(model_output.p).mean()
        return loss

    def _compute_prior_loss(self, model_output, weight):
        loss = AttrDict()
        loss.weight = weight
        loss.value = model_output.q_hat.nll(model_output.z_q.detach()).mean()
        # loss.value = model_output.q.detach().kl_divergence(model_output.q_hat).mean()
        return loss

    def _compute_total_loss(self, losses):
        total_loss = torch.stack([loss[1].value * loss[1].weight for loss in losses.items()])
        total_loss = total_loss.sum()
        return AttrDict(total_loss=total_loss)

    def _update_beta(self, kl_div):
        """Update beta use dual gradient descent.

        Args:
            kl_div (torch.Tensor): the current kl divergence.
        """
        bata_loss = self._log_beta().exp() * (self._target_kl - kl_div).detach().mean()
        self._beta_opt.zero_grad()
        bata_loss.backward()
        self._beta_opt.step()

    def decode(self, z, cond_inputs, steps):
        return self.decoder(z, cond_inputs, steps)
    
    
    def save_model(self, path):
        # inference network
        assert os.path.exists(path=path)
        path = WeightSaver(path).create_timestamp_folder()
        torch.save(self.q.state_dict(), os.path.join(path, "inference.pkl"))
        # decoder network
        torch.save(self.decoder.state_dict(), os.path.join(path, "decoder.pkl"))
        # prior network
        torch.save(self.prior.state_dict(), os.path.join(path, "prior.pkl"))
        if self.optimize_beta:
            torch.save(self._log_beta.state_dict(), os.path.join(path, "log_beta.pkl"))
        self.logger.info(f"Save model to {path}")

    def load_model(self, path):
        assert os.path.exists(path=path), f"path: {path} not exist."
        weights_list = os.listdir(path)
        assert "inference.pkl" in weights_list, "weight: inference.pkl not exist."
        assert "decoder.pkl" in weights_list, "weight: decoder.pkl not exist."
        assert "prior.pkl" in weights_list, "weight: prior.pkl not exist."
        self.q.load_state_dict(torch.load(os.path.join(path, "inference.pkl")))
        self.decoder.load_state_dict(torch.load(os.path.join(path, "decoder.pkl")))
        self.prior.load_state_dict(torch.load(os.path.join(path, "prior.pkl")))
        if self.optimize_beta:
            assert "log_beta.pkl" in weights_list, "weight: log_beta.pkl not exist."
            self._log_beta.load_state_dict(torch.load(os.path.join(path, "log_beta.pkl")))
        self.logger.info(f"Load weight from {path}")

    @contextmanager
    def val_mode(self):
        self._sample_prior  = True
        yield
        self._sample_prior = False
    
    @staticmethod
    def _get_fex_prior(tensor, bs=None, dim=None):
        """Create a standard guassion distribution. 

        Args:
            tensor (_type_): _description_
            bs (_type_, optional): _description_. Defaults to None.
            dim (_type_, optional): _description_. Defaults to None.

        Returns:
            Guassion: fixed prior distribution
        """
        if dim is not None:
            return Gaussion(tensor.new_zeros(bs, dim, 1, 1), tensor.new_zeros(bs, dim, 1, 1))
        else:
            return Gaussion(torch.zeros_like(tensor.mu), torch.zeros_like(tensor.log_sigma))
        
    @property
    def beta(self):
        """Get beta. If self.optimize_beta = false, we use const beta = self.kl_div_weight

        Returns:
            _type_: _description_
        """
        return self._log_beta().exp()[0].detach() if self.optimize_beta else self.kl_div_weight