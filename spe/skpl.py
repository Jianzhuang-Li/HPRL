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
        for _ in range(hidden_dim):
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

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # forward lstm
        out, (hn, cn) = self.lstm(x, (h0, c0))
        last_out = hn[-1,:,:]

        # Decode the hidden state of the last time step
        encoded = self.fc(last_out)
        return encoded
    
class LSTMDecoderWithInitalizer(LSTMEncoder):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 ):
        super().__init__(input_size, hidden_size, output_size, num_layers)
        # self.input_initalizer = input_initalizer
        # self.hidden_initalizer = hidden_initalizer

    def forward(self, z, cond_input, seq_length):
         # Initialize hidden state and cell state with z
        # h0 = self.hidden_initalizer(cond_input)
        # c0 = self.input_initalizer(cond_input)
        decode_input = torch.cat((cond_input, z), -1)
        x = decode_input.unsqueeze(1).repeat(1, seq_length, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        recon_x = self.fc(out)
        # decoder_input = z 
        # # Forward propagate LSTM
        # # 
        # for _ in range(seq_length):
        #     output, (h0, c0) = self.lstm(decoder_input, (h0, c0))
        #     # Decode the hidden state of the last time step
        #     out = self.fc(output)
        #     outputs.append(output)
        #     decoder_input = output
        # output_sequence = torch.cat(outputs, dim=1)
        return recon_x


class SkillPriorMdl(torch.nn.Module):
    """Skill embeding + prior model for HPRL algorithm.
    """
    def __init__(self, env):
        super(SkillPriorMdl, self).__init__()
        self.logger = logging.getLogger(__name__)
        # var
        self.log_out = 100
        self.count = 0
        self.device = "cuda"
        # TODO：beta
        self.beta = 0.0005
        self._sample_prior = False
        self.env = env
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        self.rec_mse_weight = settings.SkillPrior.rec_mse_weight
        self.n_rollout_steps = settings.SkillPrior.n_rollout_steps
        # build inference network
        infer_input_size = self.action_dim
        if settings.SkillPrior.cond_decode:
            infer_input_size += settings.SkillPrior.prior_input_dim
        self.q = LSTMEncoder(input_size=infer_input_size,
                             hidden_size=settings.SkillPrior.q_lstm_hidden_dim,
                             output_size=settings.SkillPrior.q_embed_dim * 2, # 这里*2为了分别计算mu和sigma
                             num_layers=settings.SkillPrior.q_lstm_layers
                             )
        # build decoder network
        # self.decoder_input_initalizer = MultiLayerLinear(input_dim=settings.SkillPrior.prior_input_dim,
        #                                                  hidden_dim=settings.SkillPrior.prior_hidden_dim,
        #                                                  output_dim=self.action_dim,
        #                                                  layers_num=settings.SkillPrior.prior_net_layers)
        
        # hidden_initalizer_dim = settings.SkillPrior.q_lstm_layers * settings.SkillPrior.q_lstm_hidden_dim
        # self.decoder_hidden_initalizer = MultiLayerLinear(input_dim=settings.SkillPrior.prior_input_dim,
        #                                                   hidden_dim=settings.SkillPrior.prior_hidden_dim,
        #                                                   layers_num=settings.SkillPrior.prior_net_layers,
        #                                                   output_dim=hidden_initalizer_dim)

        self.decoder = LSTMDecoderWithInitalizer(input_size=self.state_dim + settings.SkillPrior.q_embed_dim,
                                                 hidden_size=settings.SkillPrior.nz_mid_lstm,
                                                 num_layers=settings.SkillPrior.q_lstm_layers,
                                                 output_size=self.action_dim,)
        # build prior network
        # TODO: Change to MultiLayerLiner
        # self.prior = MultiGussionDistri(ladent_dim=settings.SkillPrior.q_embed_dim,
        #                                 input_dim=settings.SkillPrior.prior_input_dim,
        #                                 hidden_dim=settings.SkillPrior.prior_hidden_dim,
        #                                 output_dim=settings.SkillPrior.prior_hidden_dim,
        #                                 layers_num=settings.SkillPrior.num_prior_net_layers)
        self.prior = MultiLayerLinear(input_dim=settings.SkillPrior.prior_input_dim,
                                      hidden_dim=settings.SkillPrior.prior_hidden_dim,
                                      output_dim=settings.SkillPrior.q_embed_dim * 2,
                                      layers_num=settings.SkillPrior.prior_net_layers)
        
        # optionally: optimize beta with dual gradient descent.
        self.optimize_beta = settings.SkillPrior.optimize_beta
        if self.optimize_beta:
            self._log_beta = TensorModule(torch.zeros(1, requires_grad=True, device=self.device))
            self._beta_opt = torch.optim.Adam(params=self._log_beta.parameters(), lr=3e-4, betas=(0.9, 0.999))
            self._target_kl = settings.SkillPrior.target_kl
    
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
            cond_inputs = self._learned_prior_input(inputs), \
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
        infer_inputs = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        return MutivariateGuassion(self.q(infer_inputs))

    def _get_seq_enc(self, inputs: AttrDict):
        return inputs.states[:, :-1]
    
    def _learned_prior_input(self, inputs):
        return inputs.states[:, 0]

    def _compute_learned_prior(self, inputs):
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