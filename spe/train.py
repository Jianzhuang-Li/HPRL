from data.eplus.eplus_dataset import EplusSpirlSplitDataset
from spe.general_utils import AttrDict
from spe.skpl import SkillPriorMdl
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torch
import argparse
import logging

logging.basicConfig(level = logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='train SkillPriorMdl')
parser.add_argument('--log_dir', type=str, default="./logs", help="tensorboard log dir.")
parser.add_argument('--epoch', type=int, default=10, help="train epoch number.")
parser.add_argument('--log_interval', type=int, default=100, help="log interval.")
parser.add_argument('--weight_save_dir', type=str, default="./weights")
parser.add_argument('--batch_size', type=int, default=128, help="batch size.")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate.")
parser.add_argument('--train', type=bool, default=True, help="if train epoch")
parser.add_argument('--load_weight', type=str, default=None, help="load weight from this dir")
args = parser.parse_args()


def train_epoch(model, data_loader, total_epoch, writer, optimizer, log_out):
    step = 0
    for epoch in range(total_epoch):
        with tqdm(total=len(data_loader)) as _tqdm:
            _tqdm.set_description("epoch: {}/{}".format(epoch+1, total_epoch))
            for i, batch_data in enumerate(data_loader):
                # update tqdm
                _tqdm.update(1)
                # forward
                output = model.forward(batch_data)
                loss, loss_raw = model.loss(output, batch_data)
                # log output
                step += 1
                if step % log_out == 0:
                    writer.add_scalar("Loss/rec_loss", loss_raw.rec_loss.value.detach().cpu(), step)
                    writer.add_scalar("Loss/kl_loss", loss_raw.kl_loss.value.detach().cpu(), step)
                    writer.add_scalar("Loss/q_hat_loss", loss_raw.q_hat_loss.value.detach().cpu(), step)
                optimizer.zero_grad()
                # 反向传播
                loss.total_loss.backward()
                # 更新梯度
                optimizer.step()
    model.save_model(args.weight_save_dir)

writer = SummaryWriter(args.log_dir)

data_conf = AttrDict()
data_conf.dataset_spec = AttrDict(subseq_len = 10)
data_conf.device = "cuda"
dataset = EplusSpirlSplitDataset("", data_conf=data_conf, phase="val")
batch_size = args.batch_size
data_loader = dataset.get_data_loader(batch_size, 50)

env = AttrDict()
env.action_dim = dataset[0]["actions"][0].shape[0]
env.state_dim = dataset[0]["states"][0].shape[0]

model = SkillPriorMdl(env=env)

if args.load_weight is not None:
    model.load_model(args.load_weight)

learning_rate = args.learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

log_out = args.log_interval
total_epoch = args.epoch

if args.train:
    train_epoch(model, data_loader, total_epoch, writer, optimizer, log_out)
# print(dataset[0])
# input = AttrDict()
# input.states = torch.from_numpy(dataset[0]["states"])
# input.actions = torch.from_numpy(dataset[0]["actions"])