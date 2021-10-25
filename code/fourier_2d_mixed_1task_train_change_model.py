import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import savemat
from torch.nn.parameter import Parameter
import os
from os.path import exists
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d_Q_1layer(nn.Module):
    def __init__(self, modes1, modes2, width, task_num):
        super(FNO2d_Q_1layer, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.task_num = task_num
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.ModuleList([nn.Linear(128, 1) for i in range(task_num + 1)])

    def forward(self, x, task_idx):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2[task_idx](x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, task_num):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.task_num = task_num
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.ModuleList([nn.Linear(self.width, 128) for i in range(task_num+1)])
        self.fc2 = nn.ModuleList([nn.Linear(128, 1) for i in range(task_num+1)])

    def forward(self, x, task_idx):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1[task_idx](x)
        x = F.gelu(x)
        x = self.fc2[task_idx](x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def read_train_data(input_dir,ntrain):
    count = 0
    x_train = []
    y_train = []
    r = 5
    h = int(((421 - 1) / r) + 1)
    s = h
    for filename in os.listdir(input_dir):
        if filename.endswith(".mat") and 'f_3' not in filename :
            FILE_PATH = os.path.join(input_dir, filename)
            reader = MatReader(FILE_PATH)
            x_train.append(reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s])
            y_train.append(reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s])
            print("finished " + filename)
    x_train_mixed = torch.cat([item for item in x_train], 0)
    y_train_mixed = torch.cat([item for item in y_train], 0)
    return x_train_mixed, y_train_mixed


################################################################
# configs
################################################################
# input
TEST_PATH = '../data/Darcy/Meta_data_f_test/output3_12_train_1000_change_f_3.mat'
#model name
train_ratio = "f"
test_ratio = "3_12"

model_name = train_ratio+'1_layer_train_task1-10_1000_test_3_800_with_norm_train_model'
train_dir = '../data/Darcy/Meta_data_f'
ntrain_pertask = 1000
x_train, y_train = read_train_data(train_dir, ntrain_pertask)


RESULT_PATH = '../results/train_' + train_ratio + '_test_' + test_ratio + '/subtask_'+train_ratio+'/' + model_name + '.mat'
MODEL_PATH = '../models/train_' + train_ratio + '_test_' + test_ratio + '/' + model_name

ntest = 200
ntest_pretrain =800
task_num=9
ntrain = task_num*ntrain_pertask
# flag variable to indicate whether need to train B first
train_flag = False
batch_size = 20
learning_rate = 0.003

epochs = 500
epochs_test = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 5
h = int(((421 - 1) / r) + 1)
s = h

################################################################
# load data and data normalization
################################################################
reader = MatReader(TEST_PATH)
#x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
#y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

reader.load_file(TEST_PATH)
# x_test_pretrain = reader.read_field('coeff')[ntrain_pertask:ntrain_pertask+ntest_pretrain, ::r, ::r][:, :s, :s]
# y_test_pretrain = reader.read_field('sol')[ntrain_pertask:ntrain_pertask+ntest_pretrain, ::r, ::r][:, :s, :s]
# x_test = reader.read_field('coeff')[ntrain_pertask+ntest_pretrain:, ::r, ::r][:, :s, :s]
# y_test = reader.read_field('sol')[ntrain_pertask+ntest_pretrain:, ::r, ::r][:, :s, :s]
# print(x_train.shape)
x_test_pretrain = reader.read_field('coeff')[:800, ::r, ::r][:, :s, :s]
y_test_pretrain = reader.read_field('sol')[:800, ::r, ::r][:, :s, :s]
x_test = reader.read_field('coeff')[800:, ::r, ::r][:, :s, :s]
y_test = reader.read_field('sol')[800:, ::r, ::r][:, :s, :s]
print(x_train.shape)
print(x_test.shape)
print(x_test_pretrain.shape)

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
x_test_pretrain = x_normalizer.encode(x_test_pretrain)
print(x_test_pretrain.shape)
#
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_test_pretrain = y_normalizer.encode(y_test_pretrain)
y_normalizer.cuda()

x_train = x_train.reshape(ntrain, s, s, 1)
x_test = x_test.reshape(ntest, s, s, 1)
print(ntest_pretrain)
x_test_pretrain = x_test_pretrain.reshape(ntest_pretrain,s , s, 1)
train_loader = []
for t in range(task_num):
    train_loader.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:], y_train[t*ntrain_pertask:(t+1)*ntrain_pertask,:]), batch_size=batch_size,
                                           shuffle=True))

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_pretrain_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_pretrain, y_test_pretrain), batch_size=batch_size,
                                          shuffle=True)

################################################################
# training and evaluation
################################################################
#model = FNO2d(modes, modes, width, task_num).cuda()
model = FNO2d_Q_1layer(modes, modes, width, task_num).cuda()
#torch.save(model.state_dict(), MODEL_PATH)
print(count_params(model))
if (exists(MODEL_PATH)):
    print("load model")
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    train_flag = True
model.cuda()


myloss = LpLoss(size_average=False)

if(train_flag ==True):
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# inner loop update the last layer for each task
# outer loop update the representation of all tasks
# optimizer_test =  Adam([{'params': model.fc2[task_num].parameters()}], lr=learning_rate, weight_decay=1e-4)
optimizer_test =  Adam([{'params': model.fc2[task_num].parameters()}], lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler_test =  torch.optim.lr_scheduler.StepLR(optimizer_test, step_size=step_size, gamma=gamma)
#
if(train_flag == True):
    for ep in range(epochs):
        print("training model ")
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for i in range(int(ntrain_pertask/20)):
            losses =[]
            optimizer.zero_grad()
            for task_idx in range(task_num):
                x, y = next(iter(train_loader[task_idx]))
                x, y = x.cuda(), y.cuda()
                out = model(x, task_idx).reshape(batch_size, s, s)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                losses.append(myloss(out.view(batch_size, -1), y.view(batch_size, -1)))
            sum(losses).backward()
            optimizer.step()
            train_l2 += sum(losses).item()
        scheduler.step()

        test_l2 = 0
        if(ep %10 ==0 ):
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    out = model(x,task_num).reshape(batch_size, s, s)
                    out = y_normalizer.decode(out)
                    test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

            train_l2 /= ntrain
            test_l2 /= ntest
            t2 = default_timer()
            print(ep, t2 - t1, train_l2,test_l2)

    # save the training model
    print("save the trained model now")
    torch.save(model.state_dict(), MODEL_PATH)


test_num = ntest_pretrain
for ep in range(epochs_test):
    # meta test train last layer
    print("meta test train here ")
    model.train()
    test_pretrain_l2 = 0
    t1 = default_timer()
    for  x,y  in test_pretrain_loader:
        optimizer_test.zero_grad()
        x,y = x.cuda(),y.cuda()
        out = model(x,task_num).reshape(batch_size, s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        test_loss = myloss(out.view(batch_size,-1),y.view(batch_size,-1))
        test_pretrain_l2 += test_loss.item()
        test_loss.backward()
        optimizer_test.step()
    scheduler_test.step()


    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x,task_num).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()


    test_l2 /= ntest
    test_pretrain_l2/=test_num
    t2 = default_timer()
    print("epoch num ",ep,"time/epoch " ,t2 - t1, "meta test loss ", test_pretrain_l2,"test loss ",test_l2)

torch.save(model.state_dict(), MODEL_PATH+"_pretrain")

savemat(RESULT_PATH,
        {"sol_learn": out.detach().cpu().numpy(), 'sol_ground': y.view(batch_size, s, s).detach().cpu().numpy()})

