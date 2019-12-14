import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import time
import os
from utils import select_network, select_optimizer, str2bool
from datetime import datetime

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type',
                    type=str, default='nnRNN',
                    choices=['LSTM', 'RNN', 'expRNN', 'nnRNN'],
                    help='options: LSTM, RNN, expRNN, nnRNN')
parser.add_argument('--nhid',
                    type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--cuda', action='store_true',
                    default=False, help='use cuda')
parser.add_argument('--T', type=int, default=200,
                    help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int,
                    default=400, help='random seed')
parser.add_argument('--labels', type=int, default=8,
                    help='number of labels in the output and input')
parser.add_argument('--c-length', type=int, default=20, help='sequence length')
parser.add_argument('--no-of-ones', type=int, default=2, help='Number of ones in sequence')
parser.add_argument('--onehot', action='store_true',
                    default=False, help='Onehot inputs')

parser.add_argument('--batch', type=int,
                    default=1, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument('--optimizer',type=str, default='RMSprop',
                    choices=['Adam', 'RMSprop'],
                    help='optimizer: choices Adam and RMSprop')
parser.add_argument('--alpha',type=float,
                    default=0.99, help='alpha value for RMSprop')
parser.add_argument('--betas',type=tuple,
                    default=(0.9, 0.999), help='beta values for Adam')

parser.add_argument('--rinit', type=str, default="cayley",
                    choices=['random', 'cayley', 'henaff', 'xavier'],
                    help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier",
                    choices=['xavier', 'kaiming'],
                    help='input weight matrix initialization' )
parser.add_argument('--nonlin', type=str, default='modrelu',
                    choices=['none','modrelu', 'tanh', 'relu', 'sigmoid'],
                    help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--alam', type=float, default=0.0001,
                    help='decay for gamma values nnRNN')
parser.add_argument('--Tdecay', type=float,
                    default=10e-6, help='weight decay on upper T')
parser.add_argument('--gamma-zero-gradient', type=str2bool,
                    default=False, help='Whether to update gamma values or not')

args = parser.parse_args()

def adding_problem_generator(N, seq_len=6, high=1, number_of_ones=2):
    """ A data generator for adding problem.

    The data definition strictly follows Quoc V. Le, Navdeep Jaitly, Geoffrey E.
    Hintan's paper, A Simple Way to Initialize Recurrent Networks of Rectified
    Linear Units.

    The single datum entry is a 2D vector with two rows with same length.
    The first row is a list of random data; the second row is a list of binary
    mask with all ones, except two positions sampled by uniform distribution.
    The corresponding label entry is the sum of the masked data. For
    example:

     input          label
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    :param N: the number of the entries.
    :param seq_len: the length of a single sequence.
    :param p: the probability of 1 in generated mask
    :param high: the random data is sampled from a [0, high] uniform distribution.
    :return: (X, Y), X the data, Y the label.
    """
    # print(seq_len)
    X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    # Y = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in range(N):
        # Default uniform distribution on position sampling
        positions = np.random.choice(seq_len, size=number_of_ones, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
        # Y[i,-1, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return torch.FloatTensor(X), torch.FloatTensor(Y)
# X, y = adding_problem_generator(10)
# print(X.size(), y.size())
# print(X)
# print(y)

def generate_copying_sequence(T, labels, c_length):

    items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
    x = []
    y = []

    ind = np.random.randint(labels, size=c_length)
    for i in range(c_length):
        x.append([items[ind[i]]])
    for i in range(T - 1):
        x.append([items[8]])
    x.append([items[9]])
    for i in range(c_length):
        x.append([items[8]])

    for i in range(T + c_length):
        y.append([items[8]])
    for i in range(c_length):
        y.append([items[ind[i]]])

    x = np.array(x)
    y = np.array(y)

    return torch.FloatTensor([x]), torch.LongTensor([y])

def create_dataset(size, T, c_length=10):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_copying_sequence(T, 8, c_length)
        sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y

def onehot(inp):
    onehot_x = inp.new_zeros(inp.shape[0], args.labels+2)
    return onehot_x.scatter_(1, inp.long(), 1)

class Model(nn.Module):
    def __init__(self, hidden_size, rec_net):
        super(Model, self).__init__()
        self.rnn = rec_net
        self.lin = nn.Linear(hidden_size, 1) # should be 1 because the output is of size 1
        self.hidden_size = hidden_size
        self.loss_func = nn.MSELoss()

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x, y):
        # print("y: ", y.size(), y[0], x[0])
        # print("onehot", args.onehot)
        # print("x: ", x.size(), "y: ", y.size())
        # print("x batch size: ", x.shape[1])
        hidden = None
        loss = 0
        accuracy = 0
        if NET_TYPE == 'LSTM':
            self.rnn.init_states(x.shape[1])
        for i in range(len(x)):
            if args.onehot:
                inp = onehot(x[i])
                hidden = self.rnn.forward(inp, hidden)
            else:
                hidden = self.rnn.forward(x[i], hidden)
            out = self.lin(hidden)
            
        # print(y.size())
        
            if i >= T + args.c_length:
                preds = torch.argmax(out, dim=1)
                actual = y[i].squeeze(1)
                correct = preds == actual
                accuracy += correct.sum().item()
        # print("out size and y[i]: ", out.size(), y.squeeze(1).t())
        # print(out.size())
        # loss += self.loss_func(out-out+1, y.squeeze(1).t())
        loss += self.loss_func(out, y.squeeze(1).t())
        accuracy /= (args.c_length*x.shape[1])
        # loss /= (x.shape[0])
        # print("loss: ", loss)
        return loss, accuracy

def train_model(net, optimizer, batch_size, T, n_steps):

    accs = []
    losses = []
    rec_nets = []
    first_hid_grads = []
    
    for i in range(n_steps):
        # print(i)
        # if i==0 or i==n_steps-1:
        #     # print("P before: ", torch.mul(net.rnn.UppT, net.rnn.M))
        #     print("Gamma before: ", net.rnn.alphas)
        s_t = time.time()
        # x,y = create_dataset(batch_size, T, args.c_length)
        x,y = adding_problem_generator(batch_size, seq_len=args.c_length, number_of_ones=args.no_of_ones)
        # print(x.size(), y.size())
        # print(x, y)
        if CUDA:
            x = x.cuda()
            y = y.cuda()
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        
        optimizer.zero_grad()
        if orthog_optimizer:
            orthog_optimizer.zero_grad()
        loss, accuracy = net.forward(x, y)
        loss_act = loss
        if NET_TYPE == 'nnRNN' and alam > 0:
            alpha_loss = net.rnn.alpha_loss(alam)
            loss += alpha_loss
        loss.backward()
        # if i==0 or i==n_steps-1:
        #     # print("P after: ", torch.mul(net.rnn.UppT, net.rnn.M))
        #     print("Gamma after: ", net.rnn.alphas)
        losses.append(loss_act.item())

        if orthog_optimizer:
            net.rnn.orthogonal_step(orthog_optimizer)
            if args.gamma_zero_gradient == True:
                [net.rnn.alphas[i].grad.data.zero_() for i in range(len(net.rnn.alphas))]
                # [net.rnn.thetas[i].grad.data.zero_() for i in range(len(net.rnn.thetas))]
            # print(net.rnn.thetas[0].grad)

        optimizer.step()
        accs.append(accuracy)

        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'
              .format(i +1, time.time()- s_t, loss_act.item(), accuracy))
    
    print("Average loss: ", np.mean(np.array(losses)))
    theta_list = net.rnn.get_theta_list()
    print(theta_list)
    with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(losses, fp)

    with open(SAVEDIR + '{}_Train_Accuracy'.format(NET_TYPE),'wb') as fp:
        pickle.dump(accs, fp)

    with open(SAVEDIR + '{}_Rec_Nets'.format(NET_TYPE),'wb') as fp:
        pickle.dump(rec_nets, fp)

    with open(SAVEDIR + '{}_First_Hid_Grads'.format(NET_TYPE),'wb') as fp:
        pickle.dump(first_hid_grads, fp)

    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'time step': i
        },
        '{}_{}.pth.tar'.format(NET_TYPE, i)
        )
        
    return

def save_checkpoint(state, fname):
    filename = SAVEDIR + fname
    torch.save(state, filename)

nonlin = args.nonlin.lower()
random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
alam = args.alam
Tdecay = args.Tdecay
hidden_size = args.nhid
n_steps = 1
exp_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
SAVEDIR = os.path.join('./saves', 'adding-problem',
                       NET_TYPE, str(random_seed),exp_time)

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

inp_size = 2
T = args.T
batch_size = args.batch
out_size = args.labels + 1
if args.onehot:
    inp_size = args.labels + 2

rnn = select_network(args, inp_size)
net = Model(hidden_size, rnn)
if CUDA:
    net = net.cuda()
    net.rnn = net.rnn.cuda()

print('Copy task')
print(NET_TYPE)
print('Cuda: {}'.format(CUDA))
print(nonlin)
print(hidden_size)
# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

orthog_optimizer = None
optimizer, orthog_optimizer = select_optimizer(net, args)

with open(SAVEDIR + 'hparams.txt','w') as fp:
    for key, val in args.__dict__.items():
        fp.write(('{}: {}'.format(key,val)))
train_model(net, optimizer, batch_size, T, n_steps)
