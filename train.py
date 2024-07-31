import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import time
import io, os, sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import argparse


fc1 = 64
fc2 = 128
fc3 = 256
fc4 = 128
fc5 = 64


parser = argparse.ArgumentParser(description='Theia_Training_v1')
parser.add_argument('--batch_size',     type=int,   default=8000,
                    help='input batch size for training (default: )')
parser.add_argument('--epochs',         type=int,   default=10000,
                    help='number of epochs to train, save the best model instead of the model at last epoch (default: )')
parser.add_argument('--lr',             type=float, default=0.0015,
                    help='learning rate (default: )')
parser.add_argument('--seed',           type=int,   default=42,
                    help='random number seed (default: )')
parser.add_argument('--device',         default='cpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')

args = parser.parse_args()
args.cuda = ( args.device.find("gpu")!=-1 and torch.cuda.is_available() )

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

file_path = r"/lus/eagle/projects/RECUP/twang/physics-theia/systematic_variation.dat"

#================Loading data, normalize, and create dataloader================#

data = []
with open(file_path, 'r') as f:
    for line in f:
        a, b, val = map(float, line.split())
        data.append((a, b, val))
data = torch.tensor(data, dtype=torch.float32)
print(data.shape)

mean = data.mean(dim=0)
std = data.std(dim=0)
print("mean = ", mean)
print("std = ", std)
data = (data - mean) / std

X = data[:,0:2]
y = data[:,2]
dataset = TensorDataset(X, y)

train_ratio = 0.8
test_ratio = 0.2
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

#================Defining model, optimizer, and loss================#

class TheiaModel(torch.nn.Module):
    def __init__(self, fc1, fc2, fc3, fc4, fc5):
        super(TheiaModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, fc1) 
        self.fc2 = torch.nn.Linear(fc1, fc2) 
        self.fc3 = torch.nn.Linear(fc2, fc3) 
        self.fc4 = torch.nn.Linear(fc3, fc4)
        self.fc5 = torch.nn.Linear(fc4, fc5)
        self.fc6 = torch.nn.Linear(fc5, 1)
        self.relu = torch.nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)    
        x = self.relu(x)
        x = self.fc6(x)    
        return x

model = TheiaModel(fc1=fc1, fc2=fc2, fc3=fc3, fc4=fc4, fc5=fc5)
if args.cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()

#================Defining the training pipeline================#

def train(epoch,
          model,
          optimizer,
          train_loader,
          criterion,
          on_gpu,
          ):

    model.train()

    train_loss = torch.tensor(0.0)
    if on_gpu:
        train_loss = train_loss.cuda()

    for batch_idx, current_batch in enumerate(train_loader):
        if on_gpu:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0], current_batch[1]

        optimizer.zero_grad()
        output = model(inp)
        gndtruth = current_batch_y

        loss = criterion(output.squeeze(), gndtruth)
        loss.backward()

#        max_grad = max(param.grad.abs().max() for param in model.parameters() if param.grad is not None)
#        print("TW: param grad: |grad|_max = {:15.8f}".format(max_grad))

        optimizer.step()
        train_loss  += loss.item()

    train_loss  = train_loss  / len(train_loader)
    return train_loss
#    print("epoch: {}, Average Train loss: {:15.8f}".format(epoch, train_loss))

def test(epoch,
         model,
         test_loader,
         criterion,
         on_gpu):

    model.eval()

    test_loss = torch.tensor(0.0)
    if on_gpu:
        test_loss = test_loss.cuda()

    with torch.no_grad():
        for batch_idx, current_batch in enumerate(test_loader):
            if on_gpu:
                inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
            else:
                inp, current_batch_y = current_batch[0], current_batch[1]

            output = model(inp)
            loss = criterion(output.squeeze(), current_batch_y)
            test_loss += loss.item()

    test_loss  = test_loss  / len(test_loader)
    return test_loss
#    print("epoch: {}, Average Test loss: {:15.8f}".format(epoch, test_loss))


best_loss = 99999999.9
best_epoch = -1
time_real_train = time.time()

for epoch in range(0, args.epochs):
    train_loss = train(epoch, model, optimizer, train_loader, criterion, args.cuda)
    test_loss = test(epoch, model, test_loader, criterion, args.cuda)
    print("epoch: {}, Average Train loss: {:15.8f}, Test loss: {:15.8f}".format(epoch, train_loss, test_loss))
    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch
        checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
        print("Better epoch at {}".format(best_epoch))

torch.save(checkpoint, 'ckpt.pth')
print(f"Real best test loss without normalizaing to zero = {best_loss * std[-1] * std[-1]}")

#================Load the model and optimize for the best input================#

model = TheiaModel(fc1=fc1, fc2=fc2, fc3=fc3, fc4=fc4, fc5=fc5)
if args.cuda:
    model = model.cuda()
checkpoint = torch.load('ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])

def loss_function(x):
    x = x.unsqueeze(0)
    y_pred = model(x)
    loss = torch.mean(y_pred)
    return loss

x0 = torch.rand((1, 2), requires_grad=True)
if args.cuda:
    x0 = x0.cuda()
optimizer = torch.optim.Adam([x0], lr=0.003)

num_iter = 3000


#####################
#print(X.shape)
#print(y.shape)
#
#y_pred = model(X)
#print(y_pred.shape)
#diff = y_pred.squeeze() - y
#print(diff.shape)
#
#X_np = X.numpy()
##y_np = diff.detach().numpy()
#y_pred = y_pred * 2521.2349 + 3211.8845
#y_np = y_pred.detach().numpy()
#
#plt.figure(figsize=(10, 8))
#scatter = plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='YlGnBu', alpha=0.7)
#plt.colorbar(scatter, label='y')
#plt.xlabel('X[:, 0]')
#plt.ylabel('X[:, 1]')
#plt.title('Color plot of y vs X')
##plt.savefig('diff_plot.png')
#plt.savefig('pred_plot.png')
####################



for it in range(num_iter):
    optimizer.zero_grad()
    loss = loss_function(x0)
    loss.backward()
    optimizer.step()
    print(f'Iteration {it}: Loss = {loss.item()}')

# The optimal input
x_optimal = x0.detach().cpu().numpy()
print("Optimal Input:", x_optimal)

