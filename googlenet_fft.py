import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from __future__ import print_function
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

#FFTConv1d_thresh
from math import ceil
import itertools

global zero_counts
number_counting = 0
nonzero_counting = 0

class FFTConv1d_thresh(nn.Module):
	def __init__(self, conv, thresh=0.5):
		super(FFTConv1d_thresh, self).__init__()
		self.weight = conv.weight
		self.bias = conv.bias
		self.padding = conv.padding
		self.stride = conv.stride
		self.thresh = thresh

	def get_xfft_thresh(self, x):
		total, channel, x_row, x_col = x.size()
		x = x.view(total, channel, x_row*x_col)
		x_fft = torch.fft.fft(x)
		zero_idx = torch.nonzero((abs(x_fft.real) < self.thresh), as_tuple=True)
		#print(x_fft)
		x_fft[zero_idx] = 0
		#print(x_fft)
		#zero_counts += zero_idx

		return x_fft

	def get_Wfft(self, x, W):
		total, channel, x_row, x_col = x.size()
		out_channel, channel, W_row, W_col = W.size()

		W_pad = nn.ZeroPad2d((0, x_col-W_col, 0, x_row-W_row))(W)
		W_pad_re = W_pad.view(out_channel, channel, W_pad.size(dim=2)*W_pad.size(dim=3))
		W_fft = torch.fft.fft(
				torch.flip(
					torch.flip(W_pad_re, [0,1]), [0,1,2])
				)

		return W_fft

	def fftConv1D(self, x_fft, x_row, x_col, W_fft, W_row, W_col, channel):
		y = torch.fft.ifft(x_fft * W_fft).real.to(device)

		ans_row, ans_col = ceil((x_row - W_row + 1) / self.stride[0]), \
						ceil((x_col - W_col + 1) / self.stride[1])

		y = torch.roll(y, 1)
		y = y.view(channel, x_row, x_col)
		y = torch.sum(y, dim=0).to(device)
		y = y[::self.stride[0], ::self.stride[1]]
		y = y[:ans_row, :ans_col]

		return y

	def fftConv1D_channel(self, x, W, b):
		global zero_counts
		zero_counts = 0
		total, in_channel, x_row, x_col = x.size()
		out_channel, in_channel, W_row, W_col = W.size()
		ans_row, ans_col = ceil((x_row - W_row + 1) / self.stride[0]), \
						ceil((x_col - W_col + 1) / self.stride[1])
		ans = torch.zeros(total, out_channel, ans_row, ans_col).to(device)

		x_fft = self.get_xfft_thresh(x)
		#print(x_fft.numel())
		#print(int(torch.count_nonzero(x_fft)))
		W_fft = self.get_Wfft(x, W)
		tuple_total, tuple_out = tuple(range(total)), tuple(range(out_channel))
		for img, out_ch in itertools.product(tuple_total, tuple_out):
			ans[img,out_ch] = self.fftConv1D(x_fft[img], x_row, x_col,
											W_fft[out_ch], W_row, W_col,
											in_channel) + b[out_ch]

		return ans, x_fft.numel(), int(torch.count_nonzero(x_fft))

	def forward(self, x):
		with torch.no_grad():
			x = nn.ZeroPad2d((self.padding[0],self.padding[0],
				self.padding[1],self.padding[1]))(x)
			x, y, z = self.fftConv1D_channel(x, self.weight, self.bias)
		#print(f'[result] {x.size()}')

		return x


class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        # 1x1conv branch
        self.b1 = nn.Sequential(
            FFTConv1d_thresh(nn.Conv2d(input_channels, n1x1, kernel_size=1)),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )
        # 1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            FFTConv1d_thresh(nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1)),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            FFTConv1d_thresh(nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1)),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1conv -> 5x5conv branch
        # we use 2 3x3 conv filters stacked instead
        # of 1 5x5 filters to obtain the same receptive
        # field with fewer parameters
        self.b3 = nn.Sequential(
            FFTConv1d_thresh(nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1)),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            FFTConv1d_thresh(nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1)),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            FFTConv1d_thresh(nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3pooling -> 1x1conv
        # same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            FFTConv1d_thresh(nn.Conv2d(input_channels, pool_proj, kernel_size=1)),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            FFTConv1d_thresh(nn.Conv2d(3, 192, kernel_size=3, padding=1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)

        output = self.maxpool(output)

        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)

        output = self.maxpool(output)

        output = self.a5(output)
        output = self.b5(output)

        output = self.avgpool(output)
        output = self.dropout(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)

        return output

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    top5count = 0
    counts = int(0)
    i = 0

    with torch.no_grad():
        for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model.forward(data)

          v, result = output.topk(5, 1, True, True)
          top5count += torch.eq(result, target.view(-1, 1)
                                ).sum().int().item()

          # sum up batch loss
          test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
          # get the index of the max log-probability
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()
          counts += 1
          i += 10
          print("test case: ", counts)
          #change the number of the testcase
          if counts == 1:
            break
          

    #test_loss /= len(test_loader.dataset)
    test_loss /= i

    print('\nTest set: Average loss: {:.4f}, Top 1 Error: {}/{} ({:.2f}), Top 5 Error: {}/{} ({:.2f})\n'.format(
        test_loss,
        i - correct, i,
        1 - correct / i,
        i - top5count, i,
        1 - top5count / i,
    ))

    """print('\nTest set: Average loss: {:.4f}, Top 1 Error: {}/{} ({:.2f}), Top 5 Error: {}/{} ({:.2f})\n'.format(
        test_loss,
        len(test_loader.dataset) - correct, len(test_loader.dataset),
        1 - correct / len(test_loader.dataset),
        len(test_loader.dataset) - top5count, len(test_loader.dataset),
        1 - top5count / len(test_loader.dataset),
    ))"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PyTorch GoogleNet Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default:   64)')
  parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10)')
  parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
  parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
  parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
  #args = parser.parse_args()
  args = parser.parse_args(args=[])
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  print(use_cuda)
  device = torch.device("cuda:0")
  print(f'[device] {device}')

  path = './googlenet_cnn.pt'

  transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  test_kwargs = {'batch_size': args.test_batch_size}

  testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
  
  net = GoogleNet()
  model = net.to(device)
  model.load_state_dict(torch.load(path))

  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

  test(model, device, testloader)
