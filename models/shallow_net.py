import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Build Network
class ShallowNet(nn.Module):

	def __init__(self, emb_dim):
		self.emb_dim = emb_dim

		'''
		Define the initialization function of LeNet, this function defines
		the basic structure of the neural network
		'''
		super(ShallowNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
		self.fc1 = nn.Linear(64*7*7, 10)
		self.emb = nn.Linear(64*7*7, self.emb_dim)

		self.layer1 = None
		self.layer2 = None
		self.features = None
		self.embeddings = None
		self.norm_embeddings = None

	def forward(self, x):
		'''
		Define the forward propagation function and automatically generates
		the backward propagation function (autograd)
		'''
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		self.layer1 = x
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		self.layer2 = x
		x = x.view(-1, self.num_flat_features(x))
		self.features = x
		out1 = self.emb(x)
		embeddings = out1
		out2 = self.fc1(x)
		norm_embeddings = self.l2_normalize(out1, 1)

		return embeddings, norm_embeddings

	def num_flat_features(self, x):
		'''
			Calculate the total tensor x feature amount
		'''

		size = x.size()[1:] # All dimensions except batch dimension
		num_features = 1
		for s in size:
			num_features *= s

		return num_features

	def l2_normalize(self, x, dim):

	    if not (isinstance(x, torch.DoubleTensor) or isinstance(x, torch.FloatTensor)):
	        x = x.float()

	    if len(x.size()) == 1:
	        x = x.view(1, -1)

	    norm = torch.sqrt(torch.sum(x * x, dim=dim))
	    norm = norm.view(-1, 1)

	    return torch.div(x, norm)

	def name(self):
		return 'lenet-magnet'