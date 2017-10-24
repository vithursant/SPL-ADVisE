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

		# Inherited the parent class initialization method, that is, first
		# run nn.Module initialization function
		super(ShallowNet, self).__init__()
		#self.embed = nn.Embedding(64, 2)
		# Define convolution layer: input 1 channel (grayscale) picture,
		# output 6 feature map, convolution kernel 5x5
		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) # input is 28x28. padding=2 for same padding
		# Define convolution layer: enter 6 feature maps, output 16 feature
		# maps, convolution kernel 5x5
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # feature map size is 14*14 by pooling
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 16 * 5 * 5 nodes connected to 120 nodes
		self.fc1 = nn.Linear(64*7*7, 10)
		self.emb = nn.Linear(64*7*7, self.emb_dim)
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 120 nodes connected to 84 nodes
		#self.fc2 = nn.Linear(120, 84)
		# Define the fully connected layer: linear connection (y = Wx + b),
		# 84 nodes connected to 10 nodes
		#self.fc3 = nn.Linear(84, 10)

		#self.embedding = nn.Linear(2, self.emb_dim)

		#self.norm_emb =  L2_Normalize(self.emb)

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

		# Input x -> conv1 -> relu -> 2x2 the largest pool of windows ->
		# update to x
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		self.layer1 = x

		# Input x -> conv2 -> relu -> 2x2 window maximum pooling -> update
		# to x
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		self.layer2 = x

		# The view function changes the tens x into a one-dimensional vector
		# form with the total number of features unchanged, ready for the
		# fully connected layer
		x = x.view(-1, self.num_flat_features(x))
		#x = x.view(-1, self.num_flat_features(x))
		self.features = x
		#self.features = features
		#features.register_hook(print)

		# Input x -> fc1 -> relu, update to x
		#x = F.relu(self.fc1(x))
		out1 = self.emb(x)
		#fc1 = x
		embeddings = out1

		out2 = self.fc1(x)
		prediction = F.softmax(out2)
		#pdb.set_trace()
		# Input x -> fc2 -> relu, update to x
		#x = F.relu(self.fc2(x))
		#fc2 = x

		# Input x -> fc3 -> relu, update to x
		#x = self.fc3(x)
		#fc3 = x

		#pdb.set_trace()

		#x = self.embedding(x)
		#pdb.set_trace()
		#x.register_hook(print)
		#x = self.norm_emb(x)

		#norm_emb = x
		norm_embeddings = self.l2_normalize(out1, 1)
		#pdb.set_trace()
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