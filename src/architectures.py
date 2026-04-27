import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, extent, hidden_size, network_depth):
		super().__init__()
		self.cs = nn.ModuleList()
		for i in range(network_depth):
			in_size = 3 if i == 0 else hidden_size
			self.cs.append(nn.Conv2d(in_size, hidden_size, 3, padding=1))
		self.p = nn.AdaptiveAvgPool2d((extent, extent))

	def forward(self, x):
		for c in self.cs[:-1]:
			x = F.relu(c(x))
		x = self.cs[-1](x)
		return self.p(x)


class WorldModel(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.a = nn.Embedding(3, hidden_size)
		self.r = nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1)
		self.z = nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1)
		self.n = nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1)

	def forward(self, h, a):
		y = self.a(a).unsqueeze(-1).unsqueeze(-1)
		y = y.expand(-1, -1, h.shape[2], h.shape[3])
		x = torch.cat([h, y], 1)
		r = torch.sigmoid(self.r(x))
		z = torch.sigmoid(self.z(x))
		x = torch.cat([r * h, y], 1)
		n = torch.tanh(self.n(x))
		return (1 - z) * h + z * n


class Evaluator(nn.Module):
	def __init__(self, hidden_size, network_depth):
		super().__init__()
		self.cs = nn.ModuleList()
		for _ in range(network_depth):
			self.cs.append(nn.Conv2d(hidden_size, hidden_size, 3, padding=1))
		self.p = nn.AdaptiveAvgPool2d((1, 1))
		self.r = nn.GRU(hidden_size, hidden_size, batch_first=True)
		self.s = nn.Linear(hidden_size, 1)
		self.c = nn.Linear(hidden_size, 1)

	def forward(self, hs):
		b, t = hs.shape[:2]
		x = hs.flatten(0, 1)
		for c in self.cs:
			x = F.relu(c(x))
		x = self.p(x).flatten(1)
		x = x.view(b, t, -1)
		ys, _ = self.r(x)
		live = torch.sigmoid(self.s(ys[:, -1]).squeeze(1))
		eat = torch.sigmoid(self.c(ys).squeeze(-1))
		return live, eat


class System(nn.Module):
	def __init__(self, extent, rollout_depth, network_depth):
		super().__init__()
		hidden_size = 16 * network_depth
		self.enc = Encoder(extent, hidden_size, network_depth)
		self.wm = WorldModel(hidden_size)
		self.ev = Evaluator(hidden_size, network_depth)

	def rollout_h(self, h, acts):
		hs = []
		for i in range(acts.shape[1]):
			h = self.wm(h, acts[:, i])
			hs.append(h)
		hs = torch.stack(hs, 1)
		live, eat = self.ev(hs)
		return hs, live, eat

	def rollout(self, x0, acts):
		h = self.enc(x0)
		return self.rollout_h(h, acts)


def build_system(extent, rollout_depth, network_depth, architecture="baseline"):
	if architecture == "baseline":
		return System(extent, rollout_depth, network_depth)
	if architecture == "dinowm":
		import src.downloads.dinowm as dinowm
		return dinowm.System(extent, rollout_depth, network_depth)
	raise ValueError("architecture must be baseline or dinowm")
