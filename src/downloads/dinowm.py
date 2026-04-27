import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2VisionEncoder(nn.Module):
	def __init__(self, model_name="dinov2_vits14", frozen=True):
		super().__init__()
		self.base_model = torch.hub.load("facebookresearch/dinov2", model_name)
		self.emb_dim = self.base_model.embed_dim
		if frozen:
			for p in self.base_model.parameters():
				p.requires_grad_(False)
			self.base_model.eval()

	def forward(self, x):
		return self.base_model.forward_features(x)["x_norm_patchtokens"]


class Encoder(nn.Module):
	def __init__(self, extent):
		super().__init__()
		self.vision = DinoV2VisionEncoder(frozen=True)
		self.hidden_size = self.vision.emb_dim
		self.p = nn.AdaptiveAvgPool2d((extent, extent))
		mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
		std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
		self.register_buffer("mean", mean)
		self.register_buffer("std", std)
		self.freeze_vision()

	def freeze_vision(self):
		self.vision.eval()
		for p in self.vision.parameters():
			p.requires_grad_(False)

	def forward(self, x):
		self.vision.eval()
		x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
		x = (x - self.mean) / self.std
		with torch.no_grad():
			x = self.vision(x)
		b, n, c = x.shape
		side = int(math.isqrt(n))
		x = x.transpose(1, 2).reshape(b, c, side, side)
		return self.p(x)


class WorldModel(nn.Module):
	def __init__(self, extent, hidden_size, network_depth):
		super().__init__()
		heads = 4
		feed = 4 * hidden_size
		layer = nn.TransformerEncoderLayer(hidden_size, heads, feed, 0.0, "gelu", batch_first=True)
		self.a = nn.Embedding(3, hidden_size)
		self.p = nn.Parameter(torch.randn(1, extent * extent + 1, hidden_size) * 0.02)
		self.t = nn.TransformerEncoder(layer, network_depth)
		self.o = nn.Linear(hidden_size, hidden_size)

	def forward(self, h, a):
		b, c, hgt, wid = h.shape
		x = h.flatten(2).transpose(1, 2)
		y = self.a(a).unsqueeze(1)
		x = torch.cat([y, x], 1)
		x = x + self.p[:, :x.shape[1]]
		x = self.t(x)[:, 1:]
		x = self.o(x).transpose(1, 2).reshape(b, c, hgt, wid)
		return h + x


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
		self.enc = Encoder(extent)
		hidden_size = self.enc.hidden_size
		self.wm = WorldModel(extent, hidden_size, network_depth)
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
