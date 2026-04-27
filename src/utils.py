import pathlib
import random
import src.architectures as arch
import src.environment as env
import src.policies as policies
import struct
import sys
import time
import tomllib
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm.auto
import zlib


def default_cfg():
	return load_cfg()


def seed_all(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.use_deterministic_algorithms(True)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True


def load_cfg(path="config.toml"):
	path = pathlib.Path(path)
	data = tomllib.loads(path.read_text())
	pre = data["preprocess"]
	train = data["train"]
	test = data["test"]
	run = str(data.get("run", ""))
	run_root = path.parent / "runs"
	run_dir = run_root / run if run else run_root
	cfg = {
		"architecture": data.get("architecture", "baseline"),
		"batch_size": train["batch_size"],
		"data_dir": path.parent / "data",
		"dataset_split": pre["dataset_split"],
		"dataset_size": pre.get("dataset_size", 0),
		"device": data.get("device", "cpu"),
		"enc_w": train["enc_w"],
		"episodes_per_start": pre.get("episodes_per_start", 0),
		"epochs": train["epochs"],
		"eval_w": train["eval_w"],
		"grid_extent": pre["grid_extent"],
		"lr": train.get("lr", 1e-3),
		"max_test_samples": test.get("max_test_samples", 0),
		"network_depth": train["network_depth"],
		"planner_mode": test.get("planner_mode", "sampled"),
		"planner_samples": test["planner_samples"],
		"run": run,
		"run_dir": run_dir,
		"run_root": run_root,
		"schedule": tuple(float(x) for x in train.get("schedule", [0, 0, 1])),
		"seed": data["seed"],
		"survival_threshold": test.get("survival_threshold", 0.5),
		"upsampling_factor": pre["upsampling_factor"],
		"weight_decay": train.get("weight_decay", 1e-2),
		"wm_w": train["wm_w"],
	}
	cfg["D"] = cfg["grid_extent"] * cfg["grid_extent"] - 1
	cfg["planner_depth"] = test.get("planner_depth", cfg["D"])
	cfg["max_steps"] = test.get("max_steps", (cfg["D"] - 1) * cfg["D"])
	cfg["discount"] = tuple(float(x) for x in discount(cfg["D"]).tolist())
	return cfg


def device(cfg=None):
	name = "cpu" if cfg is None else cfg.get("device", "cpu")
	if name == "cuda" and torch.cuda.is_available():
		return torch.device("cuda")
	mps = getattr(torch.backends, "mps", None)
	if name == "mps" and mps is not None and mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def ensure_dir(path):
	path = pathlib.Path(path)
	path.mkdir(parents=True, exist_ok=True)
	return path


def split_counts(n, ratios):
	a = int(n * ratios[0])
	b = int(n * ratios[1])
	c = int(n) - a - b
	return a, b, c


def image_u8(img):
	return img.permute(2, 0, 1).contiguous()


def image_f32(img):
	return img.float() / 255.0


def act_id(act):
	return env.AI[act]


def action(i):
	return env.ACTIONS[int(i)]


def torch_gen(seed, dev="cpu"):
	gen = torch.Generator(device=dev)
	gen.manual_seed(int(seed))
	return gen


def discount(depth):
	base = 1.0 - 1.0 / depth
	i = torch.arange(depth, dtype=torch.float32)
	return torch.full((depth,), base, dtype=torch.float32).pow(i)


def sigreg(z, m=32, k=17, gen=None):
	if z.shape[0] < 2:
		return z.new_zeros(())
	ds = torch.randn((m, z.shape[1]), device=z.device, generator=gen)
	ds = ds / ds.norm(dim=1, keepdim=True).clamp_min(1e-12)
	u = z @ ds.t()
	t = torch.linspace(-3, 3, k, device=z.device)
	w = torch.exp(-(t * t))
	g = torch.exp(-0.5 * t * t)
	a = u.t().unsqueeze(-1) * t
	r = torch.cos(a).mean(1)
	i = torch.sin(a).mean(1)
	v = (r - g).square() + i.square()
	return z.shape[0] * (v * w).mean()


def say(msg):
	print(str(msg), flush=True)


def _paths(cfg):
	root = ensure_dir(cfg["run_root"])
	run = ensure_dir(cfg["run_dir"])
	return root, run


def run_dir(cfg):
	return _paths(cfg)[1]


def data_dir(cfg):
	return ensure_dir(cfg["data_dir"])


def _blank():
	return {
		"starts_snake": [],
		"starts_food": [],
		"tx0": [],
		"ta": [],
		"tx1": [],
		"rx0": [],
		"rx1": [],
		"ra": [],
		"rs": [],
		"rc": [],
	}


def _stack(xs, shape, dtype):
	if xs:
		return torch.stack(xs)
	shape = tuple(shape)
	return torch.empty(shape, dtype=dtype)


def _finalize(part, size, depth):
	n = len(env.ACTIONS)
	out = {}
	out["starts_snake"] = _stack(part["starts_snake"], (0, 2, 2), torch.long)
	out["starts_food"] = _stack(part["starts_food"], (0, 2), torch.long)
	out["tx0"] = _stack(part["tx0"], (0, 3, size, size), torch.uint8)
	out["ta"] = _stack(part["ta"], (0, n), torch.long)
	out["tx1"] = _stack(part["tx1"], (0, n, 3, size, size), torch.uint8)
	out["rx0"] = _stack(part["rx0"], (0, 3, size, size), torch.uint8)
	out["rx1"] = _stack(part["rx1"], (0, n, depth, 3, size, size), torch.uint8)
	out["ra"] = _stack(part["ra"], (0, n, depth), torch.long)
	out["rs"] = _stack(part["rs"], (0, n), torch.float32)
	out["rc"] = _stack(part["rc"], (0, n, depth), torch.float32)
	return out


def _branch_rollout(sim, policy, depth, first):
	acts = [act_id(first)]
	xs = [image_u8(sim.display())]
	cons = [float(sim.last_consumed)]
	live = 1.0
	if not sim.state.alive and not sim.state.won:
		live = 0.0
	while len(acts) < depth:
		if not sim.state.alive:
			acts.append(act_id(0))
			xs.append(image_u8(sim.display()))
			cons.append(0.0)
			continue
		act = policy.action(sim.state)
		sim.step(act)
		acts.append(act_id(act))
		xs.append(image_u8(sim.display()))
		cons.append(float(sim.last_consumed))
		if not sim.state.alive and not sim.state.won:
			live = 0.0
	acts = torch.tensor(acts, dtype=torch.long)
	xs = torch.stack(xs)
	live = torch.tensor(live, dtype=torch.float32)
	cons = torch.tensor(cons, dtype=torch.float32)
	return acts, xs, live, cons


def _preprocess_jobs(cfg, rng):
	k = cfg["episodes_per_start"]
	if k < 1:
		counts = split_counts(cfg["dataset_size"], cfg["dataset_split"])
		names = ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]
		return [(name, None) for name in names], counts
	sim = env.Simulator(cfg["grid_extent"], cfg["upsampling_factor"], rng)
	starts = list(sim.starts)
	rng.shuffle(starts)
	counts = split_counts(len(starts), cfg["dataset_split"])
	names = ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]
	jobs = []
	for name, start in zip(names, starts):
		for _ in range(k):
			jobs.append((name, start))
	counts = tuple(x * k for x in counts)
	return jobs, counts


def preprocess(cfg):
	seed_all(cfg["seed"])
	data = data_dir(cfg)
	size = cfg["grid_extent"] * cfg["upsampling_factor"]
	depth = cfg["D"]
	parts = {"train": _blank(), "val": _blank(), "test": _blank()}
	rng = random.Random(cfg["seed"])
	jobs, counts = _preprocess_jobs(cfg, rng)
	bar = tqdm.auto.tqdm(jobs, desc="preprocess")
	for dataset_split, start in bar:
		sim_seed = rng.randrange(1 << 63)
		pol_seed = rng.randrange(1 << 63)
		sim = env.Simulator(cfg["grid_extent"], cfg["upsampling_factor"], random.Random(sim_seed))
		pol_rng = random.Random(pol_seed)
		pol = policies.RandomBackbonePerturbedHamiltonianPolicy(cfg["grid_extent"], pol_rng)
		state = sim.reset(start)
		part = parts[dataset_split]
		part["starts_snake"].append(torch.tensor(state.snake, dtype=torch.long))
		part["starts_food"].append(torch.tensor(state.food, dtype=torch.long))
		while sim.state.alive:
			x0 = image_u8(sim.display())
			pol_act = pol.action(sim.state)
			pol_state = pol.rng.getstate()
			ta = []
			tx1 = []
			rx1 = []
			ra = []
			rs = []
			rc = []
			for act in env.ACTIONS:
				snap = sim.snapshot()
				pol.rng.setstate(pol_state)
				sim.step(act)
				x1 = image_u8(sim.display())
				acts, xs, live, cons = _branch_rollout(sim, pol, depth, act)
				ta.append(act_id(act))
				tx1.append(x1)
				ra.append(acts)
				rx1.append(xs)
				rs.append(live)
				rc.append(cons)
				sim.restore(snap)
			part["tx0"].append(x0)
			part["ta"].append(torch.tensor(ta, dtype=torch.long))
			part["tx1"].append(torch.stack(tx1))
			part["rx0"].append(x0)
			part["rx1"].append(torch.stack(rx1))
			part["ra"].append(torch.stack(ra))
			part["rs"].append(torch.stack(rs))
			part["rc"].append(torch.stack(rc))
			pol.rng.setstate(pol_state)
			sim.step(pol_act)
			bar.set_postfix(split=dataset_split, steps=sim.state.time)
	for name in ("train", "val", "test"):
		out = _finalize(parts[name], size, depth)
		torch.save(out, data / (name + ".pt"))
	meta = {"counts": counts, "depth": depth, "size": size}
	return meta


def _transition_loader(data, batch_size, pin_memory, shuffle, seed=None):
	if data["tx0"].shape[0] == 0:
		return None
	ds = torch.utils.data.TensorDataset(data["tx0"], data["ta"], data["tx1"])
	cls = torch.utils.data.DataLoader
	gen = torch_gen(seed) if shuffle and seed is not None else None
	return cls(ds, batch_size=batch_size, generator=gen, pin_memory=pin_memory, shuffle=shuffle)


def _rollout_loader(data, batch_size, pin_memory, shuffle, seed=None):
	if data["rx0"].shape[0] == 0:
		return None
	ds = torch.utils.data.TensorDataset(data["rx0"], data["rx1"], data["ra"], data["rs"], data["rc"])
	cls = torch.utils.data.DataLoader
	gen = torch_gen(seed) if shuffle and seed is not None else None
	return cls(ds, batch_size=batch_size, generator=gen, pin_memory=pin_memory, shuffle=shuffle)


def _repeat_h(h, n):
	s = tuple(h.shape[1:])
	return h.unsqueeze(1).expand(-1, n, *s).flatten(0, 1)


def _flat_h(h):
	return h.flatten(1)


def _discounted_sum(eat, cfg):
	g = eat.new_tensor(cfg["discount"][:eat.shape[-1]])
	return (eat * g).sum(-1)


def _gated_value(live, eat, cfg):
	gate = (live >= cfg["survival_threshold"]).to(eat.dtype)
	return gate * _discounted_sum(eat, cfg)


def _eff_rank(z):
	if z.shape[0] < 2:
		return 1.0
	z = z.detach().to("cpu")
	z = z.to(torch.float64)
	if not torch.isfinite(z).all():
		return 1.0
	z = z - z.mean(0, keepdim=True)
	e = torch.linalg.svdvals(z).square()
	e = e[torch.isfinite(e)]
	if e.numel() == 0:
		return 1.0
	total = e.sum()
	if not torch.isfinite(total) or total <= 0:
		return 1.0
	p = e / total
	p = p[p > 0]
	if p.numel() == 0:
		return 1.0
	rank = torch.exp(-(p * p.log()).sum())
	if not torch.isfinite(rank):
		return 1.0
	return float(rank.item())


def _encoder_pass(model, loader, opt, dev, cfg, train, gen=None):
	if loader is None:
		return {"loss": 0.0, "wm": 0.0, "enc": 0.0, "rank": 0.0}
	total = {"loss": 0.0, "wm": 0.0, "enc": 0.0, "rank": 0.0}
	n = 0
	model.train(train)
	if gen is None and not train:
		gen = torch_gen(cfg["seed"] + 2, dev)
	bar = tqdm.auto.tqdm(loader, leave=False, disable=not train, desc="encoder")
	for x0, _, x1 in bar:
		x0 = image_f32(x0.to(dev, non_blocking=True))
		x1 = image_f32(x1.to(dev, non_blocking=True))
		with torch.set_grad_enabled(train):
			h0 = model.enc(x0)
			h1 = model.enc(x1.flatten(0, 1))
			z = torch.cat([_flat_h(h0), _flat_h(h1)], 0)
			rank = _eff_rank(z)
			enc0 = sigreg(_flat_h(h0), gen=gen)
			enc1 = sigreg(_flat_h(h1), gen=gen)
			enc = 0.5 * (enc0 + enc1)
			loss = cfg["enc_w"] * enc
			if train:
				opt.zero_grad()
				loss.backward()
				opt.step()
		b = x0.shape[0]
		n += b
		total["loss"] += loss.item() * b
		total["enc"] += enc.item() * b
		total["rank"] += rank * b
	for key in total:
		total[key] /= max(1, n)
	return total


def _transition_pass(model, loader, opt, dev, cfg, train, gen=None):
	if loader is None:
		return {"loss": 0.0, "wm": 0.0, "enc": 0.0, "rank": 0.0}
	total = {"loss": 0.0, "wm": 0.0, "enc": 0.0, "rank": 0.0}
	n = 0
	model.train(train)
	if gen is None and not train:
		gen = torch_gen(cfg["seed"] + 2, dev)
	bar = tqdm.auto.tqdm(loader, leave=False, disable=not train, desc="transition")
	for x0, a, x1 in bar:
		x0 = image_f32(x0.to(dev, non_blocking=True))
		a = a.to(dev, non_blocking=True)
		x1 = image_f32(x1.to(dev, non_blocking=True))
		with torch.set_grad_enabled(train):
			h0 = model.enc(x0)
			h0 = _repeat_h(h0, a.shape[1])
			h1 = model.enc(x1.flatten(0, 1))
			z = torch.cat([_flat_h(h0), _flat_h(h1)], 0)
			rank = _eff_rank(z)
			pred = model.wm(h0, a.flatten())
			wm = F.mse_loss(pred, h1.detach())
			enc0 = sigreg(_flat_h(h0), gen=gen)
			enc1 = sigreg(_flat_h(h1), gen=gen)
			enc = 0.5 * (enc0 + enc1)
			loss = cfg["wm_w"] * wm + cfg["enc_w"] * enc
			if train:
				opt.zero_grad()
				loss.backward()
				opt.step()
		b = x0.shape[0]
		n += b
		total["loss"] += loss.item() * b
		total["wm"] += wm.item() * b
		total["enc"] += enc.item() * b
		total["rank"] += rank * b
	for key in total:
		total[key] /= max(1, n)
	return total


def _rollout_pass(model, loader, opt, dev, cfg, train, eval_loss=True):
	if loader is None:
		return {"loss": 0.0, "wm": 0.0, "live": 0.0, "eat": 0.0}
	total = {"loss": 0.0, "wm": 0.0, "live": 0.0, "eat": 0.0}
	n = 0
	model.train(train)
	bar = tqdm.auto.tqdm(loader, leave=False, disable=not train, desc="rollout")
	for x0, x1, acts, live, cons in bar:
		x0 = image_f32(x0.to(dev, non_blocking=True))
		x1 = image_f32(x1.to(dev, non_blocking=True).flatten(0, 2))
		acts = acts.to(dev, non_blocking=True)
		live = live.to(dev, non_blocking=True).flatten()
		cons = cons.to(dev, non_blocking=True).flatten(0, 1)
		with torch.set_grad_enabled(train):
			h0 = model.enc(x0)
			h0 = _repeat_h(h0, acts.shape[1])
			rows = acts.shape[0] * acts.shape[1]
			h1 = model.enc(x1).view(rows, acts.shape[2], *h0.shape[1:])
			pred_h, pred_live, pred_eat = model.rollout_h(h0, acts.flatten(0, 1))
			wm = F.mse_loss(pred_h, h1.detach())
			s_loss = wm.new_zeros(())
			e_loss = wm.new_zeros(())
			if eval_loss:
				s_loss = F.binary_cross_entropy(pred_live, live)
				e_loss = F.binary_cross_entropy(pred_eat, cons)
			loss = cfg["wm_w"] * wm + cfg["eval_w"] * (s_loss + e_loss)
			if train:
				opt.zero_grad()
				loss.backward()
				opt.step()
		b = x0.shape[0]
		n += b
		total["loss"] += loss.item() * b
		total["wm"] += wm.item() * b
		total["live"] += s_loss.item() * b
		total["eat"] += e_loss.item() * b
	for key in total:
		total[key] /= max(1, n)
	return total


def _save(path, model, opt, cfg, hist, epoch):
	state = {
		"model": model.state_dict(),
		"opt": opt.state_dict(),
		"cfg": cfg,
		"hist": hist,
		"epoch": epoch,
	}
	torch.save(state, path)


def train(cfg):
	seed_all(cfg["seed"])
	_, run = _paths(cfg)
	if (run / "latest.pt").exists() or (run / "best.pt").exists():
		raise FileExistsError("refusing to overwrite existing checkpoints in " + str(run))
	data = data_dir(cfg)
	train_data = torch.load(data / "train.pt", map_location="cpu")
	val_data = torch.load(data / "val.pt", map_location="cpu")
	dev = device(cfg)
	pin = dev.type == "cuda"
	extent = cfg["grid_extent"]
	model = arch.build_system(extent, cfg["D"], cfg["network_depth"], cfg["architecture"])
	model = model.to(dev)
	opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
	train_t = _transition_loader(train_data, cfg["batch_size"], pin, True, cfg["seed"] + 11)
	train_r = _rollout_loader(train_data, cfg["batch_size"], pin, True, cfg["seed"] + 12)
	val_t = _transition_loader(val_data, cfg["batch_size"], pin, False)
	val_r = _rollout_loader(val_data, cfg["batch_size"], pin, False)
	hist = []
	best = None
	wait = 0
	has_val = val_data["rx0"].shape[0] > 0 or val_data["tx0"].shape[0] > 0
	patience = cfg["epochs"] // 10 if has_val else None
	s0 = int(cfg["epochs"] * cfg["schedule"][0])
	s1 = s0 + int(cfg["epochs"] * cfg["schedule"][1])
	if cfg["architecture"] == "dinowm":
		s0 = 0
	last_phase = None

	def trainable(enc, wm, ev):
		if cfg["architecture"] == "dinowm":
			enc = False
		for p in model.enc.parameters():
			p.requires_grad_(enc)
		for p in model.wm.parameters():
			p.requires_grad_(wm)
		for p in model.ev.parameters():
			p.requires_grad_(ev)
		if hasattr(model.enc, "freeze_vision"):
			model.enc.freeze_vision()

	for epoch in range(1, cfg["epochs"] + 1):
		phase = "full"
		if epoch <= s0:
			phase = "encoder"
		elif epoch <= s1:
			phase = "world_model"
		if phase == "full" and last_phase != "full":
			best = None
			wait = 0
		last_phase = phase
		train_gen = torch_gen(cfg["seed"] + epoch * 1000 + 1, dev)
		val_gen = torch_gen(cfg["seed"] + epoch * 1000 + 2, dev)
		if phase == "encoder":
			trainable(True, False, False)
			train_tr = _encoder_pass(model, train_t, opt, dev, cfg, True, train_gen)
			train_ro = {"loss": 0.0, "live": 0.0, "eat": 0.0}
		elif phase == "world_model":
			trainable(False, True, False)
			train_tr = _transition_pass(model, train_t, opt, dev, cfg, True, train_gen)
			train_ro = _rollout_pass(model, train_r, opt, dev, cfg, True, False)
		else:
			trainable(True, True, True)
			train_tr = _transition_pass(model, train_t, opt, dev, cfg, True, train_gen)
			train_ro = _rollout_pass(model, train_r, opt, dev, cfg, True)
		with torch.no_grad():
			if phase == "encoder":
				trainable(True, False, False)
				val_tr = _encoder_pass(model, val_t, opt, dev, cfg, False, val_gen)
				val_ro = {"loss": 0.0, "live": 0.0, "eat": 0.0}
			elif phase == "world_model":
				trainable(False, True, False)
				val_tr = _transition_pass(model, val_t, opt, dev, cfg, False, val_gen)
				val_ro = _rollout_pass(model, val_r, opt, dev, cfg, False, False)
			else:
				trainable(True, True, True)
				val_tr = _transition_pass(model, val_t, opt, dev, cfg, False, val_gen)
				val_ro = _rollout_pass(model, val_r, opt, dev, cfg, False)
		row = {
			"epoch": epoch,
			"phase": phase,
			"train_transition": train_tr["loss"],
			"train_rollout": train_ro["loss"],
			"val_transition": val_tr["loss"],
			"val_rollout": val_ro["loss"],
			"train_rank": train_tr["rank"],
			"val_rank": val_tr["rank"],
		}
		row["train_loss"] = row["train_transition"] + row["train_rollout"]
		row["val_loss"] = row["val_transition"] + row["val_rollout"]
		hist.append(row)
		_save(run / "latest.pt", model, opt, cfg, hist, epoch)
		score = row["val_loss"] if has_val else row["train_loss"]
		is_best = best is None or score < best
		val_s = f"{row['val_loss']:.6f}"
		if has_val and is_best:
			val_s = "\033[1m" + val_s + "\033[0m"
		msg = f"epoch {epoch} phase={phase} train_loss={row['train_loss']:.6f}"
		msg += f" val_loss={val_s}"
		msg += f" train_rank={row['train_rank']:.3f} val_rank={row['val_rank']:.3f}"
		say(msg)
		if is_best:
			best = score
			wait = 0
			_save(run / "best.pt", model, opt, cfg, hist, epoch)
		else:
			wait += 1
			if phase == "full" and has_val and wait > patience:
				break
	return hist


def load_model(cfg, name="best.pt"):
	_, run = _paths(cfg)
	dev = device(cfg)
	sys.modules["pathlib._local"] = pathlib
	state = torch.load(run / name, map_location=dev, weights_only=False)
	model_cfg = state["cfg"]
	depth = model_cfg["grid_extent"] * model_cfg["grid_extent"] - 1
	if "D" in model_cfg:
		depth = model_cfg["D"]
	extent = model_cfg["grid_extent"]
	architecture = model_cfg.get("architecture", "baseline")
	model = arch.build_system(extent, depth, model_cfg["network_depth"], architecture)
	model = model.to(dev)
	model.load_state_dict(state["model"])
	model.eval()
	return model, dev


def _step_tuple(extent, snake, food, act):
	dr = snake[0][0] - snake[1][0]
	dc = snake[0][1] - snake[1][1]
	if act == -1:
		dr, dc = -dc, dr
	if act == 1:
		dr, dc = dc, -dr
	head = snake[0][0] + dr, snake[0][1] + dc
	grew = head == food
	if grew:
		snake = (head,) + snake
	else:
		snake = (head,) + snake[:-1]
	body = set(snake[1:])
	inside = 1 <= head[0] <= extent and 1 <= head[1] <= extent
	if not inside or head in body:
		return snake, None, False, False, False
	if not grew:
		return snake, food, True, False, False
	used = set(snake)
	for row in range(1, extent + 1):
		for col in range(1, extent + 1):
			cell = row, col
			if cell not in used:
				return snake, cell, True, False, True
	return snake, None, False, True, True


def _legal_candidate_fast(snake, food, extent, depth, first, rng):
	acts = [act_id(first)]
	snake, food, alive, _, _ = _step_tuple(extent, snake, food, first)
	while len(acts) < depth:
		if not alive:
			acts.append(act_id(0))
			continue
		act = rng.choice(env.ACTIONS)
		snake, food, alive, _, _ = _step_tuple(extent, snake, food, act)
		acts.append(act_id(act))
	return acts


def _sampled_candidate_tensor(sim, cfg, rng):
	state = sim.state
	depth = cfg["planner_depth"]
	total = max(len(env.ACTIONS), cfg["planner_samples"])
	firsts = []
	actss = []
	for i in range(total):
		first = env.ACTIONS[i] if i < len(env.ACTIONS) else rng.choice(env.ACTIONS)
		firsts.append(first)
		acts = _legal_candidate_fast(state.snake, state.food, sim.extent, depth, first, rng)
		actss.append(acts)
	return firsts, torch.tensor(actss, dtype=torch.long)


def _balanced_candidate_tensor(sim, cfg, rng):
	state = sim.state
	n = len(env.ACTIONS)
	depth = cfg["planner_depth"]
	total = max(n, cfg["planner_samples"])
	base = total // n
	extra = total % n
	firsts = []
	actss = []
	for i, first in enumerate(env.ACTIONS):
		count = base + int(i < extra)
		for _ in range(count):
			firsts.append(first)
			acts = _legal_candidate_fast(state.snake, state.food, sim.extent, depth, first, rng)
			actss.append(acts)
	return firsts, torch.tensor(actss, dtype=torch.long)


def _legal_candidate(sim, depth, first, rng):
	snap = sim.snapshot()
	acts = []
	sim.step(first)
	acts.append(act_id(first))
	while len(acts) < depth:
		if not sim.state.alive:
			acts.append(act_id(0))
			continue
		act = rng.choice(env.ACTIONS)
		sim.step(act)
		acts.append(act_id(act))
	sim.restore(snap)
	return torch.tensor(acts, dtype=torch.long)


def _sampled_candidates(sim, cfg, rng):
	total = max(len(env.ACTIONS), cfg["planner_samples"])
	for i in range(total):
		first = env.ACTIONS[i] if i < len(env.ACTIONS) else rng.choice(env.ACTIONS)
		yield first, _legal_candidate(sim, cfg["planner_depth"], first, rng)


def _balanced_candidates(sim, cfg, rng):
	n = len(env.ACTIONS)
	total = max(n, cfg["planner_samples"])
	base = total // n
	extra = total % n
	for i, first in enumerate(env.ACTIONS):
		count = base + int(i < extra)
		for _ in range(count):
			yield first, _legal_candidate(sim, cfg["planner_depth"], first, rng)


def _exact_candidates(sim, depth):
	snap = sim.snapshot()
	acts = []

	def walk():
		if len(acts) == depth or not sim.state.alive:
			out = acts + [act_id(0)] * (depth - len(acts))
			first = action(out[0])
			yield first, torch.tensor(out, dtype=torch.long)
			return
		for act in env.ACTIONS:
			cur = sim.snapshot()
			sim.step(act)
			acts.append(act_id(act))
			yield from walk()
			acts.pop()
			sim.restore(cur)

	yield from walk()
	sim.restore(snap)


def _candidate_iter(sim, cfg, rng):
	mode = cfg["planner_mode"]
	if mode == "exact":
		return _exact_candidates(sim, cfg["planner_depth"])
	if mode == "sampled":
		return _sampled_candidates(sim, cfg, rng)
	if mode == "balanced":
		return _balanced_candidates(sim, cfg, rng)
	raise ValueError("planner_mode must be exact, sampled, balanced, or cem")


def _take_best(best, score, first, acts, rng, info=None):
	score = float(score)
	if best is None:
		best = {"by_first": {}, "total": 0}
	best["total"] += 1
	cur = best["by_first"].get(first)
	if cur is None or score > cur["score"]:
		best["by_first"][first] = {"actions": acts, "first": first, "info": info, "score": score}
		best["by_first"][first]["ties"] = 1
		return best
	if score == cur["score"]:
		cur["ties"] += 1
		if rng.randrange(cur["ties"]) == 0:
			cur["actions"] = acts
			cur["info"] = info
	return best


def _select_best(best, rng):
	rows = list(best["by_first"].values())
	score = max(row["score"] for row in rows)
	rows = [row for row in rows if row["score"] == score]
	out = dict(rng.choice(rows))
	out["total"] = best["total"]
	return out


def _planner_batch_size(dev):
	if dev.type == "cuda":
		return 256
	if dev.type == "mps":
		return 128
	return 64


def _score_action_tensor(model, dev, h0, acts, cfg):
	rows = acts.tolist()
	acts = acts.to(dev, non_blocking=True)
	if acts.shape[0] == 1:
		_, live, eat = model.rollout_h(h0, acts)
		return _gated_value(live, eat, cfg).cpu()
	prev = {(): 0}
	hs = []
	parents = None
	for i in range(acts.shape[1]):
		cur = {}
		parent = []
		step = []
		inv = []
		for row in rows:
			key = tuple(row[:i + 1])
			j = cur.get(key)
			if j is None:
				j = len(cur)
				cur[key] = j
				parent.append(prev[key[:-1]])
				step.append(row[i])
			inv.append(j)
		parent = torch.tensor(parent, device=dev)
		step = torch.tensor(step, dtype=torch.long, device=dev)
		h = h0[parent] if parents is None else parents[parent]
		parents = model.wm(h, step)
		inv = torch.tensor(inv, device=dev)
		hs.append(parents[inv])
		prev = cur
	hs = torch.stack(hs, 1)
	live, eat = model.ev(hs)
	return _gated_value(live, eat, cfg).cpu()


def _score_true_actions(sim, actss, cfg):
	scores = torch.empty(len(actss), dtype=torch.float32)
	infos = []
	for i, acts in enumerate(actss):
		snap = sim.snapshot()
		live = 1.0
		eat = []
		reason = "horizon"
		for act in acts.tolist():
			if not sim.state.alive:
				eat.append(0.0)
				continue
			sim.step(action(act))
			eat.append(float(sim.last_consumed))
			if not sim.state.alive and not sim.state.won:
				live = 0.0
				reason = "dead"
			if sim.state.won:
				reason = "won"
				continue
		passes = live >= cfg["survival_threshold"]
		score = 0.0
		if passes:
			eat_t = torch.tensor(eat, dtype=torch.float32)
			score = float(_discounted_sum(eat_t, cfg).item())
		scores[i] = score
		infos.append({
			"best_score": float(score),
			"foods_eaten_in_plan": int(sum(eat)),
			"live": bool(live),
			"passes_gate": bool(passes),
			"terminal_reason": reason,
			"actions": tuple(action(act) for act in acts.tolist()),
		})
		sim.restore(snap)
	return scores, infos


def plan_action(model, sim, cfg, rng, dev):
	x0 = image_u8(sim.display())
	x0 = image_f32(x0.unsqueeze(0).to(dev, non_blocking=True))
	if cfg["planner_mode"] in ("sampled", "balanced"):
		with torch.inference_mode():
			h0 = model.enc(x0)
			if cfg["planner_mode"] == "balanced":
				firsts, acts = _balanced_candidate_tensor(sim, cfg, rng)
			else:
				firsts, acts = _sampled_candidate_tensor(sim, cfg, rng)
			scores = _score_action_tensor(model, dev, h0, acts, cfg)
		best = None
		for i, score in enumerate(scores.tolist()):
			best = _take_best(best, score, firsts[i], acts[i], rng)
		best = _select_best(best, rng)
		return best["first"], best["score"], best["total"], len(best["actions"])
	if cfg["planner_mode"] == "cem":
		n = len(env.ACTIONS)
		depth = cfg["planner_depth"]
		total = max(n, cfg["planner_samples"])
		rounds = min(4, max(1, total // n))
		base = total // rounds
		extra = total % rounds

		def draw(ps):
			x = rng.random()
			s = 0.0
			for i, p in enumerate(ps.tolist()):
				s += p
				if x <= s:
					return i
			return n - 1

		def sample(count, probs):
			state = sim.state
			firsts = []
			actss = []
			for _ in range(count):
				snake = state.snake
				food = state.food
				alive = state.alive
				acts = []
				for t in range(depth):
					if not alive:
						acts.append(act_id(0))
						continue
					i = draw(probs[t])
					act = action(i)
					snake, food, alive, _, _ = _step_tuple(sim.extent, snake, food, act)
					acts.append(i)
				firsts.append(action(acts[0]))
				actss.append(acts)
			return firsts, torch.tensor(actss, dtype=torch.long)

		def fit(probs, acts, scores):
			if scores.max().item() == scores.min().item():
				return probs
			elite = max(n, int(round(acts.shape[0] * 0.1)))
			elite = min(elite, acts.shape[0])
			ids = torch.topk(scores, elite).indices
			out = torch.zeros_like(probs)
			for t in range(depth):
				counts = torch.bincount(acts[ids, t], minlength=n).float()
				out[t] = counts / counts.sum()
			return out

		best = None
		probs = torch.full((depth, n), 1 / n)
		with torch.inference_mode():
			h0 = model.enc(x0)
			for i in range(rounds):
				count = base + int(i < extra)
				firsts, acts = sample(count, probs)
				scores = _score_action_tensor(model, dev, h0, acts, cfg)
				for j, score in enumerate(scores.tolist()):
					best = _take_best(best, score, firsts[j], acts[j], rng)
				probs = fit(probs, acts, scores)
		best = _select_best(best, rng)
		return best["first"], best["score"], best["total"], len(best["actions"])
	batch = _planner_batch_size(dev)
	best = None
	actss = []
	firsts = []

	def flush(best):
		if not actss:
			return best
		acts = torch.stack(actss)
		scores = _score_action_tensor(model, dev, h0, acts, cfg)
		for i, score in enumerate(scores.tolist()):
			best = _take_best(best, score, firsts[i], actss[i], rng)
		actss.clear()
		firsts.clear()
		return best

	with torch.inference_mode():
		h0 = model.enc(x0)
		for first, acts in _candidate_iter(sim, cfg, rng):
			firsts.append(first)
			actss.append(acts)
			if len(actss) == batch:
				best = flush(best)
		best = flush(best)
	best = _select_best(best, rng)
	return best["first"], best["score"], best["total"], len(best["actions"])


def plan_oracle_action(sim, cfg, rng):
	batch = _planner_batch_size(torch.device("cpu"))
	best = None
	actss = []
	firsts = []

	def flush(best):
		if not actss:
			return best
		scores, infos = _score_true_actions(sim, actss, cfg)
		for i, score in enumerate(scores.tolist()):
			best = _take_best(best, score, firsts[i], actss[i], rng, infos[i])
		actss.clear()
		firsts.clear()
		return best

	for first, acts in _candidate_iter(sim, cfg, rng):
		firsts.append(first)
		actss.append(acts)
		if len(actss) == batch:
			best = flush(best)
	best = flush(best)
	best = _select_best(best, rng)
	info = best["info"]
	return best["first"], best["score"], best["total"], len(best["actions"]), info


def test(cfg):
	seed_all(cfg["seed"])
	_, run = _paths(cfg)
	data = torch.load(data_dir(cfg) / "test.pt", map_location="cpu")
	model, dev = load_model(cfg)
	n = data["starts_snake"].shape[0]
	limit = n if cfg["max_test_samples"] < 1 else min(n, cfg["max_test_samples"])
	rng = random.Random(cfg["seed"] + 1)
	scores = []
	comps = []
	total = cfg["grid_extent"] * cfg["grid_extent"] - 2
	msg = "test device=" + str(dev) + " episodes=" + str(limit)
	msg += " planner_mode=" + str(cfg["planner_mode"])
	msg += " planner_samples=" + str(cfg["planner_samples"])
	msg += " planner_depth=" + str(cfg["planner_depth"])
	say(msg)
	scale = max(1, 256 // (cfg["grid_extent"] * cfg["upsampling_factor"]))
	img_root = run / "test"
	if img_root.exists() and any(img_root.iterdir()):
		raise FileExistsError("refusing to overwrite existing test frames in " + str(img_root))
	img_root = ensure_dir(img_root)
	bar = tqdm.auto.tqdm(range(limit), desc="test")
	for i in bar:
		sim_seed = rng.randrange(1 << 63)
		sim = env.Simulator(cfg["grid_extent"], cfg["upsampling_factor"], random.Random(sim_seed))
		snake = tuple(tuple(int(x) for x in seg.tolist()) for seg in data["starts_snake"][i])
		food = tuple(int(x) for x in data["starts_food"][i].tolist())
		sim.reset(snake, food)
		img_dir = ensure_dir(img_root / str(i + 1).zfill(4))
		for path in img_dir.glob("*.png"):
			path.unlink()

		def frame_png():
			img = sim.display()
			if scale > 1:
				img = img.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
			h, w, c = img.shape
			raw = img.contiguous().numpy().tobytes()
			rows = [b"\x00" + raw[j:j + w * c] for j in range(0, len(raw), w * c)]
			data = zlib.compress(b"".join(rows))
			head = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
			out = b"\x89PNG\r\n\x1a\n"
			for kind, body in ((b"IHDR", head), (b"IDAT", data), (b"IEND", b"")):
				crc = zlib.crc32(kind + body) & 0xffffffff
				out += struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)
			return out

		def write_frame(name, out):
			(img_dir / (name + ".png")).write_bytes(out)

		last = [("0000", frame_png())]
		cap = cfg["max_steps"]
		steps = 0
		t_ep = time.perf_counter()
		msg = "episode " + str(i + 1) + "/" + str(limit)
		msg += " start food=" + str(food) + " cap=" + str(cap)
		say(msg)
		while sim.state.alive and steps < cap:
			t0 = time.perf_counter()
			act, best, count, depth = plan_action(model, sim, cfg, rng, dev)
			t1 = time.perf_counter()
			sim.step(act)
			steps += 1
			name = str(steps).zfill(4)
			out = frame_png()
			last.append((name, out))
			if len(last) > 50:
				last.pop(0)
			if steps <= 3 or steps % 10 == 0:
				msg = "episode " + str(i + 1) + " step " + str(steps)
				msg += " action=" + str(act) + " score=" + str(round(best, 4))
				msg += " plans=" + str(count) + " best_depth=" + str(depth)
				msg += " plan_s=" + str(round(t1 - t0, 3))
				msg += " len=" + str(len(sim.state.snake)) + " alive=" + str(sim.state.alive)
				msg += " won=" + str(sim.state.won)
				say(msg)
		for name, out in last:
			write_frame(name, out)
		comp = max(0, min(total, len(sim.state.snake) - 2))
		score = comp / total
		comps.append(comp)
		scores.append(score)
		t_done = round(time.perf_counter() - t_ep, 3)
		msg = "episode " + str(i + 1) + " done steps=" + str(steps)
		msg += " completion=" + str(round(score, 4)) + " time_s=" + str(t_done)
		say(msg)
		bar.set_postfix(score=score)
	if comps:
		vals = sorted(comps)
		mid = len(vals) // 2
		if len(vals) % 2 == 1:
			med = vals[mid] / total
		else:
			med = (vals[mid - 1] + vals[mid]) / (2 * total)
		mean = sum(comps) / (len(comps) * total)
		med = format(med, ".10f")
		mean = format(mean, ".10f")
	else:
		med = "0.0000000000"
		mean = "0.0000000000"
	out = {"mean_completion": mean, "median_completion": med, "n": len(scores), "scores": scores}
	say("saved test PNGs to " + str(img_root))
	return out


def preview(cfg, steps=8):
	rng = random.Random(cfg["seed"])
	sim_rng = random.Random(rng.randrange(1 << 63))
	sim = env.Simulator(cfg["grid_extent"], cfg["upsampling_factor"], sim_rng)
	pol_rng = random.Random(rng.randrange(1 << 63))
	pol = policies.RandomBackbonePerturbedHamiltonianPolicy(cfg["grid_extent"], pol_rng)
	sim.reset()
	frames = [sim.display().clone()]
	while sim.state.alive and len(frames) < steps:
		act = pol.action(sim.state)
		sim.step(act)
		frames.append(sim.display().clone())
	return frames
