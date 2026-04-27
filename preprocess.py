import src.utils as u


def main():
	cfg = u.default_cfg()
	meta = u.preprocess(cfg)
	print(meta, flush=True)


if __name__ == "__main__":
	main()
