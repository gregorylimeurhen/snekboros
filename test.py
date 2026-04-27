import src.utils as u


def main():
	cfg = u.default_cfg()
	out = u.test(cfg)
	print(out, flush=True)


if __name__ == "__main__":
	main()
