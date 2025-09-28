from train import Config, run_training


cfg = Config("config.json")
run_training(cfg, device="cuda")
