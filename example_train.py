from training.train import Config, run_training


cfg = Config("configs/config.json")
run_training(cfg, device="cuda")
