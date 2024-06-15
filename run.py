from train import SimCLR
import yaml
from dataloader.dataset_wrapper import DataSetWrapper
import wandb

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    wandb.init(project="ConVIRT", config=config)
    
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
