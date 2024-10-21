import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml

from src.utils.logs import get_logger

def data_split(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
        
    logger = get_logger("DATA_SPLIT", log_level=config['base']['log_level'])
    
    logger.info("Load Features")
    dataset = pd.read_csv(config['featurize']['features_path'])
    
    logger.info("Split the Features into train and test set")
    train_data, test_data = train_test_split(dataset,
                                            test_size=config['data_split']['test_size'],
                                            random_state=config['base']['random_state']
                                            )
    
    logger.info("Save the train and test split")
    train_csv_path = config['data_split']['trainset_path']
    test_csv_path = config['data_split']['testset_path']
    # save the dataset to their respective location
    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)
    
    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    args = arg_parser.parse_args()
    
    data_split(config_path=args.config)