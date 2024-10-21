import argparse
import pandas as pd
from typing import Text
import yaml

from src.utils.logs import get_logger

def get_features(config_path: Text) -> None:
    """_summary_

    Args:
        config_path (Text): path to config
    """
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    logger  = get_logger("FEATURE EXTRACTION", log_level=config['base']['log_level'])
     
     
    # load the data
    logger.info("Load the data")
    data_path = config['data_load']['dataset_csv']
    dataset = pd.read_csv(data_path)
    
       
    # get the features
    logger.info("Feature Extraction")
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']
    featured_dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]

    logger.info('Save features')
    features_path = config['featurize']['features_path']
    featured_dataset.to_csv(features_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    get_features(config_path=args.config)