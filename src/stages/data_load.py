import yaml
import pandas as pd
import argparse
from sklearn.datasets import load_iris

def data_load(config_path)->None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    data = load_iris(as_frame=True)
    dataset = data.frame
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    dataset.to_csv(config['data_load']['dataset_csv'], index=False)
    
    print("Data Load Completed")
    
    
    
if __name__  == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()
    
    data_load(config_path=args.config)
    
    
   
    