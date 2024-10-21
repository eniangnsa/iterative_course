import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score
from typing import Text, Dict
import yaml

from src.reports.visualize import plot_confusion_matrix
from src.utils.logs import get_logger

def evaluate_model(config_path: Text) -> None:
    """_summary_

    Args:
        config_path (Text): Path to config
    """
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
        
    logger = get_logger("EVALUATE", log_level=config['base']['log_level'])
    
    #load the model
    logger.info("Load the model from model_path")
    model_path = config['train']['model_path']
    model = joblib.load(model_path)
    
    # load the test set
    logger.info("Load the test data")
    test_set_path= config['data_split']['testset_path']
    test_df = pd.read_csv(test_set_path)
    target_col = config['featurize']['target_column']
    y_test = test_df.loc[:, target_col].values
    x_test = test_df.drop(target_col, axis=1).values
    
    
    # make prediction on test set
    logger.info("Make Predictions")
    predictions = model.predict(x_test)
    
    # evaluate the performance of the model
    logger.info("Evaluate the performance of the model")
    f1 = f1_score(y_true=y_test, y_pred=predictions, average='macro')
    cm =  confusion_matrix(y_pred=predictions, y_true=y_test)
    reports = {
        'f1': f1,
        'cm': cm,
        'Actual': y_test,
        'Predicted': predictions
    }
    
    # save metrics
    logger.info("Save Metrics")
    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder  / config['evaluate']['metrics_file']
    
    json.dump(
        obj={'f1_score': reports['f1']},
        fp=open(metrics_path, 'w')
    )
    
    logger.info(f'F1 metrics file saved to : {metrics_path}')
    
    
    logger.info('Save confusion matrix')
    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=reports['cm'],
                                target_names=load_iris(as_frame=True).target_names.tolist(),
                                normalize=False)
    confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
    plt.savefig(confusion_matrix_png_path)
    logger.info(f'Confusion matrix saved to : {confusion_matrix_png_path}')
    
if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)