import os
import logging
import yaml
import pandas as pd
from logging import config


def ensure_dir(file_path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        file_path (str): Path to file
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_date(date_str):
    for fmt in ('%d/%m/%y', '%d/%m/%Y'):
        try:
            return pd.datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    raise ValueError(f'no valid date format found for {date_str}')

def setup_logging(log_config_path: str, log_dir: str):
    """Setup logging configuration

    Args:
        log_config_path (str): Path to config file
        log_dir (str): Path to log directory
    """
    
    ensure_dir(log_dir)
    with open(log_config_path, 'rt') as f:
        log_config = yaml.safe_load(f.read().replace('LOG_DIR', log_dir))

    logging.config.dictConfig(log_config)