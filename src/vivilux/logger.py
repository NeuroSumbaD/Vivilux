'''Submodule for defining a log file with a name and path matching the calling
    script. Meant to log information about the training process and provide
    error and debugging information for simulation and hardware functions.
'''

import logging
import os
import __main__

# Set up logging
def setup_logger():
    filename = os.path.basename(__main__.__file__)
    filename = filename.replace('.py', '.log')
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    
    # Check if handler already exists to avoid duplicates
    if not logger.handlers:
        # Create file handler and set formatter
        file_handler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# initialize logger
log = setup_logger()