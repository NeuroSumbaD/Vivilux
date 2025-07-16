'''Submodule for defining a log file with a name and path matching the calling
    script. Meant to log information about the training process and provide
    error and debugging information for simulation and hardware functions.
'''

import logging
import os
import __main__

# Set up logging
def setup_logger():
    main_file = os.path.abspath(__main__.__file__)
    main_dir = os.path.dirname(main_file)
    logs_dir = os.path.join(main_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)  # Ensure the logs directory exists

    filename = os.path.basename(main_file)
    filename = filename.replace('.py', '.log')
    log_path = os.path.join(logs_dir, filename)

    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    
    # Check if handler already exists to avoid duplicates
    if not logger.handlers:
        # Create file handler and set formatter
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# initialize logger
log = setup_logger()
log.info("NEW RUN STARTED.")