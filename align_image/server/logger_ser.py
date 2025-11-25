import sys
sys.path.append('.')
import logging
import logging.handlers
import time

import os

# Create a custom logger
logger = logging.getLogger(__name__)


def logging_config():
    # Create handlers
    if not os.path.exists('log'):
        os.mkdir('log')
    base = os.path.join('log', time.strftime('%Y%m%d_%H%M%S'))
    log_filename = base + '.log'
    log_running = 'running.log'
    c_handler = logging.StreamHandler()
    f_handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=104857600, backupCount=10)  # 10485760
    r_handler = logging.handlers.RotatingFileHandler(log_running, maxBytes=104857600, backupCount=1)

    # Create formatters and add it to handlers
    c_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)5s - %(module)s.%(funcName)s():%(lineno)d - %(message)s')
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)5s - %(module)s.%(funcName)s():%(lineno)d - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    r_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.addHandler(r_handler)
    logger.setLevel(logging.INFO)


logging_config()
