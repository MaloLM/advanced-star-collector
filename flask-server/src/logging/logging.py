"""
Different logging levels:

logger.debug("Detailed debugging information")
logger.info("General informational messages")
logger.warning("Warning about a minor problem")
logger.error("Error message for a major problem")
logger.critical("Critical issue, program may not be able to continue")
"""
import os
import logging
from logging.handlers import RotatingFileHandler

log_directory = '../data/logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

app_log_path = os.path.join(log_directory, 'application.log')
ml_log_path = os.path.join(log_directory, 'ml_metrics.log')


def setup_loggers():
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    app_logger = logging.getLogger('app_logger')
    if not app_logger.hasHandlers():
        app_log_handler = RotatingFileHandler(
            app_log_path, maxBytes=1e6, backupCount=5)
        app_log_handler.setFormatter(log_formatter)
        app_logger.setLevel(logging.INFO)
        app_logger.addHandler(app_log_handler)

    ml_logger = logging.getLogger('ml_logger')
    if not ml_logger.hasHandlers():
        ml_log_handler = RotatingFileHandler(
            ml_log_path, maxBytes=1e6, backupCount=5)
        ml_log_handler.setFormatter(log_formatter)
        ml_logger.setLevel(logging.INFO)
        ml_logger.addHandler(ml_log_handler)


setup_loggers()
