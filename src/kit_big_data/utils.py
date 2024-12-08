<<<<<<< HEAD
from logging_config import setup_logging

logger = setup_logging()

def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Ne pas logger les interruptions clavier
        return
=======
from logging_config import setup_logging

logger = setup_logging()

def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Ne pas logger les interruptions clavier
        return
>>>>>>> Nohaila2
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))