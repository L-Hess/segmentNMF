import pathlib
import logging
import sys
import os

def logger_setup():
    cp_dir = pathlib.Path(os.getcwd() + '/.segmentNMF')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ]
                )
    logger = logging.getLogger(__name__)
    logger.info(f'WRITING LOG OUTPUT TO {log_file}')
    #logger.handlers[1].stream = sys.stdout

    return logger, log_file