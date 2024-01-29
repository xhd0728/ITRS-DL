import logging


class Logger:
    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
            cls._logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s')
            formatter.datefmt = '%Y-%m-%d,%H:%M:%S'
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(console_handler)
        return cls._logger


def main():
    logger = Logger.get_logger()
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical message')


if __name__ == '__main__':
    main()
