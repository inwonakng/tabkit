import logging


def setup_logger(
    logger_name: str = "Logger",
    stdout_file: str | None = None,
    stderr_file: str | None = None,
    silent: bool = False,
):
    """Setup and return a logger with specified name and level, including file handler."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Prevent log messages from being duplicated in the Python root logger
    logger.propagate = False
    # Check if handlers are already added (important in scenarios where setup_logger might be called multiple times)
    if not logger.handlers:
        if not silent:
            # Create a console handler and set the level to info
            if stdout_file is not None and stderr_file is not None:
                stdout_handler = logging.FileHandler(stdout_file)
                stdout_handler.setLevel(logging.INFO)
                stderr_handler = logging.FileHandler(stderr_file)
                stderr_handler.setLevel(logging.ERROR)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.terminator = "\n"

            # Create a file handler and set the level to info
            # cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # log_file = LOGGER_DIR / f"{cur_time}-experiment.log"
            # fh = logging.FileHandler(log_file)
            # fh.setLevel(logging.INFO)
            # fh.terminator = "\n"

            # Create formatter and add it to the handlers
            formatter = logging.Formatter(
                # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                "%(asctime)s - %(name)s - %(message)s"
            )
            ch.setFormatter(formatter)
            # fh.setFormatter(formatter)

            logger.addHandler(ch)
            # logger.addHandler(fh)
        else:
            logger.addHandler(logging.NullHandler())

    return logger


def silence_logger():
    for logger_name in logging.root.manager.loggerdict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
