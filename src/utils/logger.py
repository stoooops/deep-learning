#!/usr/bin/env python

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from src.utils.file_utils import LOG_DIR
from functools import partial
from src.utils.color_utils import bcolors

# use WARN instead of WARNING and FATAL instead of CRITICAL
# This way our formatters can pad width at 5 characters instead of 8,
# allowing for 3 more characters per log line
logging._levelToName[logging.WARNING] = 'WARN'
logging._levelToName[logging.CRITICAL] = 'FATAL'


FILENAME = os.path.normpath(os.path.join(LOG_DIR, 'huli.log'))


class Logging:
    """Static logging functionality"""

    # extra handlers to be attached when constructing a new logger
    _HANDLERS = []

    # keep track of all the loggers, so that when we add extra handlers later,
    # we can attach them to the already instantiated loggers
    _LOGGERS = {}

    # the logfile handler
    _LOGFILE_HANDLER = TimedRotatingFileHandler(
        FILENAME,         # filename
        when='D', interval=1,  # once per day
        backupCount=14,        # keep two weeks of backups
        utc=False              # local time
        )

    # Format is as follows:
    # 2019-05-10 11:11:21,644 DEBUG    [menehune.curses_runner:358] (MainThread) test debug log
    # 2019-05-10 11:11:21,645 INFO     [menehune.curses_runner:359] (MainThread) test info log
    # 2019-05-10 11:11:21,646 WARNING  [menehune.curses_runner:360] (MainThread) test warn log
    # 2019-05-10 11:11:21,647 ERROR    [menehune.curses_runner:361] (MainThread) test error log
    # 2019-05-10 11:11:21,649 CRITICAL [menehune.curses_runner:362] (MainThread) test critical log
    FORMATTER = logging.Formatter('%(asctime)s %(levelname)-5s [%(name)s:%(lineno)s] (%(threadName)s) %(message)s')
    _LOGFILE_HANDLER.setFormatter(FORMATTER)

    # always use LOGFILE_HANDLER
    _HANDLERS.append(_LOGFILE_HANDLER)

    _std_out_attached = _debug_dim = _info_blue = _warn_yellow = _error_red = False

    @staticmethod
    def get_logger(name):
        """Construct a new logger with the given name. Clients should pass in __name__."""
        if Logging._LOGGERS.get(name) is not None:
            return Logging._LOGGERS[name]

        logger = logging.getLogger(name)

        # set log level
        logger.setLevel(logging.DEBUG)

        # attach any extra handlers
        for handler in Logging._HANDLERS:
            logger.addHandler(handler)

        # keep track of all the loggers
        Logging._LOGGERS[name] = logger

        # Don't let tensorflow duplicate this logger
        logger.propagate = False

        if Logging._debug_dim:
            Logging.debug_dim(logger=logger)

        if Logging._info_blue:
            Logging.info_blue(logger=logger)

        if Logging._warn_yellow:
            Logging.warn_yellow(logger=logger)

        if Logging._error_red:
            Logging.error_red(logger=logger)

        return logger

    @staticmethod
    def attach_stdout():
        if not Logging._std_out_attached:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(Logging.FORMATTER)
            Logging.attach_handler(stdout_handler)

        Logging._std_out_attached = True

    @staticmethod
    def debug_dim(logger=None):
        if logger is not None:
            logger.debug = partial(bcolors.dim, logger.info)
            return

        # else apply to all if we haven't already
        if not Logging._debug_dim:
            # update existing loggers
            for name, logger in Logging._LOGGERS.items():
                Logging.debug_dim(logger=logger)

        Logging._debug_dim = True

    @staticmethod
    def info_blue(logger=None):
        if logger is not None:
            logger.info = partial(bcolors.light_blue, logger.info)
            return

        # else apply to all if we haven't already
        if not Logging._info_blue:
            # update existing loggers
            for name, logger in Logging._LOGGERS.items():
                Logging.info_blue(logger=logger)

        Logging._info_blue = True

    @staticmethod
    def warn_yellow(logger=None):
        if logger is not None:
            logger.warn = partial(bcolors.light_yellow, logger.warn)
            logger.warning = partial(bcolors.light_yellow, logger.warning)
            return

        # else apply to all if we haven't already
        if not Logging._warn_yellow:
            # update existing loggers
            for name, logger in Logging._LOGGERS.items():
                Logging.warn_yellow(logger=logger)

        Logging._warn_yellow = True

    @staticmethod
    def error_red(logger=None):
        if logger is not None:
            logger.error = partial(bcolors.light_red, logger.error)
            logger.exception = partial(bcolors.light_red, logger.exception)
            return

        # else apply to all if we haven't already
        if not Logging._error_red:
            # update existing loggers
            for name, logger in Logging._LOGGERS.items():
                Logging.error_red(logger=logger)

        Logging._error_red = True

    @staticmethod
    def attach_handler(handler):
        """
        Add a handler to the list of extra handlers to be attached when creating a new logger instance
        """
        Logging._HANDLERS.append(handler)

        # update existing loggers
        for name, logger in Logging._LOGGERS.items():
            logger.addHandler(handler)

    @staticmethod
    def get_file_handles(logger):
        """
        Get a list of filehandle numbers from logger. This is useful for clients that need
        to explicitly preserve these file handles such as DaemonContext.files_preserve
        """
        handles = [handler.stream.fileno() for handler in logger.handlers]
        if logger.parent:
            handles += Logging.get_file_handles(logger.parent)
        return handles

