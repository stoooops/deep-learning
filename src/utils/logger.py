#!/usr/bin/env python

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from src.utils.file_utils import TMP_DIR

# use WARN instead of WARNING and FATAL instead of CRITICAL
# This way our formatters can pad width at 5 characters instead of 8,
# allowing for 3 more characters per log line
logging._levelToName[logging.WARNING] = 'WARN'
logging._levelToName[logging.CRITICAL] = 'FATAL'


FILENAME = os.path.normpath(os.path.join(TMP_DIR, 'huli.log'))


class HuliLogging:
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

    _std_out_attached = False

    @staticmethod
    def get_logger(name):
        """Construct a new logger with the given name. Clients should pass in __name__."""
        if HuliLogging._LOGGERS.get(name) is not None:
            return HuliLogging._LOGGERS[name]

        logger = logging.getLogger(name)

        # set log level
        logger.setLevel(logging.DEBUG)

        # attach any extra handlers
        for handler in HuliLogging._HANDLERS:
            logger.addHandler(handler)

        # keep track of all the loggers
        HuliLogging._LOGGERS[name] = logger

        return logger

    @staticmethod
    def attach_stdout():
        if not HuliLogging._std_out_attached:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(HuliLogging.FORMATTER)
            HuliLogging.attach_handler(stdout_handler)

        HuliLogging._std_out_attached = True

    @staticmethod
    def attach_handler(handler):
        """
        Add a handler to the list of extra handlers to be attached when creating a new logger instance
        """
        HuliLogging._HANDLERS.append(handler)

        # update existing loggers
        for name, logger in HuliLogging._LOGGERS.items():
            logger.addHandler(handler)

    @staticmethod
    def get_file_handles(logger):
        """
        Get a list of filehandle numbers from logger. This is useful for clients that need
        to explicitly preserve these file handles such as DaemonContext.files_preserve
        """
        handles = [handler.stream.fileno() for handler in logger.handlers]
        if logger.parent:
            handles += HuliLogging.get_file_handles(logger.parent)
        return handles


class HuliLoggingNameFilter:
    """Filter logs based on log record: %(name)s"""
    def __init__(self, name):
        self._name = name

    def filter(self, logRecord):
        return logRecord.name == self._name

