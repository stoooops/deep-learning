#!/usr/bin/env python


from src.utils.io_utils import pre_postfix_print_fn

class bcolors:
    """
    https://misc.flogisoft.com/bash/tip_colors_and_formatting

    Plus testing it out myself
    """

    RESET_ALL = '\033[0m'  # reset all attributes

    @staticmethod
    def indirect(prefix, print_fn, *argv, **kwargs):
        pre_postfix_print_fn(print_fn, prefix, bcolors.RESET_ALL)(*argv, **kwargs)

    BOLD = '\033[1m'

    @staticmethod
    def bold(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BOLD, print_fn, *argv, **kwargs)

    DIM = '\033[2m'  # opposite of bold

    @staticmethod
    def dim(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.DIM, print_fn, *argv, **kwargs)

    ITALICS = '\033[3m'

    @staticmethod
    def italics(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.ITALICS, print_fn, *argv, **kwargs)

    UNDERLINE = '\033[4m'

    @staticmethod
    def underline(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.UNDERLINE, print_fn, *argv, **kwargs)

    BLINK = '\033[5m'  # yes really

    @staticmethod
    def blink(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BLINK, print_fn, *argv, **kwargs)

    # 6 doesn't seem to do anything

    INVERT = '\033[7m'

    @staticmethod
    def invert(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.INVERT, print_fn, *argv, **kwargs)

    HIDDEN = '\033[8m'  # .e.g for passwords

    @staticmethod
    def hidden(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.HIDDEN, print_fn, *argv, **kwargs)

    STRIKETHROUGH = '\033[9m'

    @staticmethod
    def strikethrough(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.STRIKETHROUGH, print_fn, *argv, **kwargs)

    # 10-20 don't seem to do anything
    DOUBLE_UNDERLINE = '\033[21m'

    @staticmethod
    def double_underline(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.DOUBLE_UNDERLINE, print_fn, *argv, **kwargs)

    # 22-29 reset various other properties

    BLACK = '\033[30m'

    @staticmethod
    def black(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BLACK, print_fn, *argv, **kwargs)

    RED = '\033[31m'

    @staticmethod
    def red(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.RED, print_fn, *argv, **kwargs)

    GREEN = '\033[32m'

    @staticmethod
    def green(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.GREEN, print_fn, *argv, **kwargs)

    YELLOW = '\033[33m'

    @staticmethod
    def yellow(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.YELLOW, print_fn, *argv, **kwargs)

    BLUE = '\033[34m'

    @staticmethod
    def blue(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BLUE, print_fn, *argv, **kwargs)

    CYAN = '\033[36m'

    @staticmethod
    def cyan(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.CYAN, print_fn, *argv, **kwargs)

    MAGENTA = '\033[35m'

    @staticmethod
    def magenta(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.MAGENTA, print_fn, *argv, **kwargs)

    LIGHT_GRAY = '\033[37m'
    LIGHT_GREY = LIGHT_GRAY

    @staticmethod
    def light_gray(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_GRAY, print_fn, *argv, **kwargs)

    # 38 doesn't seem to do anything

    light_grey = light_gray

    BACKGROUND_BLACK = '\033[39m'

    @staticmethod
    def background_black(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BACKGROUND_BLACK, print_fn, *argv, **kwargs)

    BACKGROUND_RED = '\033[41m'  # basically orange

    @staticmethod
    def background_red(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_RED, print_fn, *argv, **kwargs)

    BACKGROUND_GREEN = '\033[42m'

    @staticmethod
    def background_green(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BACKGROUND_GREEN, print_fn, *argv, **kwargs)

    BACKGROUND_YELLOW = '\033[43m'

    @staticmethod
    def background_yellow(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BACKGROUND_YELLOW, print_fn, *argv, **kwargs)

    BACKGROUND_BLUE = '\033[44m'

    @staticmethod
    def background_blue(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BACKGROUND_BLUE, print_fn, *argv, **kwargs)

    BACKGROUND_MAGENTA = '\033[45m'

    @staticmethod
    def background_magenta(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BACKGROUND_MAGENTA, print_fn, *argv, **kwargs)

    BACKGROUND_CYAN = '\033[46m'

    @staticmethod
    def background_cyan(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BACKGROUND_CYAN, print_fn, *argv, **kwargs)

    BACKGROUND_BACKGROUND_GRAY = '\033[47m'
    BACKGROUND_BACKGROUND_GREY = BACKGROUND_BACKGROUND_GRAY

    @staticmethod
    def background_gray(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.BACKGROUND_BACKGROUND_GRAY, print_fn, *argv, **kwargs)

    background_grey = background_gray

    # 48-51 don't seem to do anything

    BOTTOM_DOUBLE_UNDERLINE = '\033[52m'  # the bottom of two underlines, a bit lower than normal underline

    # 53-89 don't seem to do anything
    DARK_GRAY = '\033[90m'
    DARK_GREY = DARK_GRAY

    @staticmethod
    def dark_gray(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.DARK_GRAY, print_fn, *argv, **kwargs)

    dark_grey = dark_gray

    LIGHT_RED = '\033[91m'

    @staticmethod
    def light_red(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_RED, print_fn, *argv, **kwargs)

    LIGHT_GREEN = '\033[92m'

    @staticmethod
    def light_green(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_GREEN, print_fn, *argv, **kwargs)

    LIGHT_YELLOW = '\033[93m'

    @staticmethod
    def light_yellow(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_YELLOW, print_fn, *argv, **kwargs)

    LIGHT_BLUE = '\033[94m'

    @staticmethod
    def light_blue(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_BLUE, print_fn, *argv, **kwargs)

    LIGHT_MAGENTA = '\033[95m'

    @staticmethod
    def light_magenta(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_MAGENTA, print_fn, *argv, **kwargs)

    LIGHT_CYAN = '\033[96m'

    @staticmethod
    def light_cyan(print_fn, *argv, **kwargs):
        bcolors.indirect(bcolors.LIGHT_CYAN, print_fn, *argv, **kwargs)

    # 97-99 don't seem to do anything
    BACKGROUND_DARK_GRAY = '\033[100m'
    BACKGROUND_DARK_GREY = BACKGROUND_DARK_GRAY
    BACKGROUND_LIGHT_RED = '\033[101m'
    BACKGROUND_LIGHT_GREEN = '\033[102m'
    BACKGROUND_LIGHT_YELLOW = '\033[103m'
    BACKGROUND_LIGHT_BLUE = '\033[104m'
    BACKGROUND_LIGHT_MAGENTA = '\033[105m'
    BACKGROUND_LIGHT_CYAN = '\033[106m'
    BACKGROUND_LIGHT_WHITE = '\033[107m'

