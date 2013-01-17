# Import all special functions before registering the Numba module
# type inferer
from numba.special import *
from numba import module_type_inference

import os
import sys
import logging
from numba import typesystem

__version__ = '0.5.0'

# NOTE: Be sure to keep the logging level commented out before commiting.  See:
#   https://github.com/numba/numba/issues/31
# A good work around is to make your tests handle a debug flag, per
# numba.tests.test_support.main().

class _RedirectingHandler(logging.Handler):
    '''
    A log hanlder that applies its formatter and redirect the emission
    to a parent handler.
    '''
    def set_handler(self, handler):
        self.handler = handler

    def emit(self, record):
        # apply our own formatting
        record.msg = self.format(record)
        record.args = [] # clear the args
        # use parent handler to emit record
        self.handler.emit(record)

def _config_logger():
    root = logging.getLogger(__name__)
    format = "\n\033[1m%(levelname)s -- "\
             "%(module)s:%(lineno)d:%(funcName)s\033[0m\n%(message)s"
    try:
        parent_hldr = root.parent.handlers[0]
    except IndexError: # parent handler is not initialized?
        # build our own handler --- uses sys.stderr by default.
        parent_hldr = logging.StreamHandler()
    hldr = _RedirectingHandler()
    hldr.set_handler(parent_hldr)
    fmt = logging.Formatter(format)
    hldr.setFormatter(fmt)
    root.addHandler(hldr)
    root.propagate = False # do not propagate to the root logger

_config_logger()

from . import  special
from numba.typesystem import *
from . import decorators
from numba.minivect.minitypes import FunctionType
from .decorators import *
from numba.error import *

# doctest compatible for jit or autojit numba functions
from numba.tests.test_support import testmod

EXCLUDE_TEST_PACKAGES = ["bytecode"]

def exclude_package_dirs(dirs):
    for exclude_pkg in EXCLUDE_TEST_PACKAGES:
        if exclude_pkg in dirs:
            dirs.remove(exclude_pkg)


def qualified_test_name(root):
    qname = root.replace("/", ".").replace("\\", ".").replace(os.sep, ".")
    return qname + "."

def test():
    import os
    from os.path import dirname, join
    from subprocess import call

    run = failed = 0
    for root, dirs, files in os.walk(join(dirname(__file__), 'tests')):
        qname = qualified_test_name(root)
        exclude_package_dirs(dirs)

        for fn in files:
            if fn.startswith('test_') and fn.endswith('.py'):
                modname, ext = os.path.splitext(fn)
                run += 1
                res = call([sys.executable, '-m', qname + modname])
                if res != 0:
                    failed += 1

    print "ran test files: failed: (%d/%d)" % (failed, run)
    return failed

def nose_run(module=None):
    import nose.config
    import __main__
    config = nose.config.Config()
    config.configure(['--logging-level=DEBUG',
                      '--verbosity=3',      # why is this ignored?
                      # '--with-doctest=1', # not recognized?
                      #'--doctest-tests'
                      ])
    config.verbosity = 3
    nose.run(module=module or __main__, config=config)

__all__ = typesystem.__all__ + decorators.__all__ + special.__all__
