import copy_reg
import functools
import logging
import os
import cPickle as pickle
import sys
import time

LOGDIR = os.path.expanduser('~/python/logfiles/')

def asctime():
    return time.strftime('%Y-%m-%dT%H:%M:%S')

def initialize_logging(filename=None, level=logging.INFO, filelevel=None):
    """Configure the root logger.

    The default handler for the root logger logs to stdout.  Also log
    to a file if a filename is specified.
    """

    #if filename is None:
    #    filename = os.path.join(LOGDIR, '%s.%s.log' % (__file__, asctime()))

    reload(logging)

    log = logging.getLogger()
    log.setLevel(level)
    
    formatter = logging.Formatter("PID:%(process)d "
                                  "%(levelname)s %(name)s %(asctime)s "
                                  "%(filename)s:%(lineno)d %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    log.addHandler(handler)

    if filename:
        if not filelevel:
            filelevel = level
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        handler.setLevel(filelevel)
        log.addHandler(handler)


# pickle doesn't play nice with functools.partial unless we do
# something like this.
def _reconstruct_partial(f, args, kwargs):
    return functools.partial(f, *args, **(kwargs or {}))

def _reduce_partial(p):
    return _reconstruct_partial, (p.func, p.args, p.keywords)

copy_reg.pickle(functools.partial, _reduce_partial)


def dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

