#! /usr/bin/env python
"""
Simple code to call play_en from Dan that plays an Echo Nest
feature file.

T. Bertin-Mahieux (2010) Columbia.edu
www.columbia.edu/~tb2332/
"""

import sys
from mlabwrap import mlab


def call(F):
    """
    Main function.
    Call the code using feature F
    """
    
    mlab._do('play_en(F)')
