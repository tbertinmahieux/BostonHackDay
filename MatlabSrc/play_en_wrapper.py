#!/usr/bin/env
"""
Wrapper around the matlab command play_en from Dan
needs to have the matlab files + matlab installed!
"""

import os
import sys
import string
import numpy as N
import scipy
import scipy.io



#########################################################################
# read from a process
# originally written by Douglas Eck: douglas.eck@umontreal.ca
#########################################################################
def command_with_output(cmd):
    if not type(cmd) == unicode :
        cmd = unicode(cmd,'utf-8')
    # hack just for Darwin, those lucky people
    if sys.platform=='darwin' :
        cmd = unicodedata.normalize('NFC',cmd)
    child = os.popen(cmd.encode('utf-8'))
    data = child.read()
    err = child.close()
    return data




def play_en(M) :
    """Try to call play_en matlab code from python."""


    tmpfileIn = 'dummy_playenwrapperpy_infile.mat'
    tmpfileOut = 'dummy_playenwrapperpy_outfile.mat'

    # save M to file
    d = dict()
    d['M'] = M
    scipy.io.savemat(tmpFileIn,d)

    cmd = 'run_matlab_command.sh play_en_wrapper.m '
    cmd = cmd + tmpFileIn + ' ' + tmpFileOut

    # call
    result = command_with_output(cmd)

    # read file
    mfile = scipy.io.loadmat(tmpFileOut)

    return mfile[signal]

