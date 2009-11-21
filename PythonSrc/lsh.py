#!/usr/bin/env python

#########################################################################
#
# lsh.py
# wrapper around the LSH code E2LSH-0.1 from MIT
#
# Simple idea:
#   - data to temp file
#   - call command line LSH
#   - read output file
#
#########################################################################



import string
import sys
import os
import tempfile
import unicodedata
import numpy as N



#########################################################################
# read from a process
# originally written by Douglas Eck: douglas.eck@umontreal.ca
#########################################################################
def command_with_output(cmd):
    if not type(cmd) == unicode :
        cmd = unicode(cmd,'utf-8')
    #should this be a part of slashify or command_with_output?
    if sys.platform=='darwin' :
        cmd = unicodedata.normalize('NFC',cmd)
    child = os.popen(cmd.encode('utf-8'))
    data = child.read()
    err = child.close()
    return data



#########################################################################
# write an input numpy matrix to a file and apply the E2LSH algorithm.
# E2LSH bin must be in the path, or provided as an input
# output file path is returned.
#
# INPUT
#
#     input - numpy matrix, one example per line, all examples same size
#  fOutName - E2LSH output file name, if none a dummy name will be used
#   fInName - file name for the E2LSH input, if none temp file is used
# e2lshPath - path to LSH bin, if not assume it's in path
#
# OUTPUT
#
#  fOutName - file name of the E2LSH program
#
#########################################################################
def e2lsh(input, fOutName='', fInName='', e2lshPath='') :
    print 'lsh.e2lsh, not implemented yet'

    # fOutName


    # fInName
    if fInName == '':
        fIn = tempfile.NamedTemporaryFile('w')
        fInName = fIn.name
    else :
        fIn = open(fInName,'w')

    # numpy input to file
    for l in range(N.shape(input)[0]) :
        input[l].tofile(fIn,sep=' ')
        fIn.write('\n')
    fIn.close()

    # e2lsh lsh path
    cmd = os.path.join(e2lshPath,'lsh')

    # call e2lsh

    return


#########################################################################
# help menu
#########################################################################
def die_with_usage() :
    print 'lsh.py'
    print 'should be used as a library'
    sys.exit(0)


#########################################################################
# MAIN
# does nothing...
#########################################################################
if __name__ == '__main__' :

    if len(sys.argv) <= 1 :
        die_with_usage()

    input = N.zeros([3,2])
    #e2lsh(input,'out.txt','in.txt')
    print command_with_output('ls')
    
