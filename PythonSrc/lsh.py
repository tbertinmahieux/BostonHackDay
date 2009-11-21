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
# lsh_model(input,fModelName,fInName,lshDir)
#
# Write an input numpy matrix to a file and apply the E2LSH algorithm.
# Here we just create the model! what E2LSH call the params
# We then use the lsh_query function to use this model.
# E2LSH bin must be in the path, or provided as an input.
# output file path is returned.
#
# INPUT
#
#      input - numpy matrix, one example per line, all examples same size
# fModelName - E2LSH model file name, default='lsh_model.txt'
#    fInName - file name for the E2LSH input, if none temp file is used
#     lshDir - path to LSH bin, if not assume it's in path
#
# OUTPUT
#
# fModelName - file name of the E2LSH program
#
#########################################################################
def lsh_model(input, fModelName='', fInName='', lshDir='') :

    # fModelName
    if fModelName == '' :
        fModelName = 'lsh_model.txt'

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

    # lsh_computeParams path
    cmd = os.path.join(lshDir,'lsh_computeParams')
    cmd = cmd + ' R ' + fInName + ' > ' + fOutName

    # call 
    result = command_with_output(cmd)

    # return ouput file of lsh
    return fOutName



#########################################################################
# lsh_query(queries,fModelName,fInName,lshDir)
#
# Query a given model.
# We then use the lsh_query function to use this model.
# E2LSH bin must be in the path, or provided as an input.
#
# INPUT
#
#      queries - numpy matrix, one example per line
#      fInputs - file name containing input data, fInName from lsh_model
#   fModelName - E2LSH model file name
# fQueriesName - file name for the E2LSH input, if none temp file is used
#       lshDir - path to LSH bin, if not assume it's in path
#         fRes - filename for results file, default='lsh_results.txt'
#
# OUTPUT
#
#         fRes - filename for results file (from LSH program)
#
#########################################################################
def lsh_query(queries, fInputs, fModelName, fQueriesName='',
              lshDir='', fRes='lsh_results.txt') :

    # fQueriesName
    if fInName == '':
        fIn = tempfile.NamedTemporaryFile('w')
        fInName = fIn.name
    else :
        fIn = open(fInName,'w')

    # numpy input to file
    for l in range(N.shape(queries)[0]) :
        queries[l].tofile(fIn,sep=' ')
        fIn.write('\n')
    fIn.close()

    # lsh_fromParams path
    cmd = os.path.join(lshDir,'lsh_fromParams')
    cmd = cmd + ' ' + fInputs + ' ' + fModelName
    cmd = cmd + ' > ' + fRes

    # call 
    result = command_with_output(cmd)

    # return result file name
    return fRes


 

#########################################################################
# help menu
#########################################################################
def die_with_usage() :
    print "lsh.py"
    print "should be used as a library"
    print "usage:"
    print "lsh_model(data,'lsh_model.txt','data.txt','/lsh/bin/')"
    print "lsh_query(queries,'data.txt','lsh_model.txt','queries.txt','/lsh/bin/','lsh_results.txt')"
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
    #print command_with_output('ls')
    
