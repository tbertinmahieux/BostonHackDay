#!/usr/bin/env python

#########################################################################
#
# lsh.py
# wrapper around the LSH code E2LSH-0.1 from MIT
#
# Simple idea:
#   - data to temp file
#   - call command line LSH
#   - parse output file
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
    # hack just for Darwin, those lucky people
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
#     radius - search radius, see E2LSH info
# fModelName - E2LSH model file name, default='lsh_model.txt'
#    fInName - file name for the E2LSH input, if none temp file is used
#     lshDir - dir where to find bin/lsh
#
# OUTPUT
#
# fModelName - file name of the E2LSH program
#
#########################################################################
def lsh_model(input, radius, fModelName='', fInName='', lshDir='') :

    # fInName
    if fInName == '':
        fInTemp = tempfile.NamedTemporaryFile('w')
        fInName = os.path.abspath(fInTemp.name)
        fIn = fInTemp.file
    else :
        fInName = os.path.abspath(fInName)
        fIn = open(fInName,'w')

    # numpy input to file
    for l in range(input.shape[0]) :
        input[l].tofile(fIn,sep=' ')
        fIn.write('\n')
    fIn.close()

    return lsh_model_inputfile(fInName,radius,fModelName,lshDir)


def lsh_model_inputfile(inputfilename, radius, fModelName='', lshDir='') :
    """Similar to lsh_model but the input is a file, not a numpy matrix."""

    # fModelName
    if fModelName == '' :
        fModelName = 'lsh_model.txt'
    fModelName = os.path.abspath(fModelName)

    # fInName
    fInName = os.path.abspath(inputfilename)
    fIn = open(fInName,'w')

    # numpy input to file
    for l in range(input.shape[0]) :
        input[l].tofile(fIn,sep=' ')
        fIn.write('\n')
    fIn.close()

    # get to the right dir
    currdir = os.path.abspath(os.path.curdir)
    os.chdir(lshDir)

    # lsh_computeParams path
    cmd = 'bin/lsh_computeParams'
    cmd = cmd + ' ' + str(radius)
    cmd = cmd + ' ' + fInName + ' . > ' + fModelName

    # call
    print cmd
    result = command_with_output(cmd)

    # get back to currdir
    os.chdir(currdir)

    # return ouput file of lsh
    return fModelName



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
#       lshDir - dir where to find bin/lsh
#         fRes - filename for results file, default='lsh_results.txt'
#
# OUTPUT
#
#         fRes - filename for results file (from LSH program)
#
# USAGE
#
#     res = lsh_parse_result('lsh_results.txt')
#     # for query 2
#     (pts,dists) = res[2]
#     # first point found and its dist from query 2
#     if len(pts) > 0 :
#        print pts[0]
#        print dists[0]
#
#########################################################################
def lsh_query(queries, fInputs, fModelName, fQueriesName='',
              lshDir='', fRes='lsh_results.txt') :

    # make sure we use absolute path names
    fInputs = os.path.abspath(fInputs)
    fModelName = os.path.abspath(fModelName)
    fRes = os.path.abspath(fRes)

    # fQueriesName
    if fQueriesName == '':
        fQueriesTemp = tempfile.NamedTemporaryFile('w')
        fQueriesName = fQueriesTemp.name
        fQueries = fQueriesTemp.file
    else :
        fQueriesName = os.path.abspath(fQueriesName)
        fQueries = open(fQueriesName,'w')

    # numpy input to file
    for l in range(N.shape(queries)[0]) :
        queries[l].tofile(fQueries,sep=' ')
        fQueries.write('\n')
    fQueries.close()

    # get to the right dir
    currdir = os.path.abspath(os.path.curdir)
    os.chdir(lshDir)

    # lsh_fromParams path
    cmd = 'bin/lsh_fromParams'
    cmd = cmd + ' ' + fInputs + ' ' + fQueriesName
    cmd = cmd + ' ' + fModelName
    cmd = cmd + ' > ' + fRes

    # call 
    result = command_with_output(cmd)

    # get back to currdir
    os.chdir(currdir)

    # return result file name
    return fRes




#########################################################################
# lsh_parse_result(fRes)
#
# Parse a result file from E2LSH into python
#
# INPUT
#
#  fRes - filename for results file
#
# OUTPUT
#
#   map - a dictionnary of queryID -> tuple(points,dists)
#         queryID goes from 0 to .... #queries - 1
#         tuple t contains t[0] - array of data points
#                          t[1] - array of distance to the query
#         len(t) is 2, len(t[0]) equals len(t[1])
#
#########################################################################
def lsh_parse_result(fRes):

    # init result map
    res_map = dict()


    # open file
    fIn = open(fRes,'r')

    # iterate on res file
    querycnt = -1
    for line in fIn.xreadlines() :

        # skip the Total time line...
        if line[:len('Total time')] == 'Total time' :
            continue
        # reachin Query point line, init stuff...
        if line[:len('Query point')] == 'Query point' :
            querycnt = querycnt + 1
            points = N.array([])
            dists = N.array([])
            continue
        if line[:len('Mean query')] == 'Mean query' :
            continue
        # reading one query res
        [p,d] = line.strip().split('\t')
        # save it to arrays
        points = N.append(points,int(p))
        [junk,d] = d.split(':')
        dists = N.append(dists,float(d))
        # save results to map... slow repetitive way but works
        res_map[querycnt] = (points,dists)

    # close file
    fIn.close()

    # return map
    return res_map




#########################################################################
# help menu
#########################################################################
def die_with_usage() :
    print "lsh.py"
    print "should be used as a library"
    print "usage:"
    print "lsh_model(data,10,'lsh_model.txt','data.txt','/E2LSH')"
    print "lsh_query(queries,'data.txt','lsh_model.txt','queries.txt','/E2LSH','lsh_results.txt')"
    print "lsh_parse_result(fRes)"
    sys.exit(0)


#########################################################################
# MAIN
# does nothing...
#########################################################################
if __name__ == '__main__' :

    if len(sys.argv) <= 1 :
        die_with_usage()


    print 'dummy test'

    data = N.random.rand(100,13)
    queries = N.random.rand(10,13)

    lshDir = '/home/thierry/Columbia/BostonHackDay/LSH/E2LSH-0.1'

    print 'creating model...'
    print "lsh_model(data,1,'lsh_model.txt','data.txt',lshDir)"
    lsh_model(data,1,'lsh_model.txt','data.txt',lshDir)

    print 'querying model...'
    print "lsh_query(queries,'data.txt','lsh_model.txt','queries.txt',lshDir,'lsh_results.txt')"
    lsh_query(queries,'data.txt','lsh_model.txt','queries.txt',lshDir,'lsh_results.txt')

    print 'reading results...'
    print "res = lsh_parse_result('lsh_results.txt')"
    res = lsh_parse_result('lsh_results.txt')
    print 'first result: points then distances'
    print res[0][0]
    print res[0][1]
    
    print 'done'
    
