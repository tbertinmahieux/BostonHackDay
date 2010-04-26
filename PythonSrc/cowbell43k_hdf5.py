import glob
import itertools
import os

import numpy as np
import scipy as sp
import scipy.io
import tables

def matfile_to_enid(matfile):
    """Convert matfilename to an echo nest track id."""
    return os.path.split(matfile)[-1].replace('.mat', '').upper()

MAXSTRLEN = 1024
class ENMetadata(tables.IsDescription):
    artist = tables.StringCol(MAXSTRLEN)
    bitrate = tables.IntCol()
    duration = tables.FloatCol()
    genre = tables.StringCol(MAXSTRLEN)
    id = tables.StringCol(MAXSTRLEN)
    md5 = tables.StringCol(32)
    release = tables.StringCol(MAXSTRLEN)
    samplerate = tables.IntCol()
    status = tables.StringCol(MAXSTRLEN)
    title = tables.StringCol(MAXSTRLEN)
    analysispath = tables.StringCol(MAXSTRLEN)

def matfile_to_group(matfile, h5, analysisgroup):
    M = sp.io.loadmat(matfile)['M'][0][0]
    enid = matfile_to_enid(matfile)

    # Can't store all in one big group, so we'll break it up.
    key = enid[:4]
    try:
        group = h5.getNode(analysisgroup, key)
    except tables.NoSuchNodeError:
        group = h5.createGroup(analysisgroup, key)

    entrack = h5.createGroup(group, enid)
    for key, val in M.__dict__.iteritems():
        if not key.startswith('_'):
            h5.createArray(entrack, key, val)

    analysispath = entrack._v_pathname
    return analysispath

def metafile_to_row(metafile, analysispath, row):
    ENC = 'UTF-8'
    dpwemeta = sp.io.loadmat(metafile)['data']
    for key, val in dpwemeta:
        row[key[0][:-1].encode(ENC)] = val[0].encode(ENC)
    row['analysispath'] = analysispath
    return row

def copy_mat_data_to_h5(h5filename, dpwedatadir):
    matdir = os.path.join(dpwedatadir, 'mat')
    matfiles = glob.glob('%s/*/*/*.mat' % matdir)
    metadir = os.path.join(dpwedatadir, 'meta')
    metafiles = [x.replace(matdir, metadir) for x in matfiles]

    h5 = tables.openFile(h5filename, 'w')
    analysisgroup = h5.createGroup(h5.root, 'analysis')
    metatable = h5.createTable(h5.root, 'metadata', ENMetadata)
    for matfile, metafile in itertools.izip(matfiles, metafiles):
        try:
            analysispath = matfile_to_group(matfile, h5, analysisgroup)
        except Exception, e:
            print e
            continue
        metafile_to_row(metafile, analysispath, metatable.row).append()
    metatable.flush()
    h5.close()

copy_mat_data_to_h5('/home/ronw/projects/bostonmusichackday/cowbell43k.h5',
                    '/home/ronw/projects/bostonmusichackday/mnt/')
        
