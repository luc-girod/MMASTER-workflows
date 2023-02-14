#!/bin/bash
#module purge
#module load bobtools
#module load seracmicmac

# export PATH=/uio/kant/geo-natg-u1/robertwm/scripts/dev/MMASTER-workflows/:$PATH
###################
utm=$1
tdir=$2

shift 2
###################
odir=$(pwd)
# tdir=/net/lagringshotell/uio/lagringshotell/geofag/icemass/icemass-data/Aster/XX
###################
mkdir -p $tdir/PROCESSED_INITIAL
mkdir -p $tdir/PROCESSED_FINAL
###################
for dir in "$@"; do
    cd $dir
    for f in $(ls -d AST*/); do
        # be sure to change the utm zone
        WorkFlowASTER.sh -s ${f%%/} -z "$utm" -a -i 1 -w 1 > ${f%%/}.log
    done
    # be sure to change the utm zone
    PostProcessMicMac.sh -z "$utm"
    CleanMicMac.sh
    cd PROCESSED_INITIAL
    for d in $(ls -d AST*/); do
        tar -cvzf ${d%/}.tar.gz $d
        rm -rv $d
    done
    cd ..
    mv -v PROCESSED_INITIAL/* $tdir/PROCESSED_INITIAL
    mv -v PROCESSED_FINAL/* $tdir/PROCESSED_FINAL
    rmdir PROCESSED_INITIAL PROCESSED_FINAL
    cd $odir
    mv -v $dir done.$dir
done
