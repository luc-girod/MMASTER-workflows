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
tdir=$odir
# tdir=/net/lagringshotell/uio/lagringshotell/geofag/icemass/icemass-data/Aster/XX
###################
mkdir -p PROCESSED_INITIAL
mkdir -p PROCESSED_FINAL
###################

for f in $(ls -d AST*/); do
    echo $f
    # be sure to change the utm zone
    WorkFlowASTER.sh -s ${f%%/} -z "$utm" -a -i 1 > ${f%%/}.log
    PostProcessMicMac.sh -z "$utm" -d ${f%%/}
    mkdir -p ${f%%/}_Success
    mv -v  ${f}zips/* ${f%%/}_Success
    rm -rf ${f}*

done

