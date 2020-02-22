#!/bin/bash
#export PATH=$PATH:/mn/moulin/project/Software/culture3d/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/cmake

echo $1
echo $2
echo $3
name=$1
UTM=$2
nameWaterMask=$3

echo "For scene " $name
echo "Using water mask " $nameWaterMask

#Fixed symboles

Nx="_3N.xml"
Bx="_3B.xml"
Nt="_3N.tif"
Bt="_3B.tif"
Bcor="_3B.tif_corrected.tif"
RPC="RPC_"

rm -R MEC-WaterMask
rm -R TA

#Make Files to lookup position of sea
mm3d Malt Ortho ".*$name(|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" ImOrtho="FalseColor_$name.tif" MOri=GRID ZMoy=2500 ZInc=2500 ZoomF=1 ZoomI=1 ResolTerrain=30 NbVI=2 EZA=1 DefCor=0 Regul=0.1 DirMEC=MEC-WaterMask DoMEC=0 

#GDAL MASKING

#read the UTM coordinate of UL corner and size of image
xminUTM=$(grep OriginePlani MEC-WaterMask/Z_Num5_DeZoom1_STD-MALT.xml| grep -o -P '>\d+'|grep -o -P '\d+')
ymaxUTM=$(grep OriginePlani MEC-WaterMask/Z_Num5_DeZoom1_STD-MALT.xml| grep -o -P '\d+<'|grep -o -P '\d+')
szx=$(grep NombrePixels MEC-WaterMask/Z_Num5_DeZoom1_STD-MALT.xml| grep -o -P '>\d+'|grep -o -P '\d+')
szy=$(grep NombrePixels MEC-WaterMask/Z_Num5_DeZoom1_STD-MALT.xml| grep -o -P '\d+<'|grep -o -P '\d+')

#Computing min/max coordinates in UTM
xmaxUTM=$(echo $xminUTM+30*$szx|bc)
yminUTM=$(echo $ymaxUTM-30*$szy|bc)
echo $xminUTM $yminUTM > Corner1.txt
echo $xmaxUTM $ymaxUTM > Corner2.txt

#Conversion of these coordinates in lat long
cs2cs +proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs +to +proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs -f "%.6f" Corner1.txt > Corner1Deg.txt
cs2cs +proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs +to +proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs -f "%.6f" Corner2.txt > Corner2Deg.txt
minDeg=$(grep -P '\d+' Corner1Deg.txt)
x1Deg=$(echo $minDeg | cut -d " " -f1 )
y1Deg=$(echo $minDeg | cut -d " " -f2 )
maxDeg=$(grep -P '\d+' Corner2Deg.txt)
x2Deg=$(echo $maxDeg | cut -d " " -f1 )
y2Deg=$(echo $maxDeg | cut -d " " -f2 )

echo x1Deg $x1Deg
echo x2Deg $x2Deg
echo y1Deg $y1Deg
echo y2Deg $y2Deg

#Sorting (x|y)(min|max) in latlong, add a padding of 0.5deg to keep the edge
if (( $(bc <<< "$x1Deg < $x2Deg") ));then
 xminDeg=$(echo $x1Deg-0.5 | bc)
 xmaxDeg=$(echo $x2Deg+0.5 | bc)
else
 xminDeg=$(echo $x2Deg-0.5 | bc)
 xmaxDeg=$(echo $x1Deg+0.5 | bc)
fi

if (( $(bc <<< "$y1Deg < $y2Deg") ));then
 yminDeg=$(echo $y1Deg-0.5 | bc)
 ymaxDeg=$(echo $y2Deg+0.5 | bc)
else
 yminDeg=$(echo $y2Deg-0.5 | bc)
 ymaxDeg=$(echo $y1Deg+0.5 | bc)
fi

echo xminDeg $xminDeg
echo xmaxDeg $xmaxDeg
echo yminDeg $yminDeg
echo ymaxDeg $ymaxDeg

rm Corner1.txt Corner2.txt Corner1Deg.txt Corner2Deg.txt

mkdir TA
#Clip the area of interest (using min and max from .met) and convert to UTM
ogr2ogr -t_srs "+proj=utm +zone=$UTM +ellps=WGS84 +datum=WGS84 +units=m +no_defs" -clipsrc $xminDeg $yminDeg $xmaxDeg $ymaxDeg water_zoneUTM.shp $nameWaterMask
#Rasterize data using min max from MEC-Mini/Z_Num9_DeZoom1_STD-MALT.xml
gdal_rasterize -ot Byte -i -burn 255 -of GTiff -tr 30 30 -te $xminUTM $yminUTM $xmaxUTM $ymaxUTM -l water_zoneUTM water_zoneUTM.shp TA/TA_LeChantier_Masq_ini.tif
rm water_zoneUTM.shp
rm water_zoneUTM.shx
rm water_zoneUTM.prj
rm water_zoneUTM.dbf
convert TA/TA_LeChantier_Masq_ini.tif -morphology Dilate Octagon:15 TA/TA_LeChantier_Masq.tif

rm TA/TA_LeChantier_Masq_ini.tif

cp MEC-WaterMask/Z_Num5_DeZoom1_STD-MALT.xml TA/TA_LeChantier_Masq.xml
cp MEC-WaterMask/Z_Num5_DeZoom1_STD-MALT.xml TA/TA_LeChantier.xml
