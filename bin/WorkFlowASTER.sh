#!/bin/bash
# Put all the L1A zip files in a folder 'AST_L1A_...', then, from the folder above, call something like: 
# WorkFlowASTER.sh -s AST_L1A_MyStrip -z "4 +north"
# extra options :  -t 30 -n false -c 0.7 -w false -f 1

#Fixed symboles
N="_3N"
B="_3B"
Nx="_3N.xml"
Bx="_3B.xml"
Nt="_3N.tif"
Bt="_3B.tif"
Bcor="_3B.tif_corrected.tif"
RPC="RPC_"
scene_set=0
proj_set=0
# add default values for ZoomF, RESTERR, CorThr, water_mask and NoCorDEM
ZoomF=1
RESTERR=30
CorThr=0.7
SzW=5
nameWaterMask=false
do_ply=false
do_angle=false
NoCorDEM=false
fitVersion=1

while getopts "s:z:o:c:q:wnf:t:y:ai:h" opt; do
  case $opt in
    h)
      echo "Run the second step in the MMASTER processing chain."
      echo "basic usage: WorkFlowASTER.sh -s SCENENAME -z 'UTMZONE' -f ZOOMF -t RESTERR -w MASK -a"
      echo "    -s SCENENAME   : Folder where zips of stips are located."
      echo "    -z UTMZONE     : UTM Zone of area of interest. Takes form 'NN +north(south)'. One of -z or -o must be set."
      echo "    -o PolarStereo : Use polar stereo (option N for north EPSG:3411 or S for south EPSG:3412)"
      echo "    -c CorThr      : Correlation Threshold for estimates of Z min and max (optional, default : 0.7)"
      echo "    -q SzW         : Size of the correlation window in the last step (optional, default : 4, mean 9*9)"
      echo "    -w mask        : Name of shapefile to skip masked areas (usually water, this is optional, default : none)."
      echo "    -n NoCorDEM    : Compute DEM with the uncorrected 3B image (computing with correction as well)"
      echo "    -f ZOOMF       : Run with different final resolution   (optional; default: 1)"
      echo "    -t RESTERR     : Run with different terrain resolution (optional; default: 30)"
      echo "    -y do_ply      : Write point cloud (DEM drapped with ortho in ply)"
      echo "    -a do_angle    : Compute track angle along orbit"
      echo "    -i fitVersion  : Version of Cross-track FitASTER to be used (default 1, 1 availiable)"
      echo "    -h             : displays this message and exits."
      echo " "
      exit 0
      ;;
    n)
      NoCorDEM=$OPTARG
      ;;
    a)
      do_angle=true
      ;;  
    y)
      do_ply=$OPTARG
      ;;    
    s)
      name=$OPTARG
      scene_set=1
      ;;
    z)
      proj="+proj=utm +zone=$OPTARG +datum=WGS84 +units=m +no_defs"
      proj_set=1
      echo "Projection set to $proj"
      ;;
    o)
      if [ "$OPTARG" = N ]; then
        proj="+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
      elif [ "$OPTARG" = S ]; then
        proj="+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
      fi
      proj_set=1
      echo "Projection set to $proj"
      ;;
    c)
      CorThr=$OPTARG
      echo "CorThr set to $CorThr"
      ;;  
    q)
      SzW=$OPTARG
      echo "SzW set to $SzW"
      ;;
    w)
      echo "Water mask selected: " $OPTARG
	    nameWaterMask=$OPTARG
      ;;
    f)
      ZoomF=$OPTARG
      ;;
    i)
      echo "ASTER Fit Version: " $OPTARG
	    fitVersion=$OPTARG
      ;;
    t)
      RESTERR=$OPTARG
      ;;
    \?)
      echo "WorkFlowASTER.sh: Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "WorkFlowASTER.sh: Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ "$proj_set" = "0" ]; then
  echo "WorkFlowASTER.sh: projection must be set using either -z or -o flags." >&2
  exit 1
fi

#Variable symboles
echo $name
echo $proj
cd $name
pwd

# unziping data and archiving files
mkdir RawData
mkdir zips
nscenes=$(ls *.zip | wc -l)
find ./ -maxdepth 1 -name "*.zip" | while read filename; do
        f=$(basename "$filename")
        unzip $f -d "RawData"
        mv "$f" "zips"
done  

echo "Moved and extracted zip files"

find ./ -maxdepth 1 -name "*.met" | while read filename; do
    f=$(basename "$filename")
    mv "$f" "zips"
done  

echo "Moved met files"

pwd
cd RawData
pwd
if [ "$nscenes" -gt "1" ]; then
    mm3d Satelib ASTERStrip2MM AST_L1A.* $name
else
    for f in $(ls *.*); do 
        split=($(echo $f | sed 's/\./ /')); 
        mv -v $f ${split[0]:0:25}.${split[1]}; 
    done
    mm3d SateLib ASTERGT2MM $name
fi
cd ..

mm3d SateLib Aster2Grid $name$Bx 20 "$proj" HMin=-500 HMax=9000 expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid $name$Nx 20 "$proj" HMin=-500 HMax=9000 expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid "FalseColor_$name.xml" 20 "$proj" HMin=-500 HMax=9000 expDIMAP=1 expGrid=1

mm3d Malt Ortho ".*$name(()|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" MOri=GRID ZMoy=2500 ZInc=2500 ZoomF=8 ZoomI=32 ResolTerrain=30 NbVI=2 EZA=1 Regul=0.1 DefCor=$CorThr DoOrtho=0 DirMEC=MEC-Mini

gdalinfo -nomd -norat -noct -nofl -stats MEC-Mini/Z_Num6_DeZoom8_STD-MALT.tif > gdalinfo.txt
deminfo=$(grep 'Minimum' gdalinfo.txt)
Min=$(echo $deminfo | cut -d, -f1 | tr -d ' ' | tr -d 'Minimum=' | xargs printf "%.0f")
Max=$(echo $deminfo | cut -d, -f2 | tr -d ' ' | tr -d 'Maximum=' | xargs printf "%.0f")

echo Min=$Min
echo Max=$Max

#Filter obvious error in min/max (limit to earth min/max)
if [ $Min -lt -420 ]; then Min=-420; fi # changed from Min=$((($Min)<-420 ? -420 : $Min)) because it breaks on Mac OS
if [ $Max -gt 8850 ]; then Max=8850; fi
#next 2 lines is basically if the auto min/max function failed / DEM is really bad, happen if a lot of sea or a lot of clouds
if [ $Min -gt 8850 ]; then Min=-420; fi
if [ $Max -lt -420 ]; then Max=8850; fi
#From min/max, compute the nb of grids needed in Z and the values for ZMoy and Zinc
DE=$(echo $Max - $Min| bc )
NbLvl=$(echo $DE/200| bc )
if [ $NbLvl -lt 10 ]; then NbLvl=10; fi
Mean=$(echo $Max + $Min| bc )
Mean=$(echo $Mean/2| bc )
Inc=$(echo $Max - $Mean| bc | xargs printf "%.0f")
echo Min=$Min
echo Max=$Max
echo NbLvl=$NbLvl
echo Mean=$Mean
echo Inc=$Inc
echo Min Max NbLvl Mean Inc >> Stats.txt
echo $Min $Max $NbLvl $Mean $Inc >> Stats.txt

#Re compute RPCs with updated min/max
mm3d SateLib Aster2Grid $name$Bx $NbLvl "$proj" HMin=$Min HMax=$Max expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid $name$Nx $NbLvl "$proj" HMin=$Min HMax=$Max expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid "FalseColor_$name.xml" $NbLvl "$proj" HMin=$Min HMax=$Max expDIMAP=1 expGrid=1

mm3d MMTestOrient $name$Bt $name$Nt GRIBin PB=1 MOri=GRID ZoomF=1 ZInc=$Inc ZMoy=$Mean

# if we want to compute the uncorrected DEM
if [ "$NoCorDEM" = true ]; then #check variable name!
mm3d Malt Ortho ".*$name(()|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" ImOrtho="FalseColor_$name.tif" MOri=GRID ZInc=$Inc ZMoy=$Mean ZoomF=1 ZoomI=32 ResolTerrain=30 NbVI=2 EZA=1 DefCor=0 Regul=0.1 ResolOrtho=2 DirMEC=MEC-NoCor
fi

#Applying correction to the 3B image
mm3d SateLib ApplyParralaxCor $name$Bt GeoI-Px/Px2_Num16_DeZoom1_Geom-Im.tif FitASTER=$fitVersion 
mkdir ImOrig
mv $name$Bt ImOrig/$name$Bt
mv $name$Bcor $name$Bt

# if we're using a water mask, we run that here.
if [ "$nameWaterMask" != false ]; then #check variable name!
    WorkFlow_WaterMask.sh $name "$UTM" $nameWaterMask # think this needs updating with the new projection
fi

# Correlation with corrected image
#mm3d Malt Ortho ".*$name(|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" ImOrtho="FalseColor_$name.tif" MOri=GRID ZInc=$Inc ZMoy=$Mean ZoomF=$ZoomF ZoomI=32 ResolTerrain=$RESTERR NbVI=2 EZA=1 DefCor=0 Regul=0.1 ResolOrtho=2 SzW=$SzW ZPas=0.1
# do we need to add something for a water mask above?
mm3d Malt Ortho ".*$name(()|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" ImOrtho="FalseColor_$name.tif" MOri=GRID ZInc=$Inc ZMoy=$Mean ZoomF=$ZoomF ZoomI=32 ResolTerrain=10 NbVI=2 EZA=1 DefCor=0 Regul=0.1 ResolOrtho=1  SzW=$SzW
mm3d Tawny Ortho-MEC-Malt/ RadiomEgal=0

#Check if things need to be tiled
echo 
echo "Trying to see if tiling needs to be done"
echo 
cd MEC-Malt

if [[ $(ls Z_Num*_DeZoom*_STD-MALT_Tile_*.tif) ]]; then

finalDEMs=($(ls Z_Num*_DeZoom*_STD-MALT_Tile_*.tif))
finalMasks=($(ls AutoMask_STD-MALT_Num_*_Tile_*.tif))
finalcors=($(ls Correl_STD-MALT_Num*_Tile_*.tif))
DEMind=$((${#finalDEMs[@]}-1))
Maskind=$((${#finalMasks[@]}-1))
corind=$((${#finalcors[@]}-1))
lastDEM=${finalDEMs[DEMind]}
lastMask=${finalMasks[Maskind]}
lastcor=${finalcors[corind]}
laststr="${lastDEM%.*}"
Maskstr="${lastMask%.*}"
corrstr="${lastcor%.*}"

if [ -f $lastcor ]; then
	mosaic_micmac_tiles.py -filename '${lastcor:0:${#lastcor}-13}'
fi
if [ -f $lastMask ]; then
	mosaic_micmac_tiles.py -filename '${$lastMASK:0:${#$lastMASK}-13}'
fi
if [ -f $lastDEM ]; then
	mosaic_micmac_tiles.py -filename '${$lastDEM:0:${#$lastDEM}-13}'
fi

rm *Tile*.tif
echo 
	echo "Tiling done"
echo 
else
echo 
	echo "No tiling to be done"
echo 
fi
cd ..


# find the max value of Num (with standard ZoomF, 8 and 9)
cd MEC-Malt
finalDEMs=($(ls Z_Num*_DeZoom*_STD-MALT.tif))
finalMasks=($(ls AutoMask_STD-MALT_Num_*.tif))
finalcors=($(ls Correl_STD-MALT_Num*.tif))
finalNuages=($(ls NuageImProf_STD-MALT_Etape_*.xml))
DEMind=$((${#finalDEMs[@]}-1))
Maskind=$((${#finalMasks[@]}-1))
corind=$((${#finalcors[@]}-1))
Nuageind=$((${#finalNuages[@]}-1))
lastDEM=${finalDEMs[DEMind]}
lastMask=${finalMasks[Maskind]}
lastcor=${finalcors[corind]}
lastNuage=${finalNuages[Nuageind]}
laststr="${lastDEM%.*}"
Maskstr="${lastMask%.*}"
corrstr="${lastcor%.*}"

# make geotif files out the tif+tfw
mv $lastcor Correl_FullRes.tif
cp $laststr.tfw Correl_FullRes.tfw
gdal_translate -tr $RESTERR $RESTERR -a_srs "$proj" Correl_FullRes.tif $lastcor


mv $lastMask AutoMask_FullRes.tif
cp $laststr.tfw AutoMask_FullRes.tfw
gdal_translate -tr $RESTERR $RESTERR -a_srs "$proj" AutoMask_FullRes.tif $lastMask


mv $lastDEM Z_FullRes.tif
mv $laststr.tfw Z_FullRes.tfw
mv $laststr.xml Z_FullRes.xml

gdal_translate -tr $RESTERR $RESTERR -r cubicspline -a_srs "$proj" -co TFW=YES Z_FullRes.tif $lastDEM
cd ..

if [ "$do_ply" = true ]; then
    mm3d Nuage2Ply MEC-Malt/$lastNuage Out=$name.ply Attr=Ortho-MEC-Malt/Orthophotomosaic.tif
fi

if [ "$do_angle" = true ]; then
	# computing orbit angles on DEM
	mm3d SateLib ASTERProjAngle MEC-Malt/$laststr MEC-Malt/$lastMask $name$N
	if [ -f TrackAngleMap_3N_Tile_0_0.tif ]; then
		mosaic_micmac_tiles.py -filename 'TrackAngleMap_3N'
	fi
	cp MEC-Malt/Z_Num9_DeZoom1_STD-MALT.tfw TrackAngleMap_nonGT.tfw
	mv TrackAngleMap.tif TrackAngleMap_nonGT.tif
	gdal_translate -a_srs "$proj" -a_nodata 0 TrackAngleMap_nonGT.tif TrackAngleMap_3N.tif
	rm TrackAngleMap_nonGT*
	mm3d SateLib ASTERProjAngle MEC-Malt/$laststr MEC-Malt/$lastMask $name$B
	if [ -f TrackAngleMap_3B_Tile_0_0.tif ]; then
		mosaic_micmac_tiles.py -filename 'TrackAngleMap_3B'
	fi
	cp MEC-Malt/$laststr.tfw TrackAngleMap_nonGT.tfw
	mv TrackAngleMap.tif TrackAngleMap_nonGT.tif
	gdal_translate -a_srs "$proj" -a_nodata 0 TrackAngleMap_nonGT.tif TrackAngleMap_3B.tif
	rm TrackAngleMap_nonGT*
fi

cd Ortho-MEC-Malt
# if there are no tiles, we have nothing to do.
# not sure if we want to hard-code that the tiles will always be Nx1?
if [ -f Orthophotomosaic_Tile_0_0.tif ]; then
	mosaic_micmac_tiles.py -filename 'Orthophotomosaic'
fi
mv Orthophotomosaic.tif Orthophotomosaic_FullRes.tif
mv Orthophotomosaic.tfw Orthophotomosaic_FullRes.tfw
gdal_translate -tr 15 15 -r bilinear -a_srs "$proj" Orthophotomosaic_FullRes.tif Orthophotomosaic.tif
cd ..
