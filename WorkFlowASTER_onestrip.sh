# Put all the L1A zip files in a folder 'AST_L1A_MyStrip', then, from the folder above, call something like: 
# WorkFlowASTER_onestrip.sh -s AST_L1A_MyStrip -z "4 +north"
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
utm_set=0
# add default values for ZoomF, RESTERR, CorThr, water_mask and NoCorDEM
ZoomF=1
RESTERR=30
CorThr=0.7
SzW=5
water_mask=false
do_ply=false
NoCorDEM=false

while getopts "s:z:c:q:wnf:t:yh" opt; do
  case $opt in
    h)
      echo "Run the second step in the MMASTER processing chain."
      echo "usage: WorkFlowASTER_onestrip.sh -s SCENENAME -z 'UTMZONE' -f ZOOMF -t RESTERR -w false -h"
      echo "    -s SCENENAME: Folder where zips of stips are located."
      echo "    -z UTMZONE  : UTM Zone of area of interest. Takes form 'NN +north(south)'"
      echo "    -c CorThr   : Correlation Threshold for estimates of Z min and max (optional, default : 0.7)"
      echo "    -q SzW      : Size of the correlation window in the last step (optional, default : 4, mean 9*9)"
      echo "    -w mask     : Mask large water areas."
      echo "    -n NoCorDEM : Compute DEM with the uncorrected 3B image (computing with correction as well)"
      echo "    -f ZOOMF    : Run with different final resolution   (optional; default: 1)"
      echo "    -t RESTERR  : Run with different terrain resolution (optional; default: 30)"
      echo "    -y do_ply   : Write point cloud (DEM drapped with ortho in ply)"
      echo "    -h          : displays this message and exits."
      echo " "
      exit 0
      ;;
    n)
      NoCorDEM=$OPTARG
      ;;
    y)
      do_ply=$OPTARG
      ;;    
    s)
      name=$OPTARG
      scene_set=1
      ;;
    z)
      UTM=$OPTARG
      utm_set=1
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
      water_mask=$OPTARG
      ;;
    f)
      ZoomF=$OPTARG
      ;;
    t)
      RESTERR=$OPTARG
      ;;
    \?)
      echo "RunMicMacAster.sh: Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "RunMicMacAster.sh: Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

#Variable symboles
echo $name
echo $UTM
cd $name


# unziping data and archiving files
mkdir RawData
mkdir zips
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
mm3d Satelib ASTERStrip2MM AST_L1A.* $name
cd ..

mm3d SateLib Aster2Grid $name$Bx 20 "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" HMin=-500 HMax=9000 expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid $name$Nx 20 "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" HMin=-500 HMax=9000 expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid "FalseColor_$name.xml" 20 "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" HMin=-500 HMax=9000 expDIMAP=1 expGrid=1

mm3d Malt Ortho ".*$name(|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" MOri=GRID ZMoy=2500 ZInc=2500 ZoomF=8 ZoomI=32 ResolTerrain=30 NbVI=2 EZA=1 Regul=0.1 DefCor=$CorThr DoOrtho=0 DirMEC=MEC-Mini

gdalinfo -nomd -norat -noct -nofl -stats MEC-Mini/Z_Num6_DeZoom8_STD-MALT.tif > gdalinfo.txt
deminfo=$(grep -P 'Minimum+' gdalinfo.txt)
Min=$(echo $deminfo | cut -d, -f1 | tr -d ' ' | tr -d 'Minimum=' | xargs printf "%.0f")
Max=$(echo $deminfo | cut -d, -f2 | tr -d ' ' | tr -d 'Maximum=' | xargs printf "%.0f")

echo Min=$Min
echo Max=$Max

#Filter obvious error in min/max (limit to earth min/max)
Min=$((($Min)<-420 ? -420 : $Min))
Max=$((($Max)>8850 ? 8850 : $Max))
#next 2 lines is basically if the auto min/max function failed / DEM is really bad, happen if a lot of sea or a lot of clouds
Min=$((($Min)>8850 ? -420 : $Min))
Max=$((($Max)<-420 ? 8850 : $Max))
#From min/max, compute the nb of grids needed in Z and the values for ZMoy and Zinc
DE=$(echo $Max - $Min| bc )
NbLvl=$(echo $DE/200| bc )
NbLvl=$((($NbLvl)<10 ? 10 : $NbLvl))
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
mm3d SateLib Aster2Grid $name$Bx $NbLvl "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" HMin=$Min HMax=$Max expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid $name$Nx $NbLvl "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" HMin=$Min HMax=$Max expDIMAP=1 expGrid=1
mm3d SateLib Aster2Grid "FalseColor_$name.xml" $NbLvl "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" HMin=$Min HMax=$Max expDIMAP=1 expGrid=1

mm3d MMTestOrient $name$Bt $name$Nt GRIBin PB=1 MOri=GRID ZoomF=1 ZInc=$Inc ZMoy=$Mean



# if we want to compute the uncorrected DEM
if [ "$NoCorDEM" = true ]; then #check variable name!
mm3d Malt Ortho ".*$name(|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" ImOrtho="FalseColor_$name.tif" MOri=GRID ZInc=$Inc ZMoy=$Mean ZoomF=1 ZoomI=32 ResolTerrain=30 NbVI=2 EZA=1 DefCor=0 Regul=0.1 ResolOrtho=2 DirMEC=MEC-NoCor
fi

#Applying correction to the 3B image
mm3d SateLib ApplyParralaxCor $name$Bt GeoI-Px/Px2_Num16_DeZoom1_Geom-Im.tif FitASTER=1 ExportFitASTER=1
mkdir ImOrig
mv $name$Bt ImOrig/$name$Bt
mv $name$Bcor $name$Bt

# if we're using a water mask, we run that here.
if [ "$water_mask" = true ]; then #check variable name!
    cp $MMSCRIPTS_HOME/WorkFlow_WaterMask.sh .
    bash WorkFlow_WaterMask.sh $name $UTM
fi

# Correlation with corrected image
#mm3d Malt Ortho ".*$name(|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" ImOrtho="FalseColor_$name.tif" MOri=GRID ZInc=$Inc ZMoy=$Mean ZoomF=$ZoomF ZoomI=32 ResolTerrain=$RESTERR NbVI=2 EZA=1 DefCor=0 Regul=0.1 ResolOrtho=2 SzW=$SzW ZPas=0.1

mm3d Malt Ortho ".*$name(|_3N|_3B).tif" GRIBin ImMNT="$name(_3N|_3B).tif" ImOrtho="FalseColor_$name.tif" MOri=GRID ZInc=$Inc ZMoy=$Mean ZoomF=$ZoomF ZoomI=32 ResolTerrain=10 NbVI=2 EZA=1 DefCor=0 Regul=0.1 ResolOrtho=1  SzW=$SzW
mm3d Tawny Ortho-MEC-Malt/ RadiomEgal=0

if [ "$do_ply" = true ]; then
    mm3d Nuage2Ply MEC-Malt/NuageImProf_STD-MALT_Etape_9.xml Out=$name.ply Attr=Ortho-MEC-Malt/Orthophotomosaic.tif
fi

cd MEC-Malt
mv Correl_STD-MALT_Num_8.tif Correl_STD-MALT_Num_8_FullRes.tif
cp Z_Num9_DeZoom1_STD-MALT.tfw Correl_STD-MALT_Num_8_FullRes.tfw
gdal_translate -tr $RESTERR $RESTERR -r cubicspline -a_srs "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" Correl_STD-MALT_Num_8_FullRes.tif Correl_STD-MALT_Num_8.tif

mv AutoMask_STD-MALT_Num_8.tif AutoMask_STD-MALT_Num_8_FullRes.tif
cp Z_Num9_DeZoom1_STD-MALT.tfw AutoMask_STD-MALT_Num_8_FullRes.tfw
gdal_translate -tr $RESTERR $RESTERR -r cubicspline -a_srs "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" AutoMask_STD-MALT_Num_8_FullRes.tif AutoMask_STD-MALT_Num_8.tif

mosaic_micmac_tiles.py -filename 'Z_Num9_DeZoom1_STD-MALT' 
mv Z_Num9_DeZoom1_STD-MALT.tif Z_Num9_DeZoom1_STD-MALT_FullRes.tif
mv Z_Num9_DeZoom1_STD-MALT.tfw Z_Num9_DeZoom1_STD-MALT_FullRes.tfw
mv Z_Num9_DeZoom1_STD-MALT.xml Z_Num9_DeZoom1_STD-MALT_FullRes.xml

gdal_translate -tr $RESTERR $RESTERR -r cubicspline -a_srs "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" -co TFW=YES Z_Num9_DeZoom1_STD-MALT_FullRes.tif Z_Num9_DeZoom1_STD-MALT.tif
cd ..

# computing orbit angles on DEM
mm3d SateLib ASTERProjAngle MEC-Malt/Z_Num9_DeZoom1_STD-MALT MEC-Malt/AutoMask_STD-MALT_Num_8.tif $name$N
cp MEC-Malt/Z_Num9_DeZoom1_STD-MALT.tfw TrackAngleMap_nonGT.tfw
mv TrackAngleMap.tif TrackAngleMap_nonGT.tif
gdal_translate -a_srs "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" -a_nodata 0 TrackAngleMap_nonGT.tif TrackAngleMap.tif
rm TrackAngleMap_nonGT*


cd Ortho-MEC-Malt
# if there are no tiles, we have nothing to do.
# not sure if we want to hard-code that the tiles will always be Nx1?
if [ -f Orthophotomosaic_Tile_0_0.tif ]; then
    montage Orthophotomosaic_T*.tif -mode concatenate -tile 1x  Orthophotomosaic_FullRes.tif
else
    cp Orthophotomosaic.tif Orthophotomosaic_FullRes.tif
fi
mv Orthophotomosaic.tfw Orthophotomosaic_FullRes.tfw
gdal_translate -tr 15 15 -r bilinear -a_srs "+proj=utm +zone=$UTM +datum=WGS84 +units=m +no_defs" Orthophotomosaic_FullRes.tif Orthophotomosaic.tif
cd ..
