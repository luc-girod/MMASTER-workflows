#!/bin/bash
# post-process everything output from MicMac into nice files to use:
#	- masked DEM
#	- hillshade
#	- georeferenced orthophoto
#	- footprints of DEMs (can use to clip edges, for example)
# use: bash PostProcessMicMac.sh utm_zone,
#	where utm_zone has the form ""6 +north" for the projection used in processing.
proj_set=0
sub_set=0
while getopts "z:o:d:h" opt; do
  case $opt in
    h)
      echo "Post-process outputs from MMASTER into nice files to use."
      echo "usage: PostProcessMicMac.sh -z 'UTMZONE' -h"
      echo "    -z UTMZONE     : UTM Zone of area of interest. Takes form 'NN +north(south)'"
      echo "    -o PolarStereo : Use polar stereo (option N for north EPSG:3411 or S for south EPSG:3412)"
      echo "    -d SUBDIR      : Subfolder(s) where MMASTER outputs are written. If not set, looks for folders of form AST_L1A..."
      echo "    -h             : displays this message and exits."
      echo " "
      exit 0
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
    d)
      subList+=("$OPTARG")
      sub_set=1
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

if [ "$proj_set" = "0" ]; then
  echo "RunMicMacAster.sh: projection must be set using either -z or -o flags." >&2
  exit 1
fi

if [ $sub_set -eq 0 ]; then
    echo "No subdirectories specified, looking for directories of form AST_*"
    subList=$(ls -d AST_*/);
fi
    
resize_rasters () {
    image1=$1
    image2=$2
    # first, get the raster sizes and check that they're the same
    img1size=($(gdalinfo $image1 | grep 'Size is' | grep -o '[0-9]*'))
    img2size=($(gdalinfo $image2 | grep 'Size is' | grep -o '[0-9]*'))
    # if the rasters are the same size, we continue.
    if [[ "${img1size[0]}" -eq "${img2size[0]}" && "${img1size[1]}" -eq "${img2size[1]}" ]]; then
        echo "$image1 and $image2 are the same size. Exiting..."
        return 1
    fi
    # get the upper left and lower right corners of image1
    ul=$(gdalinfo $image1 | grep 'Upper Left' | grep -Eo '[+-]?[0-9]*\.[0-9]*\,\s*?[+-]?[0-9]*\.[0-9]*' )
    lr=$(gdalinfo $image1 | grep 'Lower Right' | grep -Eo '[+-]?[0-9]*\.[0-9]*\,\s*[+-]?[0-9]*\.[0-9]*' )
    # split into two arrays    
    ul_arr=($(echo $ul | tr , ' '))
    lr_arr=($(echo $lr | tr , ' '))
    echo "gdalwarp -te ${ul_arr[0]} ${lr_arr[1]} ${lr_arr[0]} ${ul_arr[1]} -ts ${img1size[@]} $image2 ${image2%.tif}_resize.tif"
    echo "Re-sizing $image2 to agree with $image1 size."
    gdalwarp -te ${ul_arr[0]} ${lr_arr[1]} ${lr_arr[0]} ${ul_arr[1]} -ts ${img1size[@]} $image2 ${image2%.tif}_resize.tif
    mv -v ${image2%.tif}_resize.tif $image2
}

# first, get the masked dems and put them into a folder creatively called MaskedDEMs
# also get the orthophotos and put them into a folder creatively called OrthoImgs
basedir=$(pwd) #get the directory that we're starting in.
#mkdir -p MaskedDEMs OrthoImgs Footprints CorrelationImgs
mkdir -p PROCESSED_FINAL
outdir=$basedir/PROCESSED_FINAL

echo "using projection $UTM"
echo "getting masked DEMs and orthoimages."

for dir in ${subList[@]}; do
	echo $dir
    dir=${dir%%/};

	#tmpstr=${dir:11:8}
	#datestr=${tmpstr:4:4}${tmpstr:0:4}
	datestr=$dir
	mkdir -p $outdir/$datestr
	
	cd $dir

	if [ -d "MEC-Malt" ]; then
		cd MEC-Malt
		# note: this could actually be hard-coded, since it's probably always 9.
        finalimgs=($(find . -regex '.*Z_Num[0-9]_DeZoom[0-9]_STD-MALT.tif' | sort)) # note - if ZoomF is not 1, DeZoom1 will not exist.
        #finalmsks=($(ls AutoMask_STD-MALT_Num*.tif))
        #finalcors=($(ls Correl_STD-MALT_Num*.tif))
        #finalmsks=($(find . -regextype posix-extended -regex '^.*AutoMask_STD-MALT_Num_[0-9]{1}\.tif'))
        #finalcors=($(find . -regextype posix-extended -regex '^.*Correl_STD-MALT_Num_[0-9]{1}\.tif'))
        finalmsks=($(find . -regex '.*AutoMask_STD-MALT_Num_[0-9]\.tif' | sort))
        finalcors=($(find . -regex '.*Correl_STD-MALT_Num_[0-9]\.tif' | sort))
		# find the last image name. ancient systems like RHEL6 don't like the -1 index.
        lastimg=${finalimgs[-1]:2}
        lastmsk=${finalmsks[-1]:2} # note that find will return ./AutoMask_...
        lastcor=${finalcors[-1]:2}
		# here's the kludge that should work on all bash systems, even
		# the ancient ones like RHEL6.
		#imgind=$((${#finalimgs[@]}-1))
		#mskind=$((${#finalmsks[@]}-1))
		#corind=$((${#finalcors[@]}-1))

		#lastimg=${finalimgs[imgind]}
		#lastmsk=${finalmsks[mskind]}
		#lastcor=${finalmsks[corind]}

		# strip the extension
        laststr="${lastimg%.*}"
        maskstr="${lastmsk%.*}"
        corrstr="${lastcor%.*}"

		# now, assign the CRS we got to the mask, dem, and apply.
		echo "Georeferencing correlation mask"
		gdal_translate -a_nodata 0 -a_srs "$proj" $lastcor $dir\_CORR.tif
		echo "Creating temporary georeferenced DEM"
		gdal_translate -a_srs "$proj" $lastimg tmp_geo.tif
		echo "Creating temporary georeferenced Mask"
		gdal_translate -a_srs "$proj" -a_nodata 0 $lastmsk tmp_msk.tif
		cd ../
		if [ -d "Ortho-MEC-Malt" ]; then 
			cd Ortho-MEC-Malt
			echo "Creating double size correlation mask for ortho"
			gdal_translate -tr 15 15 -a_srs "$proj" -a_nodata 0 ../MEC-Malt/$lastmsk tmp_mskDouble.tif
			echo "Creating temporary georeferenced ortho"
			gdal_translate -tr 15 15 -a_srs "$proj" Orthophotomosaic.tif tmp_V123.tif
            resize_rasters tmp_mskDouble.tif tmp_V123.tif
			cd ../
		fi
		cd MEC-Malt
		# apply the mask
		echo "Applying mask to georeferenced DEM"
		
		gdal_calc.py -A tmp_msk.tif -B tmp_geo.tif --outfile=$dir\_Z.tif --calc="B*(A>0)" --NoDataValue=-9999
		cp -v $dir\_Z.tif $outdir/$datestr #might be good to code orig. wd here.
		gdaldem hillshade $dir\_Z.tif $outdir/$datestr/$dir\_HS.tif
		gdal_calc.py -A $dir\_CORR.tif --outfile=$outdir/$datestr/$dir\_CORR.tif --calc="((A.astype(float)-127)/128)*100" --NoDataValue=-9999
		#cp -v $dir\_CORR.tif $outdir/$datestr #might be good to code orig. wd here.
		rm -v tmp_msk.tif tmp_geo.tif
		rm -v $dir\_Z.tif $dir\_CORR.tif
		cd ../

		if [ -d "Ortho-MEC-Malt" ]; then 			
			cd Ortho-MEC-Malt			
			gdal_calc.py -B tmp_mskDouble.tif -A tmp_V123.tif --outfile=$dir\_V123.tif --calc="((A!=255)*(A+1)+(A==255)*A)*(B>0)" --NoDataValue=0 --allBands=A
			#Expression complicated to solve real 0 values not being NoData and 255 no being +1-ed to 0
			rm -v tmp_V123.tif tmp_mskDouble.tif
			cp -v $dir\_V123.tif $outdir/$datestr
			rm -v $dir\_V123.tif
			cd ../
		fi
		cp -v TrackAngleMap*.tif $outdir/$datestr/
		#cp -v ./zips/*.zip.met $outdir/$datestr/
		find . -name '*.zip.met' -exec cp {} $outdir/$datestr/ \;
	else
		echo "No directory MEC-Malt found. Trying to find output files in this directory."

        if [ -f "Correl_STD-MALT_Num_8.tif" ]; then
    		echo "Georeferencing correlation mask"
    		gdal_translate -a_nodata 0 -a_srs "$proj" Correl_STD-MALT_Num_8.tif $dir\_CORR.tif
        else
            echo "No file Correl_STD-MALT_Num_8.tif found in $dir. Continuing."
        fi

        if [ -f "Z_Num9_DeZoom1_STD-MALT.tif" ]; then
	    	echo "Creating temporary georeferenced DEM"
    		gdal_translate -a_srs "$proj" Z_Num9_DeZoom1_STD-MALT.tif tmp_geo.tif
        else
            echo "No file Z_Num9_DeZoom1_STD-MALT.tif found in $dir. Continuing."
        fi

        if [ -f "AutoMask_STD-MALT_Num_8.tif" ]; then
    		echo "Creating temporary georeferenced Mask"
    		gdal_translate -a_srs "$proj" -a_nodata 0 AutoMask_STD-MALT_Num_8.tif tmp_msk.tif
        else
            echo "No file AutoMask_STD-MALT_Num_8.tif found in $dir. Continuing."
        fi

		echo "Creating double size correlation mask for ortho"
		gdal_translate -tr 15 15 -a_srs "$proj" -a_nodata 0 AutoMask_STD-MALT_Num_8.tif tmp_mskDouble.tif
		echo "Creating temporary georeferenced ortho"
		gdal_translate -tr 15 15 -a_srs "$proj" Orthophotomosaic.tif tmp_V123.tif
        resize_rasters tmp_mskDouble.tif tmp_V123.tif

		echo "Applying mask to georeferenced DEM"
        gdal_calc.py -A tmp_msk.tif -B tmp_geo.tif --outfile=$dir\_Z.tif --calc="B*(A>0)" --NoDataValue=-9999
		cp -v $dir\_Z.tif $outdir/$datestr #might be good to code orig. wd here.

		gdaldem hillshade $dir\_Z.tif $outdir/$datestr/$dir\_HS.tif
		gdal_calc.py -A $dir\_CORR.tif --outfile=$outdir/$datestr/$dir\_CORR.tif --calc="((A.astype(float)-127)/128)*100" --NoDataValue=-9999
		#cp -v $dir\_CORR.tif $outdir/$datestr #might be good to code orig. wd here.
		rm -v tmp_msk.tif tmp_geo.tif
		rm -v $dir\_Z.tif $dir\_CORR.tif

		gdal_calc.py -B tmp_mskDouble.tif -A tmp_V123.tif --outfile=$dir\_V123.tif --calc="((A!=255)*(A+1)+(A==255)*A)*(B>0)" --NoDataValue=0 --allBands=A
		#Expression complicated to solve real 0 values not being NoData and 255 no being +1-ed to 0
		rm -v tmp_V123.tif tmp_mskDouble.tif
		cp -v $dir\_V123.tif $outdir/$datestr
		rm -v $dir\_V123.tif

		cp -v TrackAngleMap*.tif $outdir/$datestr/
		#cp -v ./zips/*.zip.met $outdir/$datestr/
		find . -name '*.zip.met' -exec cp {} $outdir/$datestr/ \;
	fi

	cd $basedir #back to original directory.
done 


echo "Processing is complete. Go enjoy a beverage, you've earned it."
cd $basedir
