#!/bin/bash 
# Remotes all temporary processing files from an image directory, leaving only the important stuff.
mkdir -p PROCESSED_INITIAL

for dir in $(ls -d AST*); do 
	echo $dir
	cd $dir 

	if [ -d "MEC-Malt" ]; then
		cd MEC-Malt

		mv -v AutoMask_STD-MALT_Num_8* ..
		mv -v Z_Num9_DeZoom1_STD-MALT* ..
		mv -v Correl_STD-MALT_Num_8* ..
	
		cd ../
	fi

	if [ -d "Ortho-MEC-Malt" ]; then
		cd Ortho-MEC-Malt

		mv -v Ort_FalseColor* ..
		mv -v Orthophotomosaic.* ..

		cd ../
	fi

    if [ -d "GeoI-Px" ]; then
        cd GeoI-Px
        mv -v Px2_Num16_DeZoom1_Geom-Im.tif ..
        cd ../
    fi

    rm RPC* *3B.* *3N.* FalseColor*.*
	rm -r GeoI-Px processing Pyram RawData Tmp-MM-Dir ImOrig MEC-Mini MEC-Malt Ortho-MEC-Malt
	cd ../ 
    mv -v $dir PROCESSED_INITIAL/
done
