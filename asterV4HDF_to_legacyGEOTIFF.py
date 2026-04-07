#This is completelly vibecoded with the help of Gemini Please check my work...
import os
import sys
import numpy as np
from osgeo import gdal
from pyhdf.SD import SD, SDC

def export_aster_v4_perfect_legacy(hdf_path, root_out_dir):
    # 1. SETUP DIRECTORY STRUCTURE
    scene_name = os.path.splitext(os.path.basename(hdf_path))[0]
    out_dir = os.path.join(root_out_dir, scene_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2. OPEN HANDLES
    ds = gdal.Open(hdf_path)
    sd = SD(hdf_path, SDC.READ)
    subdatasets = ds.GetSubDatasets()
    
    print(f"Processing Scene: {scene_name}")
    print(f"Output Folder: {out_dir}")

    # 3. PROCESS SUBDATASETS
    for sds_name, _ in subdatasets:
        parts = sds_name.split(':')
        group = parts[-2] 
        tag = parts[-1]   
        
        file_ext = ".tif" if tag == "ImageData" else ".txt"
        out_path = os.path.join(out_dir, f"{scene_name}.{group}.{tag}{file_ext}")

        if tag == "ImageData":
            gdal.Translate(out_path, sds_name, format='GTiff')
            
        elif tag == "LatticePoint":
            # FORMAT: Line [Space] Sample [Space] \n
            sds_ds = gdal.Open(sds_name)
            data = sds_ds.ReadAsArray() 
            with open(out_path, "w") as f:
                for block in data:
                    for pair in block:
                        f.write(f"{int(pair[0])} {int(pair[1])} \n")
                    f.write("\n") # Blank line between blocks

        else:
            # ALL TABLES: Space-separated, 6-decimal precision
            sds_ds = gdal.Open(sds_name)
            if sds_ds:
                data = sds_ds.ReadAsArray()
                if data.ndim == 3: 
                    data = data.reshape(-1, data.shape[-1])
                # Changed delimiter to a single space
                np.savetxt(out_path, data, fmt='%.6f', delimiter=' ')

    # 4. COORDINATE GRIDS (Latitude / Longitude)
    datasets = sd.datasets()
    for ds_name, info in datasets.items():
        if "Latitude" in ds_name or "Longitude" in ds_name:
            sds = sd.select(ds_name)
            data = sds.get()
            shape = data.shape
            
            # Map based on column count (11 for VNIR/TIR, 104 for SWIR)
            if shape[1] == 11:
                if "TIR" in ds_name:
                    target_bands = [f"TIR_Band{i}" for i in range(10, 15)]
                else:
                    target_bands = ["VNIR_Band1", "VNIR_Band2", "VNIR_Band3N", "VNIR_Band3B"]
            elif shape[1] == 104:
                target_bands = [f"SWIR_Band{i}" for i in range(4, 10)]
            else:
                target_bands = [f"Unknown_Swath_{ds_name}"]

            clean_tag = "Latitude" if "Latitude" in ds_name else "Longitude"
            for b in target_bands:
                latlon_out = os.path.join(out_dir, f"{scene_name}.{b}.{clean_tag}.txt")
                # Space-separated coordinates
                np.savetxt(latlon_out, data, fmt='%.6f', delimiter=' ')

    # 5. GLOBAL ANCILLARY DATA
    with open(os.path.join(out_dir, f"{scene_name}.Ancillary_Data.txt"), "w") as f:
        meta = ds.GetMetadata()
        for k, v in meta.items():
            f.write(f"{k}={v}\n")

    sd.end()
    print("Bundle extraction complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aster_to_legacy.py <path_to_hdf_file>")
        sys.exit(1)
    
    input_hdf = sys.argv[1]
    root_legacy_folder = "ASTER_LEGACY"
    export_aster_v4_perfect_legacy(input_hdf, root_legacy_folder)
