from pathlib import Path
from setuptools import find_packages, setup

readme = Path(__file__).parent / 'README.md'

setup(name='pymmaster',
      version='0.1.2-dev',
      description='pymmaster is a python package for processing of ASTER DEMs using MMASTER.',
      long_description=readme.read_text(),
      long_description_content_type='text/markdown',

      url='https://github.com/luc-girod/MMASTER-workflows',
      author='Luc Girod, Romain Hugonnet, Chris Nuth, Bob McNabb',
      author_email='robertmcnabb@gmail.com',

      doc_url='https://mmaster-workflows.readthedocs.io/',
      maintainer='iamdonovan',
      license='GPL-3.0',
      license_file='LICENSE',
      include_package_data=True,
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Unix Shell',
          'Topic :: Scientific/Engineering :: GIS',
          'Topic :: Scientific/Engineering :: Image Processing'
      ],
      python_requires='>=3.7',
      install_requires=['fiona', 'gdal', 'geopandas', 'h5py',
                        'matplotlib', 'numpy', 'pandas', 'pyproj',
                        'pybob>0.25', 'scikit-image', 'scipy', 'shapely'],
      packages=find_packages(),
      scripts=['bin/apply_mmaster_corrections.py', 'bin/batch_coregister_tiles.py', 'bin/bias_correct_tiles.py',
               'bin/create_micmac_xml.py', 'bin/mmaster_bias_correction.py',
               'bin/mosaic_micmac_tiles.py', 'bin/sort_aster_strips.py', 'bin/sort_l1a.py',
               'bin/CleanMicMac.sh', 'bin/Link_MMASTER_Files.sh', 'bin/PostProcessMicMac.sh', 'bin/process_l1a.sh',
               'bin/process_mmaster.sh', 'bin/RunMicMacAster_batch.sh', 'bin/wget_l1a.sh',
               'bin/WorkFlow_WaterMask.sh', 'bin/WorkFlowASTER.sh',
               'bin/wrapper_region_sbatch.sh'],
      zip_safe=False)
