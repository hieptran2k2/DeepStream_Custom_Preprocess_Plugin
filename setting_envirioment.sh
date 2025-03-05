#!/bin/bash

# Check if the input parameter is provided
if [ -z "$1" ]; then
  echo "Please provide the path to setting sources as a parameter."
  exit 1
fi

# Assign the input parameter to a variable SOURCES_DIR
SOURCES_DIR=$1

# Perform the required tasks
cd $SOURCES_DIR/includes
find / -type f -name "nvbufsurftransform.h" ! -path "$(pwd)/*" -exec cp ./nvbufsurftransform.h {} \;

cd $SOURCES_DIR/libs/nvdsinfer
make
bash update.sh

cd $SOURCES_DIR/gst-plugins/gst-nvdspreprocess
make
bash update.sh

cd $SOURCES_DIR/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib
make
bash update.sh
