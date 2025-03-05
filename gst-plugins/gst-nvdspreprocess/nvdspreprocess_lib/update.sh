#!/bin/bash
find / -type f -name "libcustom2d_preprocess.so" ! -path "$(pwd)/*" -exec cp ./libcustom2d_preprocess.so {} \;