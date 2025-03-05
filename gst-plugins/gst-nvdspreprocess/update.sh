#!/bin/bash
find / -type f -name "libnvdsgst_preprocess.so" ! -path "$(pwd)/*" -exec cp ./libnvdsgst_preprocess.so {} \;