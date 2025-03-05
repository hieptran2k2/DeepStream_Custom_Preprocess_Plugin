#!/bin/bash

find / -type f -name "libnvds_infer.so" ! -path "$(pwd)/*" -exec cp ./libnvds_infer.so {} \;