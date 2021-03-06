#!/bin/bash

# clone kits19 starter code including downloader
# dependencies are alreday installed in Dockerfile
git clone https://github.com/neheller/kits19.git /data/kits19
cd /data/kits19

# download KiTS19 dataset under /data/kits19/data
python -m starter_code.get_imaging
