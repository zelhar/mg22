#!/bin/bash
rsync -tlPvr --delete-excluded --delete --ignore-errors \
    --munge-links \
    --files-from='./dotfileslist.txt' \
    --exclude '.git/*' \
    --exclude '.cache/' \
    --exclude '.netrwhist' \
    --exclude 'nodejs*' \
    --exclude '*.Appimage' \
    --exclude '*.AppImage' \
    --exclude '*.appimage' \
    --exclude 'bin/*.jar' \
    --exclude 'bin/sioyek' \
    $HOME ./myusr_home/

#backup installed packages list
#pacman -Qe > ./package_list.txt
conda list -n torch > ./conda_env_package_list.txt
