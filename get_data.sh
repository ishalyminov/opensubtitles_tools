#!/bin/sh

wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/en.tar.gz -O en.tar.gz
tar -xvzf en.tar.gz
rm en.tar.gz

for file in `find OpenSubtitles -name *.gz`; do
  gzip -d $file
done
