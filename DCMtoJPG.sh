#!/bin/sh
# program pro prevedeni vsech souboru DCM v urcite slozce na JPG s kvalitou 100 %
for file in `ls *.dcm`
do
	mogrify -format jpg -quality 100 $file
done
exit 0
