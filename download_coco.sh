# download COCO
# cd into root directory of IDL_11785_Project before running

sudo apt-get install wget # for Debian/Ubuntu
mkdir data data/images data/annotations
cd data/images
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip  # do this later if not needed, takes time
wget -c http://images.cocodataset.org/zips/train2017.zip # do this later if not needed, takes time
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
rm train2017.zip
rm val2017.zip
rm test2017.zip
cd ../annotations

wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip
unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
