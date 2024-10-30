# download COCO
brew install wget # sudo apt-get install wget for Debian/Ubuntu
mkdir data/images data/annotations
cd data/images
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
rm train2017.zip
rm val2017.zip
rm test2017.zip
cd ../annotations