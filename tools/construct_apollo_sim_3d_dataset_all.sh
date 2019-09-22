# copy images
mkdir images
mkdir images/00
find ./Output/RGB/06-00/ -iname "*jpg" -exec cp {} ./images/00 \;

mkdir images/01
find ./Output/RGB/12-00/ -iname "*jpg" -exec cp {} ./images/01 \;

mkdir images/02
find ./Output/RGB/17-00/ -iname "*jpg" -exec cp {} ./images/02 \;

mkdir images/03
find ./Output2/RGB/06-00/ -iname "*jpg" -exec cp {} ./images/03 \;

mkdir images/04
find ./Output2/RGB/12-00/ -iname "*jpg" -exec cp {} ./images/04 \;

mkdir images/05
find ./Output2/RGB/17-00/ -iname "*jpg" -exec cp {} ./images/05 \;

# copy laneline labels
mkdir labels
mkdir labels/00
find ./Output/LaneLine_GroundTruth/06-00/ -iname "*txt" -exec cp {} ./labels/00 \;

mkdir labels/01
find ./Output/LaneLine_GroundTruth/12-00/ -iname "*txt" -exec cp {} ./labels/01 \;

mkdir labels/02
find ./Output/LaneLine_GroundTruth/17-00/ -iname "*txt" -exec cp {} ./labels/02 \;

mkdir labels/03
find ./Output2/LaneLine_GroundTruth/06-00/ -iname "*txt" -exec cp {} ./labels/03 \;

mkdir labels/04
find ./Output2/LaneLine_GroundTruth/12-00/ -iname "*txt" -exec cp {} ./labels/04 \;

mkdir labels/05
find ./Output2/LaneLine_GroundTruth/17-00/ -iname "*txt" -exec cp {} ./labels/05 \;

# copy segmentation labels
mkdir segmentation
mkdir segmentation/00
find ./Output/Segmentation/06-00/ -iname "*png" -exec cp {} ./segmentation/00 \;

mkdir segmentation/01
find ./Output/Segmentation/12-00/ -iname "*png" -exec cp {} ./segmentation/01 \;

mkdir segmentation/02
find ./Output/Segmentation/17-00/ -iname "*png" -exec cp {} ./segmentation/02 \;

mkdir segmentation/03
find ./Output2/Segmentation/06-00/ -iname "*png" -exec cp {} ./segmentation/03 \;

mkdir segmentation/04
find ./Output2/Segmentation/12-00/ -iname "*png" -exec cp {} ./segmentation/04 \;

mkdir segmentation/05
find ./Output2/Segmentation/17-00/ -iname "*png" -exec cp {} ./segmentation/05 \;

# copy Depth labels
mkdir depth
mkdir depth/00
find ./Output/Depth/06-00/ -iname "*png" -exec cp {} ./depth/00 \;

mkdir depth/01
find ./Output/Depth/12-00/ -iname "*png" -exec cp {} ./depth/01 \;

mkdir depth/02
find ./Output/Depth/17-00/ -iname "*png" -exec cp {} ./depth/02 \;

mkdir depth/03
find ./Output2/Depth/06-00/ -iname "*png" -exec cp {} ./depth/03 \;

mkdir depth/04
find ./Output2/Depth/12-00/ -iname "*png" -exec cp {} ./depth/04 \;

mkdir depth/05
find ./Output2/Depth/17-00/ -iname "*png" -exec cp {} ./depth/05 \;

# save relative image path into list
cd images
find . -iname "*jpg" | sort>../img_list.txt
