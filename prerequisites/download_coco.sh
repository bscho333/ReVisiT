mkdir -p ./prerequisites/coco
wget http://images.cocodataset.org/zips/val2014.zip -P ./prerequisites/coco && unzip ./prerequisites/coco/val2014.zip -d ./prerequisites/coco &
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ./prerequisites/coco && unzip ./prerequisites/coco/annotations_trainval2014.zip -d ./data/coco &