
if [ ! -d "$workspace/pytorch-YOLOv4/coco/" ]; then
    mkdir coco
    cd coco
    mkdir images
    cd images
    
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/zips/test2017.zip
    wget http://images.cocodataset.org/zips/unlabeled2017.zip
    
    unzip train2017.zip
    unzip val2017.zip
    unzip test2017.zip
    unzip unlabeled2017.zip
    
    rm train2017.zip
    rm val2017.zip
    rm test2017.zip
    rm unlabeled2017.zip 
    
    cd ../
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
    wget http://images.cocodataset.org/annotations/image_info_test2017.zip
    wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
    
    unzip annotations_trainval2017.zip
    unzip stuff_annotations_trainval2017.zip
    unzip image_info_test2017.zip
    unzip image_info_unlabeled2017.zip
    
    rm annotations_trainval2017.zip
    rm stuff_annotations_trainval2017.zip
    rm image_info_test2017.zip
    rm image_info_unlabeled2017.zip
    cd tool
    python coco_annotation.py --a $workspace/pytorch-YOLOv4/coco/annotations/instances_val2017.json --i $workspace/pytorch-YOLOv4/coco/images/val2017/
elif [ ! -f "$workspace/pytorch-YOLOv4/data/val.txt" ]; then
    cd tool
    python coco_annotation.py --a $workspace/pytorch-YOLOv4/coco/annotations/instances_val2017.json --i $workspace/pytorch-YOLOv4/coco/images/val2017/
fi
