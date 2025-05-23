# ReVisiT
```
mkdir -p ./prerequisites
mkdir -p ./prerequisites/coco
wget http://images.cocodataset.org/zips/val2014.zip -P ./prerequisites/coco && unzip ./prerequisites/coco/val2014.zip -d ./prerequisites/coco &
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ./prerequisites/coco && unzip ./prerequisites/coco/annotations_trainval2014.zip -d ./data/coco &
```
## LLaVA1.5
### Prerequisite
```
conda env create -f LLaVA1.5/ReVisiT_LLaVA.yaml
conda activate revisit_llava
pip install numpy==1.26.4
cd LLaVA1.5/data/transformers-4.31.0
pip install -e .
cd ../../..

python prerequisites/download_from_huggingface.py --model llava
```
### CHAIR Evaluation
```
cd LLaVA1.5
bash eval_chair_llava.sh
```

## Qwen2.5-VL
### Prerequisite
```
conda env create -f Qwen2.5-VL/ReVisiT_Qwen.yaml
conda activate revisit_qwen
cd Qwen2.5-VL/data/transformers-4.50.0
pip install -e .
cd ../../..

python prerequisites/download_from_huggingface.py --model llava
```
### CHAIR Evaluation
```
cd Qwen2.5-VL
bash eval_chair_qwenvl.sh
```