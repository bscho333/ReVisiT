#!/bin/bash

seed=42
model="qwenvl"
exp_name="chair"
dataset_name="coco"
####################################################################################################
# Fill HERE with your own paths
model_path="$PWD/../prerequisites/Qwen2.5-VL-7B-Instruct"
coco_path="$PWD/../prerequisites/coco"
output_path="$PWD/output/chair"

# Arguments
do_sample=False
use_revisit=True
experiment_index=000
max_new_tokens=512
####################################################################################################
data_path="${coco_path}/val2014/"
anno_path="${coco_path}/annotations/instances_val2014.json"

echo "do_sample: ${do_sample}"
echo "use_revisit: ${use_revisit}"
torchrun --nnodes=1 --nproc_per_node=1 --master_port 2323 eval/${exp_name}_eval_${model}.py \
    --seed ${seed} \
    --model_path ${model_path} \
    --model_base ${model} \
    --data_path ${data_path} \
    --anno_path ${anno_path} \
    --output_path ${output_path} \
    --do_sample ${do_sample} \
    --use_revisit ${use_revisit} \
    --experiment_index ${experiment_index} \
    --max_new_tokens ${max_new_tokens}

echo "Running chair.py"
cap_json_path="${out_path}/exp_${experiment_index}.jsonl"
echo ${cap_json_path}
python eval/chair.py \
    --cap_file ${cap_json_path} \
    --coco_path ${coco_path}/annotations \
    --save_path ${output_path}/exp_${experiment_index}_result.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --exp_idx ${experiment_index} 