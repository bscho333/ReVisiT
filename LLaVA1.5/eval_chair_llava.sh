#!/bin/bash

seed=42
model="llava"
benchmark="chair"
dataset_name="coco"

####################################################################################################
#                                  Fill HERE with your own paths                                   #
model_path="$PWD/../prerequisites/llava-v1.5-7b"
coco_path="$PWD/../prerequisites/coco"
output_path="$PWD/../output"
#                                                                                                  #
####################################################################################################
#                                           Arguments                                              #
do_sample=False
use_revisit=True
experiment_index=0
max_new_tokens=512
num_eval_samples=10
#                                                                                                  #
####################################################################################################

data_path="${coco_path}/val2014/"
anno_path="${coco_path}/annotations/instances_val2014.json"

exp_name="${benchmark}/${model}/s${seed}_m${max_new_tokens}_n${num_eval_samples}"
if [ "${do_sample}" = "True" ]; then
    exp_name="${exp_name}/sample"
else
    exp_name="${exp_name}/greedy"
fi
if [ "${use_revisit}" = "True" ]; then
    exp_name="${exp_name}_revisit"
fi
exp_name="${exp_name}_${experiment_index}"
echo "exp_name: ${exp_name}"

torchrun --nnodes=1 --nproc_per_node=1 --master_port 2323 eval/${benchmark}_eval_${model}.py \
    --seed ${seed} \
    --model_path ${model_path} \
    --model_base ${model} \
    --data_path ${data_path} \
    --anno_path ${anno_path} \
    --output_path ${output_path} \
    --do_sample ${do_sample} \
    --use_revisit ${use_revisit} \
    --exp_name ${exp_name} \
    --max_new_tokens ${max_new_tokens} \
    --num_eval_samples ${num_eval_samples}

echo "Running chair.py"
cap_json_path="${output_path}/${exp_name}.jsonl"
echo ${cap_json_path}
cd eval
python chair.py \
    --cap_file ${cap_json_path} \
    --coco_path ${coco_path}/annotations \
    --save_path ${output_path}/${exp_name}_result.jsonl \
    --image_id_key image_id \
    --caption_key caption 