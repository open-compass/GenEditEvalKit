IMAGE_DIR=$1
MODEL_NAME=$2
EVAL_ENV=$3

$CONDA_BASE/envs/$EVAL_ENV/bin/python evaluation/Qwen2.5-VL/eval_entity.py \
  --image_folder $IMAGE_DIR/entity_reasoning \
  --output_path $IMAGE_DIR/csv_result/entity_reasoning \
  --prompt_json prompts/entity_reasoning.json \
  --qs_json deepseek_evaluation_qs/evaluation_entity.json \
  --model_name $MODEL_NAME
echo "Finished evaluating entity_reasoning."

$CONDA_BASE/envs/$EVAL_ENV/bin/python evaluation/Qwen2.5-VL/eval_idiom.py \
  --image_folder $IMAGE_DIR/idiom_interpretation \
  --output_path $IMAGE_DIR/csv_result/idiom_interpretation \
  --prompt_json prompts/idiom_interpretation.json \
  --qs_json deepseek_evaluation_qs/evaluation_idiom.json \
  --model_name $MODEL_NAME
echo "Finished evaluating idiom_interpretation."

$CONDA_BASE/envs/$EVAL_ENV/bin/python evaluation/Qwen2.5-VL/eval_scientific.py \
  --image_folder $IMAGE_DIR/scientific_reasoning \
  --output_path $IMAGE_DIR/csv_result/scientific_reasoning \
  --prompt_json prompts/scientific_reasoning.json \
  --qs_json deepseek_evaluation_qs/evaluation_scientific.json \
  --model_name $MODEL_NAME
echo "Finished evaluating scientific_reasoning."

$CONDA_BASE/envs/$EVAL_ENV/bin/python evaluation/Qwen2.5-VL/eval_textual_image.py \
  --image_folder $IMAGE_DIR/textual_image_design \
  --output_path $IMAGE_DIR/csv_result/textual_image_design \
  --prompt_json prompts/textual_image_design.json \
  --qs_json deepseek_evaluation_qs/evaluation_textual_image.json \
  --model_name $MODEL_NAME
echo "Finished evaluating textual_image_design."