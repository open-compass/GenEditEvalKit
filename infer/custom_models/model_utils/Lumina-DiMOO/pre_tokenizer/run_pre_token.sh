# Step1 extract image from json file
in_filename="/your/json/path"
target_resolution=512 # for edit/mmu_single_image/mmu_multi_image, target resolution is setted in script
edit_type="t2i" # t2i, edit, mmu_single_image, mmu_multi_image
base_name=${edit_type}_$(basename "$in_filename")
main_name="${base_name%.*}"
log_dir="./pre_token/${main_name}-${target_resolution}_log"
out_dir="/pre_token/${main_name}_vae_code-${target_resolution}"
mkdir -p "$log_dir"
for i in {0..31}
do
  gpu_id=$((i % 8))
  export CUDA_VISIBLE_DEVICES=${gpu_id}

  python3 -u pre_tokenizer/pre_tokenize.py \
    --splits=32 \
    --rank=${i} \
    --in_filename "$in_filename"  \
    --out_dir "$out_dir" \
    --type ${edit_type} \
    --target_size ${target_resolution} &> "${log_dir}/${target_resolution}-${i}.log" &
done

# Step2 concatenate all the sub-records into one json file
python3 -u pre_tokenizer/concat_record.py \
  --sub_record_dir "$out_dir" \
  --save_path "$out_dir/all_records.json"
