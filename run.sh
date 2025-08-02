#!/bin/bash

echo "Starting HalluShift++ feature extraction for all models and datasets..."

# ===== LLaVA 13B Model =====
echo "Running LLaVA 13B on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id liuhaotian/llava-v1.5-13b --test_mode

echo "Running LLaVA 13B on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id liuhaotian/llava-v1.5-13b --test_mode

# ===== Llama 11B Vision Model =====
echo "Running Llama 11B Vision on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id meta-llama/Llama-3.2-11B-Vision --test_mode

echo "Running Llama 11B Vision on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id meta-llama/Llama-3.2-11B-Vision --test_mode

# ===== InstructBlip Model =====
echo "Running InstructBlip on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id Salesforce/instructblip-vicuna-7b --test_mode

echo "Running InstructBlip on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id Salesforce/instructblip-vicuna-7b --test_mode

# ===== Kosmos Model =====
echo "Running Kosmos on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id microsoft/kosmos-2-patch14-224 --model_type kosmos --test_mode

echo "Running Kosmos on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id microsoft/kosmos-2-patch14-224 --model_type kosmos --test_mode

# ===== Phi Model =====
echo "Running Phi on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id Lexius/Phi-3.5-vision-instruct --model_type phi --test_mode

echo "Running Phi on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id Lexius/Phi-3.5-vision-instruct --model_type phi --test_mode

# ===== ViT Model =====
echo "Running ViT on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id nlpconnect/vit-gpt2-image-captioning --model_type vit --test_mode

echo "Running ViT on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id nlpconnect/vit-gpt2-image-captioning --model_type vit --test_mode

# ===== SmolVLM 2B Model =====
echo "Running SmolVLM 2B on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct --model_type smolvlm2b --test_mode

echo "Running SmolVLM 2B on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct --model_type smolvlm2b --test_mode

# ===== SmolVLM Synthetic Model =====
echo "Running SmolVLM Synthetic on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id HuggingFaceTB/SmolVLM-Synthetic --model_type smolvlmsynthetic --test_mode

echo "Running SmolVLM Synthetic on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id HuggingFaceTB/SmolVLM-Synthetic --model_type smolvlmsynthetic --test_mode

# ===== Qwen Model =====
echo "Running Qwen on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id Qwen/Qwen2.5-VL-3B-Instruct --model_type qwen --test_mode

echo "Running Qwen on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id Qwen/Qwen2.5-VL-3B-Instruct --model_type qwen --test_mode

# ===== PaliGemma Model =====
echo "Running PaliGemma on MSCOCO dataset..."
python hallushift_plus.py --dataset mscoco --image_folder ./dataset/val2017 --model_id google/paligemma-3b-mix-224 --model_type paligemma --test_mode

echo "Running PaliGemma on LLaVA dataset..."
python hallushift_plus.py --dataset llava --model_id google/paligemma-3b-mix-224 --model_type paligemma --test_mode

echo "All feature extractions completed!"

# ===== Optional: Train classifiers (uncomment if you have labels) =====
# echo "Training classifiers..."
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_llava-v1.5-13b_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_llava13b_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_Llama-3.2-11B-Vision_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_llama11b_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_instructblip-vicuna-7b_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_instructblip_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_kosmos-2-patch14-224_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_kosmos_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_Phi-3.5-vision-instruct_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_phi_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_vit-gpt2-image-captioning_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_vit_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_SmolVLM2-2.2B-Instruct_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_smolvlm2b_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_SmolVLM-Synthetic_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_smolvlmsynthetic_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_Qwen2.5-VL-3B-Instruct_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_qwen_mscoco.txt
# python semantic_hallucination_classifier.py --features_csv image_hallushift_features_paligemma-3b-mix-224_mscoco_with_gt.csv --use_only_hallushift --debug > results/hallushift_paligemma_mscoco.txt