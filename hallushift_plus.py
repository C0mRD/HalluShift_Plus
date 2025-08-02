import functions
import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration
)
import warnings
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import json
from datasets import load_dataset, Dataset
import threading
import csv
from qwen_vl_utils import process_vision_info
import urllib.request

# Add your Hugging Face Access Token here
hf_token = "YOUR_HF_TOKEN"

# Mapping of model names to their identifiers for LLMs
LLM_MODELS = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b",
    'vicuna_7B': "lmsys/vicuna-7b-v1.5",
    'Qwen2.5_7B': "Qwen/Qwen2.5-7B"
}

# os diye device restrict kor
# dekh os environment set er akta line ache
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Suppress warnings
warnings.filterwarnings("ignore")

def seed_everything(seed: int):
    """Sets seeds for reproducibility across various libraries.
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def detect_model_type(model_id):
    """Automatically detect the model type based on model_id.
    
    Args:
        model_id (str): The model identifier string
        
    Returns:
        str: The model type ('llava', 'mllama', 'instructblip', 'kosmos', 'phi', 'vit', 'smolvlm2b', 'smolvlmsynthetic', 'qwen', 'paligemma')
    """
    model_id_lower = model_id.lower()
    
    # Check for specific model patterns
    if 'llava' in model_id_lower:
        return 'llava'
    elif ('llama' in model_id_lower and 'vision' in model_id_lower) or 'mllama' in model_id_lower:
        return 'mllama'
    elif 'instructblip' in model_id_lower or (('instruct' in model_id_lower or 'instruction') and 'blip' in model_id_lower):
        return 'instructblip'
    elif 'kosmos' in model_id_lower:
        return 'kosmos'
    elif 'phi' in model_id_lower and ('vision' in model_id_lower or '3.5' in model_id_lower):
        return 'phi'
    elif 'vit' in model_id_lower and ('gpt2' in model_id_lower or 'captioning' in model_id_lower):
        return 'vit'
    elif 'smolvlm' in model_id_lower and 'instruct' in model_id_lower:
        return 'smolvlm2b'
    elif 'smolvlm' in model_id_lower and 'synthetic' in model_id_lower:
        return 'smolvlmsynthetic'
    elif 'qwen' in model_id_lower and ('vl' in model_id_lower or 'vision' in model_id_lower):
        return 'qwen'
    elif 'paligemma' in model_id_lower:
        return 'paligemma'
    elif 'salesforce/instructblip' in model_id_lower:
        return 'instructblip'
    elif 'meta-llama/llama-3.2' in model_id_lower and 'vision' in model_id_lower:
        return 'mllama'
    elif 'liuhaotian/llava' in model_id_lower or 'llava-hf' in model_id_lower:
        return 'llava'
    elif 'microsoft/kosmos' in model_id_lower:
        return 'kosmos'
    elif 'nlpconnect/vit-gpt2' in model_id_lower:
        return 'vit'
    elif 'huggingfacetb/smolvlm' in model_id_lower:
        if 'synthetic' in model_id_lower:
            return 'smolvlmsynthetic'
        else:
            return 'smolvlm2b'
    elif 'qwen/qwen2.5-vl' in model_id_lower:
        return 'qwen'
    elif 'google/paligemma' in model_id_lower:
        return 'paligemma'
    else:
        print(f"Warning: Could not detect model type from '{model_id}'. Using default 'llava'.")
        return 'llava'

def load_images_from_folder(folder_path, supported_formats=('.jpg', '.jpeg', '.png')):
    """
    Load all images from a folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        supported_formats (tuple): Supported image file extensions
        
    Returns:
        list: List of dictionaries containing image paths and names
    """
    image_files = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")
    
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            image_files.append({
                'path': str(file_path),
                'name': file_path.stem,
                'filename': file_path.name,
                'image': None  # Will be loaded later
            })
    
    if not image_files:
        raise ValueError(f"No supported image files found in {folder_path}")
    
    print(f"Found {len(image_files)} images in {folder_path}")
    return image_files

def load_llava_dataset():
    """
    Load LLaVA dataset from HuggingFace.
    
    Returns:
        list: List of dictionaries containing image data and names
    """
    print("Loading LLaVA dataset from HuggingFace...")
    ds = load_dataset("BUAADreamer/llava-en-zh-2k", "en")
    train_data = ds['train']
    
    image_files = []
    for idx, item in enumerate(train_data):
        # Extract the image from the dataset
        image = item['images'][0] if item['images'] else None  # Take first image if multiple
        if image is not None:
            image_files.append({
                'path': f"llava_dataset_image_{idx}",  # Virtual path since it's from dataset
                'name': f"llava_image_{idx}",
                'filename': f"llava_image_{idx}.jpg",
                'image': image,  # PIL Image object from dataset
                'messages': item['messages']  # Store messages for potential future use
            })
    
    print(f"Found {len(image_files)} images in LLaVA dataset")
    return image_files

def load_text_dataset_by_name(dataset_name, max_samples=None):
    """Loads a text dataset based on the provided name (from haL_detection.py).
    Args:
        dataset_name (str): The name of the dataset to load.
        max_samples (int): Maximum number of samples to load (for testing)
    Returns:
        Dataset: The loaded dataset.
    """
    
    # Load the TruthfulQA dataset's validation split
    if dataset_name == "truthfulqa":
        dataset = load_dataset("truthful_qa", 'generation', token=hf_token)['validation']
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    
    # Load the TriviaQA dataset and remove duplicate questions
    elif dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation", token=hf_token)
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    
    # Load the TyDiQA dataset and filter for English questions
    elif dataset_name == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        dataset = dataset.filter(lambda row: "english" in row["id"])
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    
    # Load the CoQA dataset
    elif dataset_name == 'coqa':
        dataset = load_coqa_dataset()
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    
    # Load a specific subset of the HaluEval dataset
    elif dataset_name == 'haluevaldia':
        dataset = load_dataset("pminervini/HaluEval", "dialogue")['data']
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    elif dataset_name == 'haluevalqa':
        dataset = load_dataset("pminervini/HaluEval", "qa")['data']
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    elif dataset_name == 'haluevalsum':
        dataset = load_dataset("pminervini/HaluEval", "summarization")['data']
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset
    else:
        raise ValueError("Invalid dataset name")

def load_coqa_dataset():
    """
    Downloads and processes the CoQA dataset (from haL_detection.py).
    Returns:
        Dataset: The processed CoQA dataset.
    """
    save_path = './coqa_dataset'
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(f"{save_path}/coqa-dev-v1.0.json"):
        # Download the CoQA dataset if not already present
        url = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"
        try:
            urllib.request.urlretrieve(url, f"{save_path}/coqa-dev-v1.0.json")
        except Exception as e:
            print(f"Failed to download coqa dataset file: {e}")
    
    # Load and process the dataset
    with open('./coqa_dataset/coqa-dev-v1.0.json', 'r') as infile:
        data = json.load(infile)['data']
        dataset = {
            'story': [],
            'question': [],
            'answer': [],
            'additional_answers': [],
            'id': []
        }
        for sample in data:
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']
            additional_answers = sample['additional_answers']
            for question_index, question in enumerate(questions):
                dataset['story'].append(story)
                dataset['question'].append(question['input_text'])
                dataset['answer'].append({
                    'text': answers[question_index]['input_text'],
                    'answer_start': answers[question_index]['span_start']
                })
                dataset['id'].append(sample['id'] + '_' + str(question_index))
                additional_answers_list = [
                    additional_answers[str(i)][question_index]['input_text'] for i in range(3)
                ]
                dataset['additional_answers'].append(additional_answers_list)
                story += f' Q: {question["input_text"]} A: {answers[question_index]["input_text"]}'
                if story[-1] != '.':
                    story += '.'
        return Dataset.from_dict(dataset)

def process_with_threads(args, images, process_func, max_workers, csv_writer, csv_lock, results_list):
    """Processes images in parallel using threading with incremental saving.
    Args:
        images (list): The list of images to process.
        process_func (callable): The function to apply to each image.
        max_workers (int): The maximum number of threads to use.
        csv_writer: CSV writer object for saving results
        csv_lock: Threading lock for file writing
        results_list: Thread-safe list to collect results for pickle
    Returns:
        None: Results are written directly to files
    """
    def process_and_save(image_info):
        result = process_func(image_info)
        # Thread-safe writing to CSV
        with csv_lock:
            csv_writer.writerow(result)
            results_list.append(result)
        return result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_and_save, images), total=len(images), desc=f"Extracting features from images..."))

def main():
    """
    Main function to extract HalluShift++ features from images using various VLM models.
    """
    parser = argparse.ArgumentParser(
        description="Extract HalluShift++ features from images using various VLM models (LLaVA, Llama Vision, InstructBlip, Kosmos, Phi, ViT, SmolVLM, Qwen, PaliGemma)",
        epilog="""
Examples:
  # VLM Examples:
  # LLaVA model (auto-detected)
  python hallushift.py --dataset mscoco --image_folder /path/to/images --model_id llava-hf/llava-1.5-13b-hf
  
  # Llama Vision model
  python hallushift.py --dataset mscoco --image_folder /path/to/images --model_id meta-llama/Llama-3.2-11B-Vision --model_type mllama
  
  # InstructBlip model  
  python hallushift.py --dataset mscoco --image_folder /path/to/images --model_id Salesforce/instructblip-vicuna-7b --model_type instructblip
  
  # LLM Examples:
  # TruthfulQA with Llama2 7B
  python hallushift.py --dataset truthfulqa --use_only_llm --llm_model_name llama2_7B --max_samples 1000
  
  # TyDiQA with Llama3 8B (1k samples)
  python hallushift.py --dataset tydiqa --use_only_llm --llm_model_name llama3_8B --max_samples 1000
  
  # TriviaQA with Qwen2.5 7B
  python hallushift.py --dataset triviaqa --use_only_llm --llm_model_name Qwen2.5_7B --max_samples 500
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', type=str, choices=['mscoco', 'llava', 'truthfulqa', 'triviaqa', 'tydiqa', 'coqa', 'haluevaldia', 'haluevalqa', 'haluevalsum'], required=True, help='Dataset to use: mscoco/llava (VLM) or text datasets (LLM)')
    parser.add_argument('--image_folder', type=str, help='Path to folder containing images (required for mscoco dataset)')
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-1.5-7b-hf', help='VLM/LLM model identifier')
    parser.add_argument('--model_type', type=str, choices=['llava', 'mllama', 'instructblip', 'kosmos', 'phi', 'vit', 'smolvlm2b', 'smolvlmsynthetic', 'qwen', 'paligemma'], help='Model type (auto-detected if not specified)')
    parser.add_argument('--llm_model_name', type=str, choices=list(LLM_MODELS.keys()), help='LLM model name (for text datasets)')
    parser.add_argument('--use_only_llm', action='store_true', help='Use textual LLM instead of VLM for text datasets')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to process (for testing)')
    parser.add_argument('--single_gpu', action='store_true', help='Force single GPU usage (useful for multi-GPU setups with device conflicts)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threads to use (recommend 1 for GPU memory)')
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Maximum number of new tokens to generate')
    parser.add_argument('--prompt_template', type=str, default="What do you see in this image?", help='Prompt template for image description')
    parser.add_argument('--output_dir', type=str, default='./image_features/', help='Output directory for saved features')
    parser.add_argument('--test_mode', action='store_true', help='Use only first 1000 images for quick testing')
    args = parser.parse_args()
    
    # Auto-detect model type if not specified (only for VLM mode)
    if not args.use_only_llm and not args.model_type:
        args.model_type = detect_model_type(args.model_id)
        print(f"Auto-detected model type: {args.model_type}")
    elif args.use_only_llm:
        args.model_type = None  # Not needed for LLM mode
    
    # Validate arguments based on dataset choice
    if args.dataset == 'mscoco' and not args.image_folder:
        parser.error("--image_folder is required when using mscoco dataset")
    
    # Check if using LLM mode
    text_datasets = ['truthfulqa', 'triviaqa', 'tydiqa', 'coqa', 'haluevaldia', 'haluevalqa', 'haluevalsum']
    if args.dataset in text_datasets and not args.use_only_llm:
        parser.error(f"--use_only_llm is required when using text dataset: {args.dataset}")
    
    if args.use_only_llm and not args.llm_model_name:
        parser.error("--llm_model_name is required when using --use_only_llm")

    # Set random seed for reproducibility
    seed_everything(42)

    # Prepare display information
    if args.use_only_llm:
        data_source_info = f"Text Dataset: {args.dataset}"
        workflow_source = "text QA dataset"
        model_info = f"LLM: {args.llm_model_name} ({LLM_MODELS[args.llm_model_name]})"
    elif args.dataset == 'mscoco':
        data_source_info = f"Image Folder: {args.image_folder}"
        workflow_source = "folder"
        model_info = f"VLM: {args.model_id}"
    else:
        data_source_info = "Using LLaVA HuggingFace Dataset"
        workflow_source = "HuggingFace dataset"
        model_info = f"VLM: {args.model_id}"
    
    test_mode_info = "TEST MODE: Using only first 1000 images" if args.test_mode else ""
    
    mode_title = "Text LLM HalluShift++ Feature Extraction" if args.use_only_llm else "Image HalluShift++ Feature Extraction"
    
    print(f"""
    =========================================================================
                        {mode_title} Started
    =========================================================================
    Dataset: {args.dataset}
    {data_source_info}
    {model_info}
    Output Directory: {args.output_dir}
    Max Samples: {args.max_samples if args.use_only_llm else 'All'}
    {test_mode_info}

    Workflow:
    1. Load data from {workflow_source}
    2. Initialize {'LLM' if args.use_only_llm else 'VLM'} model
    3. Generate {'answers' if args.use_only_llm else 'image descriptions'} using model-specific input formats
    4. Extract HalluShift++ features from decoder layers only
    5. Save features incrementally as each {'question' if args.use_only_llm else 'image'} is processed

    Features Extracted:
    - Original HalluShift features (hidden states, attention, probability) - 62 features
    - HalluShift++ New features (layer consistency, attention concentration, 
      perplexity/confidence, token repetition/novelty) - 12 features
    - Total: 74 features (0-73)
    - Generated text {'answers' if args.use_only_llm else 'descriptions'}
    
    {'Text Datasets Supported:' if args.use_only_llm else 'VLM Models Supported:'}
    {'- TruthfulQA, TriviaQA, TyDiQA, CoQA, HaluEval variants' if args.use_only_llm else '- LLaVA, Llama Vision, InstructBlip, Kosmos, Phi, ViT, SmolVLM, Qwen, PaliGemma'}
    =========================================================================
    """)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create filename suffix with model and dataset names
    if args.use_only_llm:
        model_name = args.llm_model_name
    else:
        model_name = args.model_id.split('/')[-1]  # Extract model name from path
    filename_suffix = f"{model_name}_{args.dataset}"
    
    # Prepare output files
    csv_file = os.path.join(args.output_dir, f'image_hallushift_features_{filename_suffix}.csv')
    feature_file = os.path.join(args.output_dir, f'image_hallushift_features_{filename_suffix}.pkl')
    metadata_file = os.path.join(args.output_dir, f'extraction_metadata_{filename_suffix}.json')
    
    # Load data based on dataset choice
    if args.use_only_llm:
        print(f"Loading {args.dataset} text dataset...\n")
        dataset = load_text_dataset_by_name(args.dataset, max_samples=args.max_samples)
        print(f"Text dataset successfully loaded: {len(dataset)} samples.\n")
        
        # Convert to list format for consistency
        data_items = []
        for idx, item in enumerate(dataset):
            data_items.append({
                'name': f"{args.dataset}_sample_{idx}",
                'filename': f"{args.dataset}_sample_{idx}.txt",
                'path': f"{args.dataset}_sample_{idx}",
                'data': item  # Store the full dataset item
            })
        images = data_items  # Reuse variable name for consistency
        
    elif args.dataset == 'mscoco':
        print("Loading images from folder...\n")
        images = load_images_from_folder(args.image_folder)
    else:  # llava dataset
        print("Loading images from LLaVA HuggingFace dataset...\n")
        images = load_llava_dataset()
    
    # If in test mode for VLM, limit to first few items
    if args.test_mode and not args.use_only_llm:
        print("TEST MODE: Limiting to first 2 images for quick testing\n")
        images = images[:2]
    
    print(f"Data successfully loaded: {len(images)} items.\n")

    # Prepare CSV headers
    feature_headers = [f'feature_{i}' for i in range(74)]  # 74 features total
    csv_headers = feature_headers + ['image_name', 'image_filename', 'image_path', 'generated_description']
    
    # Initialize CSV file
    csv_file_handle = open(csv_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file_handle)
    csv_writer.writerow(csv_headers)
    
    # Thread-safe results collection
    csv_lock = threading.Lock()
    results_list = []

    # Initialize model based on mode
    if args.use_only_llm:
        print(f"Initializing LLM model: {args.llm_model_name}...\n")
        
        # Initialize LLM model and tokenizer
        model_id = LLM_MODELS[args.llm_model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir="./models")
        
        # Set padding token if not present (needed for some LLMs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            cache_dir="./models",
            attn_implementation="eager"
        )
        
        print(f"LLM model successfully initialized.\n")
        
    else:
        print(f"Initializing {args.model_type.upper()} model...\n")
        
        # Initialize VLM model and processor
        if args.model_type == 'llava':
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token
            )
            processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token, cache_dir="./models", use_fast=True)
            
        elif args.model_type == 'mllama':
            model = MllamaForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for Llama Vision as in the example
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token
            )
            processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token, cache_dir="./models")
            
        elif args.model_type == 'instructblip':
            model = InstructBlipForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for InstructBlip as in the example
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token
            )
            processor = InstructBlipProcessor.from_pretrained(args.model_id, token=hf_token, cache_dir="./models")
            
        elif args.model_type == 'kosmos':
            model = AutoModelForVision2Seq.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token
            )
            processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token, cache_dir="./models")
            
        elif args.model_type == 'phi':
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                _attn_implementation="eager",
                token=hf_token,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(
                args.model_id, 
                token=hf_token, 
                cache_dir="./models",
                trust_remote_code=True,
                use_fast=True
            )
            
        elif args.model_type == 'vit':
            model = VisionEncoderDecoderModel.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token
            )
            feature_extractor = ViTImageProcessor.from_pretrained(args.model_id, token=hf_token, cache_dir="./models")
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, cache_dir="./models")
            # Create a combined processor for consistency
            processor = type('Processor', (), {
                'feature_extractor': feature_extractor,
                'tokenizer': tokenizer
            })()
        
        elif args.model_type == 'smolvlm2b':
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(
                args.model_id, 
                token=hf_token, 
                cache_dir="./models",
                trust_remote_code=True,
                use_fast=True
            )
            
        elif args.model_type == 'smolvlmsynthetic':
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(
                args.model_id, 
                token=hf_token, 
                cache_dir="./models",
                trust_remote_code=True,
                use_fast=True
            )
            
        elif args.model_type == 'qwen':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token
            )
            processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token, cache_dir="./models")
            
        elif args.model_type == 'paligemma':
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir="./models",
                attn_implementation="eager",
                token=hf_token
            )
            processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token, cache_dir="./models")
            
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        print(f"{args.model_type.upper()} model successfully initialized.\n")

    def process_image(image_info):
        """Process a single image and extract features."""
        try:
            # Load the image based on source
            if image_info['image'] is not None:
                # Image is already loaded (from HuggingFace dataset)
                raw_image = image_info['image'].convert('RGB')
            else:
                # Load image from file path
                raw_image = Image.open(image_info['path']).convert('RGB')

            print("processing input")
            
            # Prepare inputs based on model type
            if args.model_type == 'llava':
                # LLaVA conversation format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.prompt_template},
                            {"type": "image"},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device)
                
            elif args.model_type == 'mllama':
                # Llama Vision format
                prompt = f"<|image|><|begin_of_text|>{args.prompt_template}"
                inputs = processor(raw_image, prompt, return_tensors="pt").to(model.device)
                
            elif args.model_type == 'instructblip':
                # InstructBlip format
                inputs = processor(images=raw_image, text=args.prompt_template, return_tensors="pt").to(model.device)
                
            elif args.model_type == 'kosmos':
                # Kosmos format - similar to the example in run_kosmos.py
                prompt = f"<grounding>{args.prompt_template}"
                inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
                
            elif args.model_type == 'phi':
                # Phi format - similar to the example in run_phi.py
                messages = [
                    {"role": "user", "content": f"<|image_1|>\n{args.prompt_template}"},
                ]
                prompt = processor.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                inputs = processor(prompt, [raw_image], return_tensors="pt").to(model.device)
                
            elif args.model_type == 'vit':
                # ViT format - based on run_vit.py
                pixel_values = processor.feature_extractor(images=[raw_image], return_tensors="pt").pixel_values
                inputs = {"pixel_values": pixel_values.to(model.device)}
                
            elif args.model_type in ['smolvlm2b', 'smolvlmsynthetic']:
                # SmolVLM format - similar to the examples in smolvlm scripts
                inputs = processor(images=raw_image, text=args.prompt_template, return_tensors="pt").to(model.device)
                
            elif args.model_type == 'qwen':
                # Qwen format - similar to qwenvlm3b.py
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": raw_image,
                            },
                            {"type": "text", "text": args.prompt_template},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                
            elif args.model_type == 'paligemma':
                # PaliGemma format - similar to paligemma.py
                inputs = processor(args.prompt_template, raw_image, return_tensors="pt").to(model.device)
            
            prompt_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            
            print("Starting generation")
            # Generate response with all required outputs
            if args.model_type == 'instructblip':
                # InstructBlip has different generation parameters
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=prompt_len + args.max_new_tokens,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # InstructBlip decoding
                decoded = processor.batch_decode(generated.sequences[0][prompt_len:].unsqueeze(0), skip_special_tokens=True)[0].strip()
            elif args.model_type == 'kosmos':
                # Kosmos generation
                generated = model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # Kosmos decoding
                generated_text = processor.batch_decode(generated.sequences[0][prompt_len:].unsqueeze(0), skip_special_tokens=True)[0]
                decoded = processor.post_process_generation(generated_text, cleanup_and_extract=True)[0]
            elif args.model_type == 'phi':
                # Phi generation
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # Phi decoding
                decoded = processor.batch_decode(
                    generated.sequences[0][prompt_len:].unsqueeze(0), 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            elif args.model_type == 'vit':
                # ViT generation
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # ViT decoding
                decoded = processor.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
            elif args.model_type in ['smolvlm2b', 'smolvlmsynthetic']:
                # SmolVLM generation
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # SmolVLM decoding
                decoded = processor.tokenizer.batch_decode(generated.sequences[0][prompt_len:], skip_special_tokens=True)[0].strip()
            elif args.model_type == 'qwen':
                # Qwen generation
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # Qwen decoding
                response = processor.batch_decode(generated.sequences[0][prompt_len:].unsqueeze(0), skip_special_tokens=True)[0]
                decoded = response.strip()
            elif args.model_type == 'paligemma':
                # PaliGemma generation
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # PaliGemma decoding
                decoded = processor.batch_decode(generated.sequences[0][prompt_len:].unsqueeze(0), skip_special_tokens=True)[0]
            else:
                # LLaVA and Llama Vision generation
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True
                )
                # Standard decoding
                if hasattr(processor, 'tokenizer'):
                    decoded = processor.tokenizer.decode(generated.sequences[0][prompt_len:], skip_special_tokens=True)
                else:
                    decoded = processor.decode(generated.sequences[0][prompt_len:], skip_special_tokens=True)

            print("Feature gen")
            
            # Extract original HalluShift features
            features = (
                functions.plot_internal_state_2(generated, model_type=args.model_type) +  # Hidden state features
                functions.plot_internal_state_2(generated, state="attention", model_type=args.model_type) +  # Attention features
                functions.probability_function(generated)  # Original probability features only
            )
            
            # Add HalluShift++ New Features
            print("hallu++")
            
            # Layer-wise Prediction Consistency features
            layer_consistency = functions.layer_prediction_consistency(generated, model_type=args.model_type)
            
            # Attention Concentration features  
            attention_concentration = functions.attention_concentration_features(generated)
            
            # Perplexity and Confidence features
            perplexity_confidence = functions.perplexity_confidence_features(generated)
            
            # Token Repetition and Novelty features
            if args.model_type == 'vit':
                tokenizer = processor.tokenizer
            else:
                tokenizer = getattr(processor, 'tokenizer', processor)
            repetition_novelty = functions.token_repetition_novelty_features(decoded, tokenizer)
            
            # Add these features to the list
            features += (
                layer_consistency +      # 2 features: consistency, inconsistency
                attention_concentration + # 2 features: mean_concentration, std_concentration  
                perplexity_confidence +  # 5 features: mean_ppl, std_ppl, confidence_trend, mean_conf, low_conf_count
                repetition_novelty       # 3 features: repetition_ratio, bigram_repetition, normalized_unique_tokens
            )

            print("Adding features")
            
            # Add image metadata and decoded text
            features += [
                image_info['name'],  # Image name
                image_info['filename'],  # Image filename
                image_info['path'],  # Image path
                decoded  # Generated description
            ]
            
            return features
            
        except Exception as e:
            print(f"Error processing image {image_info['path']}: {str(e)}")
            # Return empty features with metadata in case of error
            # Original HalluShift: 30 + 30 + 2 = 62 features
            # New HalluShift++: 2 + 2 + 5 + 3 = 12 features
            # Total: 62 + 12 = 74 features
            empty_features = [0.0] * 74  
            empty_features += [
                image_info['name'],
                image_info['filename'], 
                image_info['path'],
                ""  # Empty description
            ]
            return empty_features

    def process_text_data(data_info):
        """Process a single text data item and extract features for LLM."""
        try:
            # Configure prompt templates for different datasets (from haL_detection.py)
            base_prompts = {
                'truthfulqa': "Answer the question concisely. Q: {question} A:",
                'triviaqa': "Answer the question concisely. Q: {question} A:",
                'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
                'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question} A:",
                'haluevaldia': "You are an assistant that answers questions concisely and accurately. Use the knowledge and conversation to respond naturally to the most recent message.\nKnowledge: {knowledge}.\nConversations: {dialogue_history}.\nYour Response:",
                'haluevalqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
                'haluevalsum': "{document} \n Please summarize the above article concisely. A:"
            }
            
            base_prompt = base_prompts.get(args.dataset, "")
            row = data_info['data']
            
            # Format prompt based on dataset
            if args.dataset in ['truthfulqa', 'triviaqa']:
                prompt_text = base_prompt.format(question=row['question'])
            elif args.dataset == 'tydiqa':
                prompt_text = base_prompt.format(context=row['context'], question=row['question'])
            elif args.dataset == 'coqa':
                prompt_text = base_prompt.format(story=row['story'], question=row['question'])
            elif args.dataset == 'haluevaldia':
                prompt_text = base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history'])
            elif args.dataset == 'haluevalqa':
                prompt_text = base_prompt.format(context=row['knowledge'], question=row['question'])
            elif args.dataset == 'haluevalsum':
                prompt_text = base_prompt.format(document=row['document'])
            else:
                prompt_text = str(row.get('question', ''))
            
            # Tokenize the prompt
            inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]
            
            print("Starting LLM generation")
            
            # Generate response with all required outputs
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,     
                output_attentions=True,      
                output_logits=True
            )
            
            # Decode the generated text
            decoded = tokenizer.decode(
                generated.sequences[0, prompt_len:],
                skip_special_tokens=True
            )
            
            print("LLM Feature extraction")
            
            # Extract original HalluShift features (using LLM-appropriate model_type)
            # For LLMs, we need to determine the number of layers
            num_layers = len(model.model.layers) if hasattr(model.model, 'layers') else 32
            
            features = (
                functions.plot_internal_state_2(generated, state="hidden", model_type="llm") +  # Hidden state features
                functions.plot_internal_state_2(generated, state="attention", model_type="llm") +  # Attention features
                functions.probability_function(generated)  # Original probability features only
            )
            
            # Add HalluShift++ New Features
            print("LLM HalluShift++ features")
            
            # Layer-wise Prediction Consistency features
            layer_consistency = functions.layer_prediction_consistency(generated, model_type="llm")
            
            # Attention Concentration features  
            attention_concentration = functions.attention_concentration_features(generated)
            
            # Perplexity and Confidence features
            perplexity_confidence = functions.perplexity_confidence_features(generated)
            
            # Token Repetition and Novelty features
            repetition_novelty = functions.token_repetition_novelty_features(decoded, tokenizer)
            
            # Add these features to the list
            features += (
                layer_consistency +      # 2 features: consistency, inconsistency
                attention_concentration + # 2 features: mean_concentration, std_concentration  
                perplexity_confidence +  # 5 features: mean_ppl, std_ppl, confidence_trend, mean_conf, low_conf_count
                repetition_novelty       # 3 features: repetition_ratio, bigram_repetition, normalized_unique_tokens
            )

            print("Adding LLM metadata")
            
            # Add data metadata and decoded text
            features += [
                data_info['name'],  # Data name
                data_info['filename'],  # Data filename
                data_info['path'],  # Data path
                decoded  # Generated answer
            ]
            
            return features
            
        except Exception as e:
            print(f"Error processing text data {data_info['path']}: {str(e)}")
            # Return empty features with metadata in case of error
            # Original HalluShift: 30 + 30 + 2 = 62 features
            # New HalluShift++: 2 + 2 + 5 + 3 = 12 features
            # Total: 62 + 12 = 74 features
            empty_features = [0.0] * 74  
            empty_features += [
                data_info['name'],
                data_info['filename'], 
                data_info['path'],
                ""  # Empty answer
            ]
            return empty_features

    # Process data with incremental saving
    try:
        if args.num_workers == 1 or (args.use_only_llm or args.model_type in ['llava', 'mllama', 'instructblip', 'kosmos', 'phi', 'vit', 'smolvlm2b', 'smolvlmsynthetic', 'qwen', 'paligemma']):
            # Single-threaded processing for GPU models or when explicitly requested
            if args.use_only_llm:
                desc_text = f"Extracting features from {args.dataset} text data..."
                process_func = process_text_data
            else:
                desc_text = "Extracting features from images..."
                process_func = process_image
                
            print(f"Processing with single thread for GPU model safety...")
            for data_info in tqdm(images, desc=desc_text):
                result = process_func(data_info)
                csv_writer.writerow(result)
                csv_file_handle.flush()  # Ensure data is written to disk
                results_list.append(result)
        else:
            # Multi-threaded processing with thread-safe writing (CPU models only)
            if args.use_only_llm:
                process_func = process_text_data
            else:
                process_func = process_image
                
            print(f"Processing with {args.num_workers} threads...")
            process_with_threads(args, images, process_func, max_workers=args.num_workers, 
                               csv_writer=csv_writer, csv_lock=csv_lock, results_list=results_list)
    
    finally:
        # Close CSV file
        csv_file_handle.close()

    print("\nFeature extraction completed.\n")

    # Convert results to DataFrame for pickle saving
    df = pd.DataFrame(results_list)
    
    # Extract metadata and generated text (last 4 columns)
    data_names = df.iloc[:, -4]
    data_filenames = df.iloc[:, -3]
    data_paths = df.iloc[:, -2]
    generated_text = df.iloc[:, -1]
    
    # Keep only the feature columns (exclude metadata)
    features_df = df.iloc[:, :-4]
    
    print("Saving additional data formats...\n")
    
    # Save features and metadata in pickle format
    feature_data = {
        'features': features_df,
        'data_names': data_names,  
        'data_filenames': data_filenames,
        'data_paths': data_paths,
        'generated_text': generated_text,
        'metadata': {
            'model_id': args.model_id if not args.use_only_llm else LLM_MODELS[args.llm_model_name],
            'model_type': args.llm_model_name if args.use_only_llm else args.model_type,
            'dataset': args.dataset,
            'prompt_template': args.prompt_template,
            'max_new_tokens': args.max_new_tokens,
            'num_samples': len(images),
            'use_only_llm': args.use_only_llm,
            'max_samples': args.max_samples if args.use_only_llm else None,
            'feature_extraction_date': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save as pickle file
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_data, f)
    
    # Save metadata as JSON
    with open(metadata_file, 'w') as f:
        json.dump(feature_data['metadata'], f, indent=2)
    
    print(f"Features saved to:")
    print(f"  - CSV file (incremental): {csv_file}")
    print(f"  - Pickle file: {feature_file}")
    print(f"  - Metadata: {metadata_file}")
    
    mode_text = "Text LLM" if args.use_only_llm else "Image VLM"
    data_text = "samples" if args.use_only_llm else "images"
    
    print(f"\n{mode_text} HalluShift++ feature extraction completed successfully.")
    print(f"Processed {len(images)} {data_text}")
    print(f"Extracted {features_df.shape[1]} features per {data_text[:-1]}")
    print("Features were saved incrementally during processing.")
    print("Features are ready for classifier training.\n")
    print("=========================================================================\n")

if __name__ == '__main__':
    main() 