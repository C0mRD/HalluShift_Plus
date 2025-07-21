import itertools
import torch 
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

def get_model_layer_config(model_type):
    """Get the layer configuration for different VLM models.
    
    Args:
        model_type (str): Type of model ('llava', 'mllama', 'instructblip', 'kosmos', 'phi', 'vit', 'smolvlm2b', 'smolvlmsynthetic', 'qwen', 'paligemma', 'llm')
        
    Returns:
        int: Starting index for text/language decoder layers
    """
    model_configs = {
        'llava': 24,        # LLaVA: Skip 24 CLIP vision layers (0-23), text layers start at 24
        'mllama': 40,       # Llama Vision: Skip 40 vision layers (32 local + 8 global), text layers start at 40
        'instructblip': 51, # InstructBlip: Skip 39 vision + 12 QFormer layers, text layers start at 51
        'kosmos': 24,       # Kosmos: Skip 24 vision layers (0-23), text layers start at 24
        'phi': 24,          # Phi: Skip 24 CLIP vision layers (0-23), main decoder layers start at 24
        'vit': 0,           # ViT: VisionEncoderDecoderModel outputs only decoder layers during generation
        'smolvlm2b': 0,     # SmolVLM 2B: Separate vision/text components, outputs.hidden_states contains only text decoder layers
        'smolvlmsynthetic': 0, # SmolVLM Synthetic: Separate vision/text components, outputs.hidden_states contains only text decoder layers
        'qwen': 32,         # Qwen: Skip 32 vision layers (0-31), text layers start at 32
        'paligemma': 27,    # PaliGemma: Skip 27 vision layers (0-26), text layers start at 27
        'llm': 0            # LLM: Pure text models, no vision layers to skip
    }
    
    return model_configs.get(model_type, 24)  # Default to LLaVA config

def normalize_as_distribution(tensor):
    """Normalize the input tensor as a probability distribution.
    This function reshapes the input tensor and then applies the softmax
    function to it, effectively normalizing it into a probability distribution.
    Args:
        tensor: The input tensor.
    Returns:
        The normalized tensor (probability distribution).
    """
    # Ensure tensor is in float32 format for numerical stability
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    tensor = tensor.view(-1)
    return F.softmax(tensor, dim=-1)


def wasserstein_dist(p, q):
    """Calculate the Wasserstein distance between two distributions.
    This function computes the Wasserstein distance, a measure of the distance
    between two probability distributions.
    Args:
        p: The first distribution.
        q: The second distribution.
    Returns:
        The Wasserstein distance between p and q.
    """
    p = p.to(torch.float32).cpu().numpy()
    q = q.to(torch.float32).cpu().numpy()
    return wasserstein_distance(p, q)

def cosine_similarity(tensor1, tensor2):
    """Calculate the cosine similarity between two tensors.
    This function computes the cosine similarity, a measure of similarity between two non-zero tensors.
    Args:
        tensor1: The first tensor.        
        tensor2: The second tensor.
    Returns:
        The cosine similarity between the two tensors.
    """
    tensor1 = tensor1.view(-1, tensor1.size(-1)).to(torch.float32)  
    tensor2 = tensor2.view(-1, tensor2.size(-1)).to(torch.float32)
    return F.cosine_similarity(tensor1, tensor2).item()

def plot_internal_state_2(outputs, state="hidden", model_type="llava"):
    """Analyze internal model states (hidden and attention) and calculate various metrics.

    This function processes hidden states or attentions from text decoder layers only (skipping vision layers),
    calculates Wasserstein distances and cosine similarities between consecutive states,
    and averages these metrics across all tokens.

    Args:
        outputs: Model outputs containing hidden states or attentions.
        state: The type of state to analyze, either "hidden" or "attentions". Defaults to "hidden".
        model_type: The type of VLM model ('llava', 'mllama', 'instructblip', 'kosmos', 'phi'). Defaults to "llava".

    Returns:
        A list of average Wasserstein distances and cosine similarities.
    """
    results = []
    index_sums = [0] * 30
    
    # Get the starting index for text decoder layers based on model type
    text_start_idx = get_model_layer_config(model_type)
    
    print(f"Using model type: {model_type}, text decoder layers start at index: {text_start_idx}")
    
    if state == "hidden":
        for tup in outputs.hidden_states:
            max_layer = len(tup) - 1
            
            # For all VLM models, skip vision layers to extract only text decoder features
            text_indices = [i for i in range(text_start_idx + 2, min(text_start_idx + 33, max_layer + 1), 2)]
            
            # Ensure we have at least 2 layers to compare
            if len(text_indices) < 2:
                text_indices = [i for i in range(max(0, max_layer - 15), max_layer + 1, 2)]
            
            # Convert each tensor to float32 before normalization
            vec = [normalize_as_distribution(tup[i].to(torch.float32)) for i in text_indices if i <= max_layer]
            
            if len(vec) > 1:
                div = [wasserstein_dist(vec[i], vec[i+1]) for i in range(len(vec)-1)]
                div.extend(cosine_similarity(vec[i], vec[i+1]) for i in range(len(vec)-1))
                results.append(div)
    else:
        for tup in outputs.attentions:
            max_layer = len(tup) - 1
            
            # For all VLM models, skip vision layers to extract only text decoder features
            text_indices = [i for i in range(text_start_idx + 2, min(text_start_idx + 33, max_layer + 1), 2)]
            
            # Ensure we have at least 2 layers to compare
            if len(text_indices) < 2:
                text_indices = [i for i in range(max(0, max_layer - 15), max_layer + 1, 2)]
            
            # Convert each tensor to float32 before normalization
            vec = [normalize_as_distribution(tup[i].to(torch.float32)) for i in text_indices if i <= max_layer]
            
            if len(vec) > 1:
                div = [wasserstein_dist(vec[i], vec[i+1]) for i in range(len(vec)-1)]
                div.extend(cosine_similarity(vec[i], vec[i+1]) for i in range(len(vec)-1))
                results.append(div)

    # Ensure we always return 30 features (pad with zeros if necessary)
    if results:
        for res, i in itertools.product(results, range(min(30, len(results[0])))):
            if i < len(results[0]):
                index_sums[i] += res[i]
        
        # Pad with zeros if we have fewer than 30 features
        final_results = [sum_val / len(results) for sum_val in index_sums]
    else:
        # If no results, return zeros
        final_results = [0.0] * 30
        
    return final_results
        

def probability_function(output):
    """Calculate the maximum and minimum probabilities from model logits.
    This function iterates through the logits in the model output, 
    calculates the softmax probabilities for each logit, 
    and then determines the maximum and minimum probability values.
    Args:
        output: The model output containing logits.

    Returns:
        A tuple containing two lists: the maximum probabilities and the minimum probabilities.
    """
    max_prob_results = []
    min_prob_results = []
    for logit in output.logits:
        # Convert to float32 before softmax
        logit_float = logit.to(torch.float32)
        probabilities = F.softmax(logit_float[0], dim=0)
        max_prob = probabilities.max().item()
        min_prob = probabilities.min().item()
        max_prob_results.append(max_prob)
        min_prob_results.append(min_prob)
    return [max_prob_results, min_prob_results]

# ----- HalluShift++ Parameter-Space Features -----

def layer_prediction_consistency(outputs, model_type="llava"):
    """Measure how much different layers agree on predictions.
    
    Args:
        outputs: Model output containing hidden states
        model_type: The type of VLM model ('llava', 'mllama', 'instructblip', 'kosmos', 'phi', 'vit', 'smolvlm2b', 'smolvlmsynthetic', 'qwen', 'paligemma'). Defaults to "llava".
        
    Returns:
        list: [consistency_score, inconsistency_score]
    """
    if not hasattr(outputs, 'hidden_states') or len(outputs.hidden_states) < 2:
        return [0.0, 0.0]
    
    # Get the starting index for text decoder layers based on model type
    text_start_idx = get_model_layer_config(model_type)
    max_idx = len(outputs.hidden_states) - 1
    
    # Select early and late text decoder layers for all VLM models
    early_idx = min(text_start_idx + 5, max_idx - 10)  # Early text layer
    late_idx = max_idx - 2  # Late text layer
    
    # Ensure valid indices
    if early_idx >= late_idx or early_idx < text_start_idx:
        early_idx = max(text_start_idx, max_idx // 2)
        late_idx = max_idx - 1
    
    try:
        early_layer = outputs.hidden_states[early_idx]
        late_layer = outputs.hidden_states[late_idx]
        
        # Handle tuple inputs
        if isinstance(early_layer, tuple):
            early_layer = early_layer[0]
        if isinstance(late_layer, tuple):
            late_layer = late_layer[0]
        
        # Convert to float32 and flatten
        early_flat = early_layer.to(torch.float32).view(-1)
        late_flat = late_layer.to(torch.float32).view(-1)
        
        # Compute cosine similarity
        consistency = F.cosine_similarity(early_flat, late_flat, dim=0).item()
        
        # Ensure consistency is between 0 and 1
        consistency = max(0.0, min(1.0, (consistency + 1.0) / 2.0))
        inconsistency = 1.0 - consistency
        
        return [consistency, inconsistency]
        
    except Exception as e:
        print(f"Error in layer_prediction_consistency: {e}")
        return [0.0, 0.0]

def attention_concentration_features(outputs):
    """Measure how concentrated vs dispersed attention patterns are.
    
    Args:
        outputs: Model output containing attentions
        
    Returns:
        list: [mean_concentration, std_concentration]
    """
    if not hasattr(outputs, 'attentions') or not outputs.attentions:
        return [0.0, 0.0]
    
    concentrations = []
    
    # Analyze last 3 attention layers
    for attention_layer in outputs.attentions[-3:]:
        try:
            if isinstance(attention_layer, tuple):
                attention_layer = attention_layer[0]
            
            # Convert to float32
            attn = attention_layer.to(torch.float32)
            
            # Take mean across heads: [batch, heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
            if attn.dim() == 4:
                attn = attn.mean(dim=1)
            
            # Take first batch item: [seq_len, seq_len]
            if attn.dim() == 3:
                attn = attn[0]
            
            # Flatten attention matrix
            attn_flat = attn.view(-1)
            
            # Compute concentration using Gini coefficient approximation
            sorted_attn, _ = torch.sort(attn_flat)
            n = len(sorted_attn)
            
            # Avoid division by zero
            if torch.sum(sorted_attn) == 0:
                concentration = 0.0
            else:
                cumsum = torch.cumsum(sorted_attn, 0)
                concentration = (2 * torch.sum(cumsum) / (n * torch.sum(sorted_attn)) - 1.0).item()
            
            concentrations.append(abs(concentration))  # Take absolute value
            
        except Exception as e:
            print(f"Error in attention concentration: {e}")
            concentrations.append(0.0)
    
    if concentrations:
        return [np.mean(concentrations), np.std(concentrations)]
    else:
        return [0.0, 0.0]

def perplexity_confidence_features(outputs):
    """Extract various confidence-related features including perplexity.
    
    Args:
        outputs: Model output containing logits
        
    Returns:
        list: [mean_perplexity, std_perplexity, confidence_trend, mean_confidence, low_confidence_count]
    """
    if not hasattr(outputs, 'logits') or not outputs.logits:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    max_probs = []
    perplexities = []
    
    for logit in outputs.logits:
        try:
            # Convert to float32 before softmax
            logit_float = logit.to(torch.float32)
            probs = F.softmax(logit_float[0], dim=0)
            max_prob = probs.max().item()
            
            # Calculate perplexity (avoid log(0))
            perplexity = 1.0 / max(max_prob, 1e-10)
            
            max_probs.append(max_prob)
            perplexities.append(perplexity)
            
        except Exception as e:
            print(f"Error in perplexity computation: {e}")
            # max_probs.append(0.5)
            # perplexities.append(2.0)
    
    # Calculate confidence trend (are we getting less confident over time?)
    if len(max_probs) > 1:
        confidence_trend = np.polyfit(range(len(max_probs)), max_probs, 1)[0]
    else:
        confidence_trend = 0.0
    
    # Count low-confidence predictions
    low_confidence_count = len([p for p in max_probs if p < 0.5])
    
    return [
        np.mean(perplexities),
        np.std(perplexities),
        confidence_trend,
        np.mean(max_probs),
        low_confidence_count / len(max_probs) if max_probs else 0.0  # Normalize by total count
    ]

def token_repetition_novelty_features(generated_text, tokenizer):
    """Analyze repetition patterns and token novelty.
    
    Args:
        generated_text: Generated text string
        tokenizer: Tokenizer to encode text
        
    Returns:
        list: [repetition_ratio, bigram_repetition, normalized_unique_tokens]
    """
    try:
        # Encode the text
        tokens = tokenizer.encode(generated_text, add_special_tokens=False)
        
        if len(tokens) == 0:
            return [0.0, 0.0, 0.0]
        
        # Repetition metrics
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        repetition_ratio = 1 - (unique_tokens / total_tokens) if total_tokens > 0 else 0.0
        
        # N-gram repetition (bigrams)
        if len(tokens) > 1:
            bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
            unique_bigrams = len(set(bigrams))
            bigram_repetition = 1 - (unique_bigrams / len(bigrams)) if len(bigrams) > 0 else 0.0
        else:
            bigram_repetition = 0.0
        
        # Normalize unique tokens (per 100 words)
        normalized_unique_tokens = (unique_tokens / total_tokens) if total_tokens > 0 else 0.0
        
        return [repetition_ratio, bigram_repetition, normalized_unique_tokens]
        
    except Exception as e:
        print(f"Error in token repetition analysis: {e}")
        return [0.0, 0.0, 0.0]

# ----- HalluShift++ Output-Level Features -----

def perplexity_ratio(output, prompt_len, generation_len):
    """Computes ratio of generation perplexity to prompt perplexity.
    
    Args:
        output: Model output containing logits
        prompt_len: Length of prompt in tokens
        generation_len: Length of generated text in tokens
        
    Returns:
        float: Ratio of generation perplexity to prompt perplexity
    """
    logits = output.logits
    prompt_ppl = 0
    gen_ppl = 0
    
    for i, logit in enumerate(logits):
        # Convert to float32 before softmax
        logit_float = logit.to(torch.float32)
        probs = F.softmax(logit_float[0], dim=0)
        max_prob = probs.max().item()
        # Add a small epsilon to avoid log(0)
        log_prob = np.log(max(max_prob, 1e-10))
        
        if i < prompt_len:
            prompt_ppl -= log_prob
        else:
            gen_ppl -= log_prob
    
    # Normalize by length
    if prompt_len > 0:
        prompt_ppl = np.exp(prompt_ppl / prompt_len)
    else:
        prompt_ppl = 1.0
        
    if generation_len > 0:
        gen_ppl = np.exp(gen_ppl / generation_len)
    else:
        gen_ppl = 1.0
        
    return gen_ppl / prompt_ppl

def token_probability_statistics(output):
    """Computes various statistics of token probabilities in generated text.
    
    Args:
        output: Model output containing logits
        
    Returns:
        dict: Dictionary of probability statistics
    """
    max_probs = []
    for logit in output.logits:
        # Convert to float32 before softmax
        logit_float = logit.to(torch.float32)
        probs = F.softmax(logit_float[0], dim=0)
        max_probs.append(probs.max().item())
    
    return {
        'mean': np.mean(max_probs),
        'std': np.std(max_probs),
        'min': np.min(max_probs),
        'max': np.max(max_probs),
        'median': np.median(max_probs),
        'iqr': np.percentile(max_probs, 75) - np.percentile(max_probs, 25)
    }

def token_probability_transition(output):
    """Analyzes transitions in token probabilities to detect sudden shifts.
    
    Args:
        output: Model output containing logits
        
    Returns:
        dict: Dictionary of transition metrics
    """
    max_probs = []
    for logit in output.logits:
        # Convert to float32 before softmax
        logit_float = logit.to(torch.float32)
        probs = F.softmax(logit_float[0], dim=0)
        max_probs.append(probs.max().item())
    
    # Calculate transitions (differences between consecutive probabilities)
    transitions = np.diff(max_probs)
    
    # Compute window drop only if we have enough transitions
    if len(transitions) >= 3:
        window_drops = np.convolve(transitions, np.ones(3), 'valid')
        largest_window_drop = max(0, -min(np.min(window_drops), 0))
    else:
        largest_window_drop = 0
        
    return {
        'max_drop': min(0, np.min(transitions)),
        'max_increase': max(0, np.max(transitions)),
        'total_volatility': np.sum(np.abs(transitions)),
        'num_significant_drops': np.sum(transitions < -0.3),  # Threshold for significant drop
        'largest_window_drop': largest_window_drop
    }

def truncate_after_words(text, num_words=128):
    """Truncates a string after a specified number of words.
    Args:
        text (str): The input string to truncate.
        num_words (int, optional): The maximum number of words to keep. Defaults to 128.

    Returns:
        str: The truncated string.
    """
    words = text.split()
    return " ".join(words[:num_words])

def column_to_txt(dataset, column_name, txt_file):
    """Writes the contents of a specified column in a dataframe to a text file.
    This function iterates through each row of the dataset and writes the content
    of the specified column to a text file. Newlines and carriage returns are
    replaced with spaces to ensure each entry is on a single line.

    Args:
        dataset: The input dataset (pandas DataFrame).
        column_name (str): The name of the column to extract.
        txt_file (str): The path to the output text file.
    """
    try:
        with open(txt_file, mode='w', encoding='utf-8') as txtfile:
            for text in dataset[column_name]:
                sanitized_text = text.replace('\n', ' ').replace('\r', ' ')
                txtfile.write(sanitized_text + '\n')

    except Exception as e:
        print(f"An error occurred while creating txt files: {e}")

def bleurt_processing(file1, file2, threshold=0.5):
    """Processes BLEURT scores to detect hallucinations.

    Reads BLEURT scores from a file, groups them by ID and keep the maximum, then assigns a hallucination
    label based on a threshold.  If the maximum BLEURT score for an ID is above the
    threshold, it's considered not a hallucination (0), otherwise it is (1).

    Args:
        file1 (str): Path to the file containing IDs.
        file2 (str): Path to the file containing BLEURT scores.
        threshold (float, optional): The threshold for BLEURT score. Defaults to 0.5.

    Returns:
        pandas.DataFrame: DataFrame with 'id', 'bleurt_score', and 'hallucination' columns.
        Returns None if the input files have different lengths.
    """
    try:
        with open(file1, 'r', encoding='utf-8') as f3:
            column1 = [line.strip() for line in f3.readlines()]
        with open(file2, 'r', encoding='utf-8') as f4:
            column2 = [line.strip() for line in f4.readlines()]

        if len(column1) == len(column2) :
            df = pd.DataFrame({
                'id' : column1,
                'bleurt_score': column2
            })
            df = df.groupby('id', as_index=False, sort=False)['bleurt_score'].max()
            df['hallucination'] = df['bleurt_score'].astype(float).apply(lambda x: 0 if x > threshold else 1)
            return df
        else :
            raise ValueError("All columns are not of same length during bleurt processing")
    except Exception as e:
        raise ValueError(f"An error occurred while bleurt processing: {e}")

def normalized_entropy(prob_list):
    """Calculates the normalized entropy of token probabilities across a LLM generated response.
    
    Args:
        prob_list: A list of probabilities.

    Returns:
        float: The normalized entropy, a value between 0 and 1.
        Returns 0 if the input list is empty or contains only zeros.
    """
    entropy = -np.sum([p * np.log(p) for p in prob_list if p > 0]) 
    max_entropy = np.log(len(prob_list))
    return entropy / max_entropy if max_entropy > 0 else 0

def count_low_probs(prob_list, threshold=0.1):
    """Counts the number of token probabilities across a LLM generated response below a threshold (outlier).

    Args:
        prob_list: A list of probabilities.
        threshold (float, optional): The threshold value. Defaults to 0.1.
    Returns:
        int: The number of probabilities below the threshold.
    """
    return sum(p < threshold for p in prob_list)

def count_high_probs(prob_list, threshold=0.9):
    """Counts the number of token probabilities across a LLM generated response above a threshold (outlier).

    Args:
        prob_list: A list of probabilities.
        threshold (float, optional): The threshold value. Defaults to 0.9.

    Returns:
        int: The number of probabilities above the threshold.
    """
    return sum(p > threshold for p in prob_list)

def probability_gradients(prob_list):
    """Computes the absolute gradients of token probabilities across a LLM generated response by taking the
    absolute difference between consecutive elements.

    Args:
        prob_list: A list of probabilities.

    Returns:
        list: A list of absolute probability gradients.
        Returns an empty list if the input list has fewer than two elements.
    """
    return [abs(prob_list[i+1] - prob_list[i]) for i in range(len(prob_list) - 1)]

def mean_gradient(prob_list):
    """Computes the mean absolute gradient of token probabilities across a LLM generated response by taking the
    absolute difference between consecutive elements and calculating the average of these differences.

    Args:
        prob_list: A list of probabilities.

    Returns:
        float: The mean absolute probability gradient.
        Returns 0 if the input list has fewer than two elements.
    """
    gradients = probability_gradients(prob_list)
    return np.mean(gradients) if gradients else 0

def max_gradient(prob_list):
    """Computes the maximum absolute gradient of token probabilities across a LLM generated response by taking the
    absolute difference between consecutive elements and returning the maximum of these differences.

    Args:
        prob_list: A list of probabilities.

    Returns:
        float: The maximum absolute probability gradient.
        Returns 0 if the input list has fewer than two elements.
    """
    gradients = probability_gradients(prob_list)
    return max(gradients) if gradients else 0

def percentile(prob_list, q):
    """Calculates the q-th percentile of token probabilities across a LLM generated response.

    Args:
        prob_list: A list of probabilities.
        q (float or int): Percentile to compute, which must be between 0 and 100 inclusive.

    Returns:
        float: The q-th percentile of the probability list.
    """
    return np.percentile(prob_list, q)

def data_preparation(df_1, df_2):
    """Prepares data for hallucination detection classifier training.
    This function engineers features from divergence, similarity measures and probability distributions, merges them with hallucination labels,
    and prepares the dataset for training a hallucination detection model.

    Args:
        df_1 (pandas.DataFrame): DataFrame containing divergence, similarity measures and probability distribution features.
        df_2 (pandas.DataFrame): DataFrame containing hallucination labels.

    Returns:
        pandas.DataFrame: The merged and feature-engineered DataFrame.

    Raises:
        ValueError: If the input DataFrames have different lengths.
    """

    # Creating Probabilistic Features
    temp_120 = df_1[60].copy()
    temp_121 = df_1[61].copy()

    # Maximum spread (Mps)
    df_1[61] = df_1.apply(
        lambda row: max(a - b for a, b in zip(row[60], row[61])), axis=1 )
    df_1[60] = df_1[60].apply(lambda x : min(x))

    # Other features
    df_1['norm_entropy_max'] = temp_120.apply(normalized_entropy)
    df_1['norm_entropy_min'] = temp_121.apply(normalized_entropy)

    df_1['low_prob_count_max'] = temp_120.apply(lambda x: count_low_probs(x, threshold=0.1))
    df_1['low_prob_count_min'] = temp_121.apply(lambda x: count_low_probs(x, threshold=0.1))

    df_1['mean_grad_max'] = temp_120.apply(mean_gradient)
    df_1['mean_grad_min'] = temp_121.apply(mean_gradient)

    df_1['p25_max'] = temp_120.apply(lambda x: percentile(x, 25))
    df_1['p50_max'] = temp_120.apply(lambda x: percentile(x, 50))
    df_1['p75_max'] = temp_120.apply(lambda x: percentile(x, 75))

    # HalluShift++ New Features are already in simple format (no complex processing needed)
    # The new features are:
    # - Layer consistency features (already scalar values)
    # - Attention concentration features (already scalar values)
    # - Perplexity and confidence features (already scalar values)
    # - Token repetition and novelty features (already scalar values)
    
    # Note: New features are designed to be simple scalar values that don't require 
    # additional processing, unlike the old CKA/Mahalanobis features

    if df_1.shape[0] != df_2.shape[0]:
        raise ValueError("Lengths of DataFrames are not same")

    merged_df = pd.concat([df_1, df_2['hallucination']], axis=1)
    
    return merged_df
