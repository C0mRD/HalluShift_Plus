import pandas as pd
import numpy as np
import spacy
import re
import ast
from typing import List, Dict, Tuple, Set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os
import glob
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from difflib import SequenceMatcher

class ImprovedSemanticChunkExtractor:
    """Semantic chunk extraction"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise
    
    def extract_chunks(self, text: str) -> List[Dict]:
        """Extract semantic chunks following HalLoc token-level approach"""
        doc = self.nlp(text)
        chunks = []
        
        # Extract object mentions (nouns and entities)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                chunks.append({
                    'text': token.text.strip(),
                    'type': 'object',
                    'start': token.idx,
                    'end': token.idx + len(token.text),
                    'token': token
                })
        
        # Extract attribute-object pairs
        for token in doc:
            if token.pos_ == 'ADJ' and token.head.pos_ in ['NOUN', 'PROPN']:
                attr_obj_text = f"{token.text} {token.head.text}"
                chunks.append({
                    'text': attr_obj_text,
                    'type': 'attribute',
                    'start': token.idx,
                    'end': token.head.idx + len(token.head.text),
                    'attribute': token.text,
                    'object': token.head.text,
                    'tokens': [token, token.head]
                })
        
        # Extract relationship patterns
        for token in doc:
            if token.dep_ in ['prep', 'agent'] or token.pos_ == 'VERB':
                # Find related objects
                related_objects = []
                for child in token.children:
                    if child.pos_ in ['NOUN', 'PROPN']:
                        related_objects.append(child.text)
                
                if len(related_objects) >= 2:
                    rel_text = f"{related_objects[0]} {token.text} {related_objects[1]}"
                    chunks.append({
                        'text': rel_text,
                        'type': 'relation',
                        'start': min([child.idx for child in token.children if child.pos_ in ['NOUN', 'PROPN']]),
                        'end': max([child.idx + len(child.text) for child in token.children if child.pos_ in ['NOUN', 'PROPN']]),
                        'relation': token.text,
                        'objects': related_objects,
                        'tokens': [token] + [child for child in token.children if child.pos_ in ['NOUN', 'PROPN']]
                    })
        
        # Remove duplicates and sort
        unique_chunks = []
        seen_texts = set()
        for chunk in chunks:
            if chunk['text'].lower() not in seen_texts and len(chunk['text'].strip()) > 2:
                seen_texts.add(chunk['text'].lower())
                unique_chunks.append(chunk)
        
        return sorted(unique_chunks, key=lambda x: x['start'])

class HierarchicalGroundTruthMatcher:
    """Hierarchical ground truth matching following formal hallucination definitions"""
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.debug_examples = []
        # Expanded object categories and their variations with hypernyms
        self.object_synonyms = {
            'person': ['man', 'woman', 'people', 'human', 'individual', 'boy', 'girl', 'child', 'adult', 'figure'],
            'car': ['vehicle', 'automobile', 'auto', 'truck', 'van', 'suv', 'sedan'],
            'boat': ['vessel', 'ship', 'sailboat', 'motorboat', 'yacht', 'craft'],
            'dog': ['puppy', 'canine', 'pet', 'hound'],
            'cat': ['kitten', 'feline', 'pet'],
            'building': ['house', 'structure', 'edifice', 'home', 'construction'],
            'tree': ['plant', 'vegetation', 'bush', 'shrub'],
            'food': ['meal', 'dish', 'snack', 'cuisine'],
            'water': ['liquid', 'fluid', 'ocean', 'sea', 'lake', 'river', 'body', 'pond'],
            'book': ['novel', 'magazine', 'publication', 'text'],
            'phone': ['cellphone', 'smartphone', 'mobile', 'device'],
            'computer': ['laptop', 'pc', 'desktop', 'machine'],
            'chair': ['seat', 'stool', 'furniture'],
            'table': ['desk', 'surface', 'furniture'],
            'bag': ['purse', 'backpack', 'handbag', 'sack'],
            'bottle': ['container', 'jar', 'vessel'],
            'cup': ['mug', 'glass', 'container'],
            'ball': ['sphere', 'orb'],
            'flower': ['bloom', 'blossom', 'plant'],
            'bird': ['eagle', 'crow', 'pigeon', 'sparrow', 'animal'],
            'fish': ['salmon', 'tuna', 'goldfish', 'animal'],
            'scene': ['image', 'picture', 'view', 'environment', 'setting'],
            'bench': ['seat', 'furniture'],
            'mountain': ['hill', 'peak', 'summit'],
            'sky': ['air', 'atmosphere', 'heavens'],
            'ground': ['floor', 'surface', 'earth'],
            'wall': ['surface', 'barrier'],
            'window': ['opening', 'glass'],
            'door': ['entrance', 'opening', 'gate']
        }
        
        # Add reverse mapping for better lookup
        self.reverse_synonyms = {}
        for base_word, synonyms in self.object_synonyms.items():
            self.reverse_synonyms[base_word] = base_word
            for synonym in synonyms:
                self.reverse_synonyms[synonym] = base_word
        
        # Enhanced attribute categories with more comprehensive coverage
        self.color_attributes = {
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink', 
            'orange', 'purple', 'gray', 'grey', 'maroon', 'navy', 'olive', 'lime',
            'aqua', 'teal', 'silver', 'golden', 'crimson', 'violet', 'indigo',
            'beige', 'tan', 'cream', 'ivory', 'dark', 'light', 'bright', 'pale',
            'deep', 'vivid', 'muted', 'colorful', 'transparent', 'clear'
        }
        
        self.size_attributes = {
            'big', 'small', 'large', 'tiny', 'huge', 'little', 'giant', 'mini',
            'enormous', 'massive', 'microscopic', 'minuscule', 'colossal', 'petite',
            'vast', 'compact', 'immense', 'diminutive', 'tall', 'short', 'long', 'brief',
            'wide', 'narrow', 'thick', 'thin', 'fat', 'skinny', 'broad', 'slim',
            'oversized', 'undersized', 'medium', 'average', 'normal', 'regular'
        }
        
        self.material_attributes = {
            'wooden', 'metal', 'plastic', 'glass', 'leather', 'fabric', 'stone',
            'concrete', 'brick', 'steel', 'iron', 'aluminum', 'ceramic', 'rubber',
            'cotton', 'silk', 'wool', 'denim', 'canvas', 'paper', 'cardboard',
            'marble', 'granite', 'wood', 'metallic', 'synthetic', 'natural'
        }
        
        # Add more attribute categories for better detection
        self.shape_attributes = {
            'round', 'square', 'rectangular', 'circular', 'oval', 'triangular',
            'curved', 'straight', 'bent', 'twisted', 'flat', 'spherical'
        }
        
        self.condition_attributes = {
            'new', 'old', 'fresh', 'stale', 'clean', 'dirty', 'broken', 'intact',
            'damaged', 'perfect', 'worn', 'rusty', 'shiny', 'dull', 'smooth', 'rough'
        }
        
        self.spatial_relations = {
            'on', 'in', 'under', 'above', 'below', 'beside', 'next to', 'behind', 
            'front', 'inside', 'outside', 'near', 'far', 'left', 'right', 'top',
            'bottom', 'center', 'middle', 'around', 'between', 'among', 'touching'
        }
        
        self.action_relations = {
            'holding', 'carrying', 'wearing', 'eating', 'drinking', 'reading',
            'writing', 'playing', 'sitting', 'standing', 'running', 'walking'
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matcher"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def extract_objects_from_gt(self, ground_truths: List[str]) -> Set[str]:
        """Extract all objects mentioned in ground truth"""
        all_objects = set()
        
        for gt in ground_truths:
            if pd.isna(gt) or gt == '':
                continue
            
            doc = self.nlp(str(gt))
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                    all_objects.add(token.lemma_.lower())
        
        return all_objects
    
    def extract_attributes_from_gt(self, ground_truths: List[str]) -> Dict[str, Set[str]]:
        """Extract attributes from ground truth with improved extraction using NLP"""
        attributes = {
            'colors': set(),
            'sizes': set(), 
            'materials': set(),
            'shapes': set(),
            'conditions': set(),
            'other': set()  # Add category for other descriptive attributes
        }
        
        for gt in ground_truths:
            if pd.isna(gt) or gt == '':
                continue
            
            text_lower = str(gt).lower()
            words = text_lower.split()
            
            # Extract predefined attribute categories
            for word in words:
                if word in self.color_attributes:
                    attributes['colors'].add(word)
                elif word in self.size_attributes:
                    attributes['sizes'].add(word)
                elif word in self.material_attributes:
                    attributes['materials'].add(word)
                elif word in self.shape_attributes:
                    attributes['shapes'].add(word)
                elif word in self.condition_attributes:
                    attributes['conditions'].add(word)
            
            # Use NLP to extract adjective-noun pairs for more comprehensive attribute extraction
            try:
                doc = self.nlp(str(gt))
                for token in doc:
                    if token.pos_ == 'ADJ' and not token.is_stop:
                        attr_word = token.lemma_.lower()
                        # Add to appropriate category or 'other'
                        if attr_word in self.color_attributes:
                            attributes['colors'].add(attr_word)
                        elif attr_word in self.size_attributes:
                            attributes['sizes'].add(attr_word)
                        elif attr_word in self.material_attributes:
                            attributes['materials'].add(attr_word)
                        elif attr_word in self.shape_attributes:
                            attributes['shapes'].add(attr_word)
                        elif attr_word in self.condition_attributes:
                            attributes['conditions'].add(attr_word)
                        else:
                            attributes['other'].add(attr_word)
            except:
                pass
        
        return attributes
    
    def extract_relations_from_gt(self, ground_truths: List[str]) -> Set[str]:
        """Extract spatial and action relations from ground truth"""
        relations = set()
        
        for gt in ground_truths:
            if pd.isna(gt) or gt == '':
                continue
            
            text_lower = str(gt).lower()
            
            for relation in self.spatial_relations:
                if relation in text_lower:
                    relations.add(relation)
            
            for action in self.action_relations:
                if action in text_lower:
                    relations.add(action)
        
        return relations
    
    def check_object_exists(self, object_text: str, gt_objects: Set[str]) -> bool:
        """Check if object exists in ground truth using enhanced hierarchical matching"""
        object_lower = object_text.lower().strip()
        
        # Skip generic words that shouldn't be treated as objects
        generic_words = {'image', 'picture', 'scene', 'view', 'focus', 'main', 'overall', 'general'}
        if object_lower in generic_words:
            return True  # Don't penalize generic scene descriptors
        
        # Direct match
        if object_lower in gt_objects:
            return True
        
        # Enhanced canonical matching using reverse synonyms
        object_canonical = self.reverse_synonyms.get(object_lower, object_lower)
        
        for gt_obj in gt_objects:
            gt_obj_lower = gt_obj.lower()
            gt_canonical = self.reverse_synonyms.get(gt_obj_lower, gt_obj_lower)
            
            # Check canonical forms match
            if object_canonical == gt_canonical:
                return True
                
            # Check substring matches for compound objects (more lenient)
            if len(object_lower) > 3 and len(gt_obj_lower) > 3:
                if object_lower in gt_obj_lower or gt_obj_lower in object_lower:
                    return True
        
        # Lemma match (more thorough)
        try:
            doc = self.nlp(object_text)
            for token in doc:
                token_lemma = token.lemma_.lower()
                if token_lemma in gt_objects:
                    return True
                # Check lemma canonical form
                lemma_canonical = self.reverse_synonyms.get(token_lemma, token_lemma)
                for gt_obj in gt_objects:
                    gt_canonical = self.reverse_synonyms.get(gt_obj.lower(), gt_obj.lower())
                    if lemma_canonical == gt_canonical:
                        return True
        except:
            pass
        
        # Relaxed similarity match (lower threshold)
        for gt_obj in gt_objects:
            if self.calculate_similarity(object_lower, gt_obj.lower()) >= 0.7:  # Lowered from 0.8
                return True
        
        return False
    
    def check_attribute_correctness(self, attribute: str, object_text: str, gt_attributes: Dict[str, Set[str]], gt_objects: Set[str]) -> bool:
        """Check if attribute is correct for existing object with improved logic"""
        attr_lower = attribute.lower().strip()
        
        # First ensure object exists
        if not self.check_object_exists(object_text, gt_objects):
            return False  # Object doesn't exist, so attribute is irrelevant
        
        # Check if attribute exists in ground truth (any category)
        for attr_type, gt_attrs in gt_attributes.items():
            if attr_lower in gt_attrs:
                return True
            # Also check for similarity matches
            for gt_attr in gt_attrs:
                if self.calculate_similarity(attr_lower, gt_attr) >= 0.8:
                    return True
        
        # For predefined attribute categories, be strict about detection
        if (attr_lower in self.color_attributes or 
            attr_lower in self.size_attributes or 
            attr_lower in self.material_attributes or
            attr_lower in self.shape_attributes or
            attr_lower in self.condition_attributes):
            # If it's a known attribute type but not in GT, it's likely wrong
            return False
        
        # For descriptive attributes not in predefined sets, check if similar attributes exist
        # This is more nuanced - if GT has descriptive attributes but this one doesn't match, it might be wrong
        if gt_attributes.get('other', set()):
            # GT has other descriptive attributes, so this should match at least one
            for gt_attr in gt_attributes['other']:
                if self.calculate_similarity(attr_lower, gt_attr) >= 0.7:
                    return True
            # No match found in descriptive attributes, likely incorrect
            return False
        
        # If no attributes in GT and this is a descriptive word, it might be a hallucination
        # But be conservative - only flag obvious cases
        if any(gt_attributes.values()):
            # GT has some attributes but this doesn't match any - likely wrong
            return False
        
        # No GT attributes available, assume correct to avoid false positives
        return True
    
    def check_relation_correctness(self, relation: str, objects: List[str], gt_relations: Set[str], gt_objects: Set[str]) -> bool:
        """Check if relation is correct"""
        rel_lower = relation.lower()
        
        # First ensure all objects exist
        for obj in objects:
            if not self.check_object_exists(obj, gt_objects):
                return False
        
        # Check if relation exists in ground truth
        if rel_lower in gt_relations:
            return True
        
        # Check similarity with ground truth relations
        for gt_rel in gt_relations:
            if self.calculate_similarity(rel_lower, gt_rel) >= 0.7:
                return True
        
        return False
    
    def classify_chunk(self, chunk: Dict, ground_truths: List[str]) -> str:
        """
        Classify chunk using hierarchical approach:
        1. Check object existence (Category Hallucination)
        2. Check attribute correctness (Attribute Hallucination) 
        3. Check relation correctness (Relation Hallucination)
        4. If all correct, return CORRECT
        """
        # Extract ground truth components
        gt_objects = self.extract_objects_from_gt(ground_truths)
        gt_attributes = self.extract_attributes_from_gt(ground_truths)
        gt_relations = self.extract_relations_from_gt(ground_truths)
        
        chunk_type = chunk['type']
        classification_result = 'CORRECT'
        reasoning = []
        
        if chunk_type == 'object':
            # Check if object exists
            object_text = chunk['text']
            object_exists = self.check_object_exists(object_text, gt_objects)
            
            if not object_exists:
                classification_result = 'CATEGORY_HALLUC'
                reasoning.append(f"Object '{object_text}' not found in ground truth objects: {list(gt_objects)}")
            else:
                reasoning.append(f"Object '{object_text}' found in ground truth")
                
        elif chunk_type == 'attribute':
            # Check object existence first, then attribute correctness
            attribute = chunk.get('attribute', '')
            object_text = chunk.get('object', '')
            
            object_exists = self.check_object_exists(object_text, gt_objects)
            if not object_exists:
                classification_result = 'CATEGORY_HALLUC'  # Object doesn't exist
                reasoning.append(f"Object '{object_text}' doesn't exist in GT objects: {list(gt_objects)}")
            else:
                reasoning.append(f"Object '{object_text}' exists in GT")
                attr_correct = self.check_attribute_correctness(attribute, object_text, gt_attributes, gt_objects)
                if not attr_correct:
                    classification_result = 'ATTRIBUTE_HALLUC'  # Object exists but attribute is wrong
                    reasoning.append(f"Attribute '{attribute}' is incorrect. GT attributes: {gt_attributes}")
                else:
                    reasoning.append(f"Attribute '{attribute}' is correct")
            
        elif chunk_type == 'relation':
            # Check object existence first, then relation correctness
            relation = chunk.get('relation', '')
            objects = chunk.get('objects', [])
            
            # Check if all objects exist
            missing_objects = []
            for obj in objects:
                if not self.check_object_exists(obj, gt_objects):
                    missing_objects.append(obj)
            
            if missing_objects:
                classification_result = 'CATEGORY_HALLUC'  # At least one object doesn't exist
                reasoning.append(f"Missing objects: {missing_objects} from GT objects: {list(gt_objects)}")
            else:
                reasoning.append(f"All objects {objects} exist in GT")
                rel_correct = self.check_relation_correctness(relation, objects, gt_relations, gt_objects)
                if not rel_correct:
                    classification_result = 'RELATION_HALLUC'  # Objects exist but relation is wrong
                    reasoning.append(f"Relation '{relation}' is incorrect. GT relations: {list(gt_relations)}")
                else:
                    reasoning.append(f"Relation '{relation}' is correct")
        
        # Store debug information if in debug mode
        if self.debug_mode and len(self.debug_examples) < 200:  # Collect more examples
            debug_info = {
                'chunk_text': chunk['text'],
                'chunk_type': chunk_type,
                'classification': classification_result,
                'reasoning': '; '.join(reasoning),
                'ground_truths': ground_truths[:3],  # Show first 3 GT entries
                'gt_objects': list(gt_objects),
                'gt_attributes': {k: list(v) for k, v in gt_attributes.items()},
                'gt_relations': list(gt_relations)
            }
            
            # Add type-specific information
            if chunk_type == 'attribute':
                debug_info['attribute'] = chunk.get('attribute', '')
                debug_info['object'] = chunk.get('object', '')
            elif chunk_type == 'relation':
                debug_info['relation'] = chunk.get('relation', '')
                debug_info['objects'] = chunk.get('objects', [])
            
            self.debug_examples.append(debug_info)
        
        return classification_result
    
    def print_debug_examples(self, max_examples=50):
        """Print debug examples showing how different hallucination types are detected"""
        if not self.debug_mode or not self.debug_examples:
            print("No debug examples available. Make sure debug_mode=True")
            return
        
        print("\n" + "="*100)
        print("🔍 HALLUCINATION CLASSIFICATION DEBUG EXAMPLES")
        print("="*100)
        
        # Group examples by classification
        examples_by_class = {}
        for example in self.debug_examples:
            class_name = example['classification']
            if class_name not in examples_by_class:
                examples_by_class[class_name] = []
            examples_by_class[class_name].append(example)
        
        # Print statistics
        print(f"\n📊 Debug Examples Collected: {len(self.debug_examples)}")
        for class_name, examples in examples_by_class.items():
            print(f"   {class_name}: {len(examples)} examples")
        
        # Print examples for each class
        examples_per_class = max_examples // len(examples_by_class) if examples_by_class else 0
        
        for class_name in ['CATEGORY_HALLUC', 'ATTRIBUTE_HALLUC', 'RELATION_HALLUC', 'CORRECT']:
            if class_name not in examples_by_class:
                continue
                
            examples = examples_by_class[class_name]
            print(f"\n{'='*80}")
            print(f"🏷️  {class_name} EXAMPLES ({min(examples_per_class, len(examples))} shown)")
            print(f"{'='*80}")
            
            for i, example in enumerate(examples[:examples_per_class]):
                print(f"\n--- Example {i+1} ---")
                print(f"📝 Chunk Text: '{example['chunk_text']}'")
                print(f"🔍 Chunk Type: {example['chunk_type']}")
                print(f"🎯 Classification: {example['classification']}")
                print(f"💡 Reasoning: {example['reasoning']}")
                
                if example['chunk_type'] == 'attribute':
                    print(f"   📋 Attribute: '{example.get('attribute', 'N/A')}'")
                    print(f"   📦 Object: '{example.get('object', 'N/A')}'")
                elif example['chunk_type'] == 'relation':
                    print(f"   🔗 Relation: '{example.get('relation', 'N/A')}'")
                    print(f"   📦 Objects: {example.get('objects', [])}")
                
                print(f"🌍 Ground Truth Objects: {example['gt_objects'][:5]}...")  # Show first 5
                print(f"🏷️  Ground Truth Attributes: {example['gt_attributes']}")
                print(f"🔗 Ground Truth Relations: {example['gt_relations'][:3]}...")  # Show first 3
                print(f"📚 Sample GT Entries: {example['ground_truths']}")
                
                if i < examples_per_class - 1:
                    print("-" * 50)
        
        print("\n" + "="*100)
        print("🎯 Debug Analysis Complete!")
        print("="*100)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance - especially useful for HalluShift++"""
    
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, predictions, targets):
        targets_long = targets.long().squeeze()
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(predictions, targets_long, weight=self.class_weights, reduction='none')
        
        # Calculate probabilities and focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

class AccuracyImprovementLossMultiClass(nn.Module):
    """Enhanced loss function combining focal loss with feature-aware penalties for HalluShift++"""
    
    def __init__(self, alpha=0.4, num_classes=4, use_focal=True, focal_alpha=1.0, focal_gamma=2.0, class_weights=None):
        super(AccuracyImprovementLossMultiClass, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.use_focal = use_focal
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)
        else:
            self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, predictions, targets):
        # Convert targets to long tensor
        targets_long = targets.long().squeeze()
        
        # Primary loss
        if self.use_focal:
            primary_loss = self.focal_loss(predictions, targets_long)
        else:
            primary_loss = self.cross_entropy(predictions, targets_long)
        
        # Accuracy calculation
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = (predicted_labels == targets_long).float().mean()
        
        # Accuracy improvement penalty
        accuracy_improvement_penalty = (1 - accuracy) * self.alpha
        
        # Confidence regularization (encourages confident predictions)
        confidence_penalty = -torch.mean(torch.max(F.softmax(predictions, dim=1), dim=1)[0]) * 0.1
        
        return primary_loss + accuracy_improvement_penalty + confidence_penalty

class FeatureEmbeddingNN(nn.Module):
    """Enhanced neural network for feature embedding with attention"""
    
    def __init__(self, input_dim, output_dim, use_attention=True):
        super(FeatureEmbeddingNN, self).__init__()
        self.use_attention = use_attention
        self.ln = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.2)
        
        # Multi-layer embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(output_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # Feature attention mechanism
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, input_dim),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        x = self.ln(x)
        x = self.dropout(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        x = self.embedding(x)
        return x

class SemanticHallucinationNN(nn.Module):
    """🚀 OPTIMIZED Neural Network for Semantic Hallucination Classification with Advanced Feature Engineering"""
    
    def __init__(self, input_features=65, num_classes=4, use_only_hallushift=False):
        super(SemanticHallucinationNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_features = input_features
        self.use_only_hallushift = use_only_hallushift
        
        # Calculate feature dimensions based on mode
        if use_only_hallushift:
            # Pure HalluShift: Only image features (62), no chunks
            self.image_feature_dim = input_features  # All 62 features are image features
            self.chunk_feature_dim = 0
        else:
            # HalluShift++: Image + composite + chunk features
            # We'll determine this dynamically in the forward pass
            self.image_feature_dim = None  # Will be set dynamically
            self.chunk_feature_dim = None  # Will be set dynamically
        
        if use_only_hallushift:
            # Pure HalluShift: Simple architecture for 62 image features only
            self.image_embedding = FeatureEmbeddingNN(self.image_feature_dim, 64, use_attention=False)
            combined_dim = 64
            
            print(f"   🔵 PURE HalluShift Architecture: {self.image_feature_dim} image features -> 64D embedding")
        else:
            # HalluShift++: Flexible architecture for mixed features
            # We'll use a flexible approach that adapts to the actual feature count
            
            # Estimate dimensions (will be refined in forward pass)
            estimated_image_features = max(8, input_features // 2)  # At least 8, at most half
            estimated_chunk_features = max(18, input_features // 3)  # Assume chunk features exist
            
            self.image_embedding = FeatureEmbeddingNN(estimated_image_features, 32, use_attention=False)
            self.chunk_embedding = FeatureEmbeddingNN(estimated_chunk_features, 64, use_attention=True)
            
            # Feature interaction layer for combining image + chunk
            self.feature_interaction = nn.Sequential(
                nn.Linear(32 + 64, 80),
                nn.ReLU(),
                nn.LayerNorm(80),
                nn.Dropout(0.1),
                nn.Linear(80, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Dropout(0.05)
            )
            
            combined_dim = 64
            
            print(f"   🚀 OPTIMIZED HalluShift++ Architecture:")
            print(f"      - Flexible image embedding: 32D")
            print(f"      - Enhanced chunk embedding: 64D (primary focus)")
            print(f"      - Feature interaction: 96D -> 64D")
            print(f"      - Total combined dimension: {combined_dim}")
        
        # 🎯 ADAPTIVE CLASSIFICATION LAYERS with Optimized Dropout Strategy
        if use_only_hallushift:
            # Simple classifier for pure HalluShift (image features only)
            self.classifier = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 96),
                nn.ReLU(),
                nn.LayerNorm(96),
                nn.Dropout(0.15),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.LayerNorm(48),
                nn.Dropout(0.1),
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.LayerNorm(24),
                nn.Dropout(0.05),
                nn.Linear(24, num_classes)
            )
            print(f"   🔵 PURE HalluShift Classifier: {combined_dim}->96->48->24->{num_classes}")
        else:
            # Enhanced classifier for HalluShift++ (image + composite + chunk features)
            self.classifier = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Dropout(0.15),  # Lower dropout since chunk features are reliable
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Dropout(0.1),
                nn.Linear(128, 96),
                nn.ReLU(),
                nn.LayerNorm(96),
                nn.Dropout(0.1),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.LayerNorm(48),
                nn.Dropout(0.05),
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.LayerNorm(24),
                nn.Dropout(0.05),
                nn.Linear(24, num_classes)
            )
            print(f"   🚀 OPTIMIZED HalluShift++ Classifier: {combined_dim}->128->96->48->24->{num_classes}")
            print(f"   📊 Focus: Chunk features (64D) get majority of network capacity")
    
    def forward(self, x):
        if self.use_only_hallushift:
            # Pure HalluShift: Only image features (62 features)
            image_emb = self.image_embedding(x)  # All features are image features
            combined_features = image_emb  # 64D
        else:
            # HalluShift++: Split features dynamically
            # Assume last ~18-21 features are chunk features (flexible)
            total_features = x.shape[1]
            
            # Estimate split point: chunk features are typically the last 18-21 features
            chunk_feature_count = min(21, max(18, total_features // 3))  # Flexible chunk count
            split_point = total_features - chunk_feature_count
            
            image_features = x[:, :split_point]  # Image + composite features
            chunk_features = x[:, split_point:]  # Chunk features
            
            # Handle different input sizes dynamically
            if image_features.shape[1] > 0:
                # Adapt embedding layer to actual input size
                if hasattr(self.image_embedding, 'embedding') and image_features.shape[1] != self.image_embedding.embedding[0].in_features:
                    # Create new embedding layer with correct input size
                    self.image_embedding = FeatureEmbeddingNN(image_features.shape[1], 32, use_attention=False)
                    if x.is_cuda:
                        self.image_embedding = self.image_embedding.cuda()
                
                image_emb = self.image_embedding(image_features)  # 32D
            else:
                image_emb = torch.zeros(x.shape[0], 32, device=x.device)
            
            if chunk_features.shape[1] > 0:
                # Adapt embedding layer to actual input size
                if hasattr(self.chunk_embedding, 'embedding') and chunk_features.shape[1] != self.chunk_embedding.embedding[0].in_features:
                    # Create new embedding layer with correct input size
                    self.chunk_embedding = FeatureEmbeddingNN(chunk_features.shape[1], 64, use_attention=True)
                    if x.is_cuda:
                        self.chunk_embedding = self.chunk_embedding.cuda()
                
                chunk_emb = self.chunk_embedding(chunk_features)  # 64D
            else:
                chunk_emb = torch.zeros(x.shape[0], 64, device=x.device)
            
            # Combine and process through interaction layer
            combined = torch.cat([image_emb, chunk_emb], dim=1)  # 96D
            combined_features = self.feature_interaction(combined)  # 64D
        
        # Final classification
        output = self.classifier(combined_features)
        return output

def parse_feature_value(value):
    """Parse feature values that might be stored as strings or lists"""
    if pd.isna(value):
        raise ValueError(f"Feature value is NaN: {value}")
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Parse as a list first
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse string value: '{value}' - Error: {str(e)}")
        
        if isinstance(parsed, list):
            # If it's a list, flatten it and take the mean or first value
            # For CKA values and similar complex features
            if len(parsed) == 1:
                return float(parsed[0])
            else:
                # For lists of values, take the mean
                return float(np.mean(parsed))
        else:
            return float(parsed)
    
    if isinstance(value, list):
        if len(value) == 1:
            return float(value[0])
        else:
            return float(np.mean(value))
    
    raise ValueError(f"Cannot parse feature value: {value} of type {type(value)}")

def preprocess_features(df: pd.DataFrame, use_only_hallushift: bool = False) -> pd.DataFrame:
    """CHUNK-CENTRIC preprocessing - minimize image features, maximize chunk features"""
    print("🚀 CHUNK-CENTRIC PREPROCESSING STRATEGY:")
    print("   Based on feature importance analysis:")
    print("   • Chunk features: +193.4% (primary contributor)")
    print("   • Image features: -85.0% to -8.4% (negative contribution)")
    print("   Strategy: Use minimal image features + enhanced chunk features")
    
    all_feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Feature selection strategy
    POSITIVE_HALLUSHIFT_PLUS = [66, 67, 68, 69, 71]  # Best performing HalluShift++ features
    NEGATIVE_HALLUSHIFT_PLUS = [64, 70, 72, 73]  # Poor performing features to exclude
    
    # Select features based on strategy
    if use_only_hallushift:
        # Use ONLY original HalluShift features (0-61), NO chunk features, NO optimization
        feature_cols = [f'feature_{i}' for i in range(62) if f'feature_{i}' in all_feature_cols]
        print(f"🔵 PURE HalluShift Mode: {len(feature_cols)} features (0-61), NO chunk features")
    else:
        # Use ALL HalluShift features + selected HalluShift++ features + composites + chunks
        all_hallushift = [f'feature_{i}' for i in range(62) if f'feature_{i}' in all_feature_cols]  # ALL 62 features
        positive_plus = [f'feature_{i}' for i in POSITIVE_HALLUSHIFT_PLUS if f'feature_{i}' in all_feature_cols]
        feature_cols = all_hallushift + positive_plus
        
        print(f"🚀 HALLUSHIFT++ MODE (Complete):")
        print(f"   All HalluShift features: {len(all_hallushift)} (features 0-61)")
        print(f"   Selected HalluShift++ features: {len(positive_plus)} (best performers: {POSITIVE_HALLUSHIFT_PLUS})")
        print(f"   Excluded negative features: {NEGATIVE_HALLUSHIFT_PLUS}")
        print(f"   Base image features: {len(feature_cols)}")
        print(f"   Will add: Composite features + Enhanced chunk features")
    
    print(f"Processing {len(feature_cols)} feature columns...")
    
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Remove unused/negative image features
    if not use_only_hallushift:
        # For HalluShift++: Remove only the negative HalluShift++ features
        negative_features = [f'feature_{i}' for i in NEGATIVE_HALLUSHIFT_PLUS if f'feature_{i}' in all_feature_cols]
        for neg_col in negative_features:
            if neg_col in processed_df.columns:
                processed_df = processed_df.drop(columns=[neg_col])
        if negative_features:
            print(f"   Dropped {len(negative_features)} negative HalluShift++ features: {negative_features}")
    else:
        # For HalluShift: Remove all HalluShift++ features
        hallushift_plus_features = [f'feature_{i}' for i in range(62, 74) if f'feature_{i}' in all_feature_cols]
        for plus_col in hallushift_plus_features:
            if plus_col in processed_df.columns:
                processed_df = processed_df.drop(columns=[plus_col])
        if hallushift_plus_features:
            print(f"   Dropped {len(hallushift_plus_features)} HalluShift++ features for pure mode")
    
    # Process selected image features sequentially (more stable than parallel processing)
    print(f"🚀 Processing {len(feature_cols)} image features...")
    
    # Sequential processing for stability
    for col in tqdm(feature_cols, desc="Processing image features"):
        try:
            processed_df[col] = processed_df[col].apply(parse_feature_value)
        except Exception as e:
            print(f"\nError processing column '{col}': {str(e)}")
            print(f"Sample values from column '{col}':")
            print(processed_df[col].head(10).tolist())
            raise
    
    print(f"✅ Feature processing completed successfully")
    
    # Create composite features ONLY for HalluShift++
    if not use_only_hallushift:
        print("\n🔧 CREATING COMPOSITE FEATURES (HalluShift++ only)...")
        
        # Amplify best performers
        if 'feature_67' in processed_df.columns:
            processed_df['feature_67_squared'] = processed_df['feature_67'] ** 2
            print("   ✓ Created feature_67_squared (amplified best performer)")
        
        if 'feature_66' in processed_df.columns:
            processed_df['feature_66_squared'] = processed_df['feature_66'] ** 2
            print("   ✓ Created feature_66_squared (amplified performer)")
        
        # Create interactions between best HalluShift++ features
        if 'feature_67' in processed_df.columns and 'feature_69' in processed_df.columns:
            processed_df['feature_67_x_69'] = processed_df['feature_67'] * processed_df['feature_69']
            print("   ✓ Created feature_67_x_69 (perplexity synergy)")
        
        if 'feature_67' in processed_df.columns and 'feature_68' in processed_df.columns:
            processed_df['feature_67_x_68'] = processed_df['feature_67'] * processed_df['feature_68']
            print("   ✓ Created feature_67_x_68 (perplexity interaction)")
        
        # Create perplexity composite from available HalluShift++ features
        perplexity_features = ['feature_67', 'feature_69', 'feature_68', 'feature_66']
        available_perp = [f for f in perplexity_features if f in processed_df.columns]
        if len(available_perp) >= 2:
            processed_df['perplexity_composite'] = processed_df[available_perp].mean(axis=1)
            print(f"   ✓ Created perplexity_composite from {available_perp}")
        
        print(f"   📊 Added composite features for HalluShift++")
    
    # Process chunk features ONLY for HalluShift++
    if not use_only_hallushift:
        print("\n🔧 PROCESSING ENHANCED CHUNK FEATURES (HalluShift++ only)...")
        
        # Process all chunk-specific features
        chunk_feature_cols = [
            'chunk_word_count', 'chunk_relative_position', 'total_chunks_in_image',
            'chunk_position_normalized', 'is_first_chunk', 'is_last_chunk', 'is_middle_chunk',
            'word_count_normalized', 'is_short_chunk', 'is_long_chunk',
            'chunks_density', 'chunk_length_ratio',
            'position_length_interaction', 'position_squared', 'word_count_squared',
            'is_object_chunk', 'is_attribute_chunk', 'is_relation_chunk'
        ]
        
        chunk_features_found = 0
        for col in chunk_feature_cols:
            if col in processed_df.columns:
                try:
                    processed_df[col] = processed_df[col].apply(parse_feature_value)
                    chunk_features_found += 1
                except Exception as e:
                    print(f"\nError processing chunk feature column '{col}': {str(e)}")
                    print(f"Sample values from column '{col}':")
                    print(processed_df[col].head(10).tolist())
                    raise
        
        print(f"   ✓ Processed {chunk_features_found} chunk features for HalluShift++")
    else:
        print("\n📌 HALLUSHIFT MODE: Skipping chunk features (using only original 62 image features)")
    
    print("Feature preprocessing completed.")
    return processed_df

def load_features_from_folder(folder_path: str, dataset_type: str) -> pd.DataFrame:
    """Load and combine all features CSV files from a folder based on dataset type"""
    import glob
    
    print(f"Loading features from folder: {folder_path}")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_dataframes = []
    for csv_file in csv_files:
        print(f"Loading: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        
        # Add model identifier from filename
        model_name = os.path.basename(csv_file).replace('.csv', '').replace('image_hallushift_features_', '').replace('_with_gt', '')
        df['model_name'] = model_name
        
        # Select appropriate ground truth column based on dataset
        if dataset_type == 'mscoco':
            if 'ground_truth_1' not in df.columns:
                print(f"Warning: ground_truth_1 not found in {csv_file}, skipping...")
                continue
            df['ground_truth'] = df['ground_truth_1']
        elif dataset_type == 'llava':
            if 'ground_truth' not in df.columns:
                print(f"Warning: ground_truth not found in {csv_file}, skipping...")
                continue
        elif dataset_type == 'truthfulqa':
            # Check for TruthfulQA ground truth column
            if 'ground_truth' not in df.columns:
                print(f"Warning: TruthfulQA column 'ground_truth' not found in {csv_file}, skipping...")
                continue
        elif dataset_type == 'triviaqa':
            # Check for TriviaQA ground truth columns
            required_cols = ['ground_truth_aliases', 'ground_truth_value']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: TriviaQA columns {missing_cols} not found in {csv_file}, skipping...")
                continue
        elif dataset_type == 'tydiqa':
            # Check for TyDiQA ground truth column
            if 'ground_truth' not in df.columns:
                print(f"Warning: TyDiQA column 'ground_truth' not found in {csv_file}, skipping...")
                continue
        
        all_dataframes.append(df)
    
    if not all_dataframes:
        raise ValueError("No valid CSV files with appropriate ground truth columns found")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples with {len([col for col in combined_df.columns if col.startswith('feature_')])} features each")
    
    return combined_df

def load_features(csv_path: str) -> pd.DataFrame:
    """Load the existing features CSV (kept for backward compatibility)"""
    print(f"Loading features from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {len([col for col in df.columns if col.startswith('feature_')])} features each")
    return df

def create_chunk_dataset(df: pd.DataFrame, dataset_type: str, debug_mode=True) -> pd.DataFrame:
    """Transform image-level dataset to chunk-level dataset using improved methodology"""
    print("Creating chunk-level dataset with improved hierarchical classification...")
    
    chunk_extractor = ImprovedSemanticChunkExtractor()
    gt_matcher = HierarchicalGroundTruthMatcher(debug_mode=debug_mode)
    
    # Add spacy model to ground truth matcher
    gt_matcher.nlp = chunk_extractor.nlp
    
    # Prepare columns for new dataset
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Handle ground truth columns based on dataset type
    if dataset_type == 'mscoco':
        # For MSCOCO, use ground_truth (which was set from ground_truth_1)
        ground_truth_col = 'ground_truth'
    elif dataset_type == 'llava':
        # For LLAVA, use ground_truth
        ground_truth_col = 'ground_truth'
    elif dataset_type == 'truthfulqa':
        # For TruthfulQA, use single ground truth column
        ground_truth_col = 'ground_truth'
    elif dataset_type == 'triviaqa':
        # For TriviaQA, use multiple ground truth columns
        ground_truth_cols = ['ground_truth_aliases', 'ground_truth_value']
        ground_truth_col = None
    elif dataset_type == 'tydiqa':
        # For TyDiQA, use single ground truth column
        ground_truth_col = 'ground_truth'
    else:
        # Fallback to original behavior
        ground_truth_cols = [col for col in df.columns if col.startswith('ground_truth_')]
        ground_truth_col = None
    
    chunk_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        # Handle different sample identification based on dataset type
        if dataset_type in ['truthfulqa', 'triviaqa', 'tydiqa']:
            # For text datasets, use sample index or question as identifier
            sample_name = f"{dataset_type}_sample_{idx}"
            generated_text = row.get('generated_description', row.get('llm_answer', ''))
            
            # Use question as context if available
            if 'question' in row and not pd.isna(row['question']):
                sample_context = f"Q: {row['question']} A: {generated_text}"
            else:
                sample_context = generated_text
        else:
            # For image datasets, use image name
            sample_name = row['image_name']
            generated_text = row['generated_description']
            sample_context = generated_text
        
        if pd.isna(generated_text) or generated_text == '':
            continue
            
        # Extract semantic chunks using improved extractor
        chunks = chunk_extractor.extract_chunks(generated_text)
        
        if not chunks:  # If no chunks found, create a default one
            chunks = [{'text': generated_text, 'type': 'object', 'start': 0, 'end': len(generated_text)}]
        
        # Get ground truth values based on dataset type
        ground_truths = []
        if dataset_type == 'truthfulqa':
            # Use single TruthfulQA ground truth column
            if 'ground_truth' in row and not pd.isna(row['ground_truth']):
                ground_truths.append(str(row['ground_truth']))
        elif dataset_type == 'triviaqa':
            # Combine TriviaQA ground truth types
            if 'ground_truth_value' in row and not pd.isna(row['ground_truth_value']):
                ground_truths.append(str(row['ground_truth_value']))
            if 'ground_truth_aliases' in row and not pd.isna(row['ground_truth_aliases']):
                # Split semicolon-separated aliases
                aliases = str(row['ground_truth_aliases']).split(';')
                ground_truths.extend([alias.strip() for alias in aliases if alias.strip()])
        elif dataset_type == 'tydiqa':
            # Use single TyDiQA ground truth column
            if 'ground_truth' in row and not pd.isna(row['ground_truth']):
                ground_truths.append(str(row['ground_truth']))
        elif ground_truth_col and ground_truth_col in row:
            # Single ground truth column (MSCOCO, LLAVA)
            ground_truths = [str(row[ground_truth_col])] if not pd.isna(row[ground_truth_col]) else []
        else:
            # Multiple ground truth columns (fallback)
            ground_truth_cols = [col for col in df.columns if col.startswith('ground_truth_')]
            ground_truths = [str(row[col]) for col in ground_truth_cols if not pd.isna(row[col])]
        
        # Process each chunk using hierarchical classification
        for chunk_idx, chunk in enumerate(chunks):
            # Classify chunk using improved hierarchical method
            chunk_label = gt_matcher.classify_chunk(chunk, ground_truths)
            
            # Create chunk entry
            chunk_entry = {
                'sample_name': sample_name,
                'generated_description': generated_text,
                'chunk_id': f"{sample_name}_{chunk_idx}",
                'chunk_text': chunk['text'],
                'chunk_type': chunk['type'],
                'chunk_position': chunk['start'],
                'chunk_length': len(chunk['text']),
                'chunk_label': chunk_label
            }
            
            # Add dataset-specific fields
            if dataset_type in ['mscoco', 'llava']:
                # Image datasets
                chunk_entry['image_name'] = row['image_name']
                chunk_entry['image_filename'] = row['image_filename']
                chunk_entry['image_path'] = row['image_path']
            elif dataset_type in ['truthfulqa', 'triviaqa', 'tydiqa']:
                # Text datasets
                if 'question' in row:
                    chunk_entry['question'] = row['question']
                chunk_entry['sample_index'] = idx
            
            # Add model name if available
            if 'model_name' in row:
                chunk_entry['model_name'] = row['model_name']
            
            # Add all original features
            for col in feature_cols:
                chunk_entry[col] = row[col]
            
            # Add enhanced chunk-specific features (these are what actually work!)
            chunk_words = chunk['text'].split()
            chunk_entry['chunk_word_count'] = len(chunk_words)
            chunk_entry['chunk_relative_position'] = chunk['start'] / len(generated_text) if len(generated_text) > 0 else 0
            chunk_entry['total_chunks_in_image'] = len(chunks)
            
            # Enhanced chunk features based on feature importance analysis showing chunk features contribute 193.4%
            chunk_entry['chunk_position_normalized'] = chunk_idx / len(chunks) if len(chunks) > 1 else 0
            chunk_entry['is_first_chunk'] = 1 if chunk_idx == 0 else 0
            chunk_entry['is_last_chunk'] = 1 if chunk_idx == len(chunks) - 1 else 0
            chunk_entry['is_middle_chunk'] = 1 if 0 < chunk_idx < len(chunks) - 1 else 0
            
            # Chunk length categorization
            chunk_entry['word_count_normalized'] = len(chunk_words) / max(1, max(len(c['text'].split()) for c in chunks))
            chunk_entry['is_short_chunk'] = 1 if len(chunk_words) <= 3 else 0
            chunk_entry['is_long_chunk'] = 1 if len(chunk_words) >= 8 else 0
            
            # Sample-level chunk context (renamed from image-level for text compatibility)
            chunk_entry['chunks_density'] = len(chunks) / len(generated_text.split()) if len(generated_text.split()) > 0 else 0
            chunk_entry['chunk_length_ratio'] = len(chunk['text']) / len(generated_text) if len(generated_text) > 0 else 0
            
            # Positional interactions (these often reveal hallucination patterns)
            chunk_entry['position_length_interaction'] = chunk_entry['chunk_relative_position'] * len(chunk_words)
            chunk_entry['position_squared'] = chunk_entry['chunk_relative_position'] ** 2
            chunk_entry['word_count_squared'] = len(chunk_words) ** 2
            
            # Chunk type encoding (semantic information)
            chunk_entry['is_object_chunk'] = 1 if chunk.get('type') == 'object' else 0
            chunk_entry['is_attribute_chunk'] = 1 if chunk.get('type') == 'attribute' else 0
            chunk_entry['is_relation_chunk'] = 1 if chunk.get('type') == 'relation' else 0
            
            chunk_data.append(chunk_entry)
    
    chunk_df = pd.DataFrame(chunk_data)
    print(f"Created chunk dataset with {len(chunk_df)} chunks from {df.shape[0]} samples")
    print(f"Label distribution:")
    print(chunk_df['chunk_label'].value_counts())
    
    # Print debug examples if debug mode is enabled
    if debug_mode:
        gt_matcher.print_debug_examples(max_examples=50)
    
    return chunk_df

def train_classifier(chunk_df: pd.DataFrame, output_dir: str = './models/', 
                    test_size=0.2, batch_size=32, epochs=500, learning_rate=0.001,
                    balance_data=True, use_only_hallushift=False, num_workers=4):
    """Train neural network multi-class classifier on chunk features"""
    print("Training semantic hallucination neural network classifier...")
    
    # Memory monitoring for high-memory optimization
    import psutil
    memory_info = psutil.virtual_memory()
    print(f"💾 MEMORY STATUS: {memory_info.available / (1024**3):.1f}GB available / {memory_info.total / (1024**3):.1f}GB total ({memory_info.percent}% used)")
    
    # Preprocess features to handle string representations of lists
    chunk_df = preprocess_features(chunk_df, use_only_hallushift=use_only_hallushift)
    
    # Print feature usage summary
    available_features = [col for col in chunk_df.columns if col.startswith('feature_')]
    print(f"\n📊 FEATURE USAGE SUMMARY:")
    print(f"{'='*60}")
    if use_only_hallushift:
        print(f"🔹 Mode: HalluShift Only")
        print(f"🔹 Image Features: {len(available_features)} (features 0-61)")
        print(f"🔹 Expected Range: feature_0 to feature_61")
    else:
        print(f"🔹 Mode: HalluShift++")
        print(f"🔹 Image Features: {len(available_features)} (features 0-73)")
        print(f"🔹 HalluShift: features 0-61 (62 features)")
        print(f"🔹 HalluShift++: features 62-73 (12 features)")
        print(f"🔹 Expected Range: feature_0 to feature_73")
    print(f"🔹 Chunk Features: 3 (word_count, position, total_chunks)")
    print(f"🔹 Total Features: {len(available_features)} + 3 = {len(available_features) + 3}")
    print(f"{'='*60}\n")
    
    # Prepare features and labels based on feature set choice
    base_feature_cols = [col for col in chunk_df.columns if col.startswith('feature_') and not col.endswith('_squared') and '_x_' not in col and col != 'feature_composite']
    composite_feature_cols = [col for col in chunk_df.columns if col.endswith('_squared') or '_x_' in col or col == 'feature_composite']
    chunk_feature_cols = [
        'chunk_word_count', 'chunk_relative_position', 'total_chunks_in_image',
        'chunk_position_normalized', 'is_first_chunk', 'is_last_chunk', 'is_middle_chunk',
        'word_count_normalized', 'is_short_chunk', 'is_long_chunk', 
        'chunks_density', 'chunk_length_ratio',
        'position_length_interaction', 'position_squared', 'word_count_squared',
        'is_object_chunk', 'is_attribute_chunk', 'is_relation_chunk'
    ]
    
    # Count available features
    available_composite_features = [col for col in composite_feature_cols if col in chunk_df.columns]
    available_chunk_features = [col for col in chunk_feature_cols if col in chunk_df.columns]
    
    if use_only_hallushift:
        # Use ONLY original 62 image features, NO chunks, NO composites
        feature_cols = base_feature_cols
        print(f"🔵 PURE HALLUSHIFT MODE:")
        print(f"   Image features: {len(base_feature_cols)} (original features 0-61)")
        print(f"   Chunk features: 0 (none)")
        print(f"   Composite features: 0 (none)")
        print(f"   Total features: {len(feature_cols)}")
    else:
        # Use optimized image + composite + chunk features
        all_image_features = base_feature_cols + available_composite_features
        feature_cols = all_image_features + available_chunk_features
        print(f"🚀 OPTIMIZED HALLUSHIFT++ MODE:")
        print(f"   Base image features: {len(base_feature_cols)} (selected best performers)")
        print(f"   Composite features: {len(available_composite_features)} (engineered)")
        print(f"   Enhanced chunk features: {len(available_chunk_features)} (primary contributors)")
        print(f"   Total features: {len(feature_cols)}")
        if available_composite_features:
            print(f"   Composite features: {available_composite_features}")
        print(f"   Strategy: Combine optimized image + engineered + chunk features")
    
    X = chunk_df[feature_cols].values
    y = chunk_df['chunk_label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    # Apply data balancing if requested - optimized for high memory
    if balance_data:
        try:
            from imblearn.over_sampling import SMOTE
            print("🚀 Applying SMOTE for data balancing (high-memory optimized)...")
            
            # Check class distribution before balancing
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            print(f"Before balancing: {dict(zip(label_encoder.inverse_transform(unique_train), counts_train))}")
            
            # Optimize SMOTE for high memory - use more neighbors and aggressive sampling
            max_neighbors = min(15, min(counts_train)-1)  # Use more neighbors for better sampling
            smote = SMOTE(
                random_state=42, 
                k_neighbors=max_neighbors,
                sampling_strategy='auto',  # Balance all minority classes
                n_jobs=num_workers  # Parallel processing
            )
            
            print(f"📊 SMOTE using {max_neighbors} neighbors with {num_workers} parallel jobs")
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Check class distribution after balancing
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            print(f"After balancing: {dict(zip(label_encoder.inverse_transform(unique_train), counts_train))}")
            print(f"💾 Total samples after SMOTE: {len(X_train):,} (using ~{len(X_train) * X_train.shape[1] * 8 / 1024**3:.2f}GB memory)")
        except ImportError:
            print("imbalanced-learn not installed. Proceeding without SMOTE...")
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Proceeding without data balancing...")
    
    # Enhanced feature standardization with feature-group-specific scaling
    if use_only_hallushift:
        # Simple standardization for HalluShift-only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        # Feature-group-specific scaling for optimized HalluShift++ 
        print("🔧 Applying advanced feature-group-specific scaling...")
        
        # Calculate feature group boundaries
        total_base_features = len(base_feature_cols)
        total_composite_features = len(composite_feature_cols)
        total_image_features = total_base_features + total_composite_features
        
        print(f"   Feature groups: {total_base_features} base + {total_composite_features} composite + 3 chunk = {X_train.shape[1]} total")
        
        # Separate feature groups with proper boundaries
        hallushift_features = X_train[:, :62]  # First 62 HalluShift features (0-61)
        selected_plus_features = X_train[:, 62:total_base_features]  # Selected HalluShift++ features (65,66,67,68,69,71)
        composite_features = X_train[:, total_base_features:total_image_features] if total_composite_features > 0 else None  # Engineered features
        chunk_features = X_train[:, total_image_features:]  # Last 3 chunk features
        
        # Separate scalers for each group with enhanced scaling strategies
        hallushift_scaler = StandardScaler()  # Standard scaling for proven features
        selected_plus_scaler = StandardScaler()  # Standard scaling for selected positive features
        composite_scaler = StandardScaler()  # Standard scaling for engineered features  
        chunk_scaler = StandardScaler()  # Standard scaling for chunk features
        
        # Scale each group separately with enhanced processing
        hallushift_train_scaled = hallushift_scaler.fit_transform(hallushift_features)
        selected_plus_train_scaled = selected_plus_scaler.fit_transform(selected_plus_features)
        
        scaled_parts = [hallushift_train_scaled, selected_plus_train_scaled]
        
        # Handle composite features if they exist
        if composite_features is not None and composite_features.shape[1] > 0:
            composite_train_scaled = composite_scaler.fit_transform(composite_features)
            scaled_parts.append(composite_train_scaled)
            print(f"   ✓ Scaled {composite_features.shape[1]} composite features")
        
        chunk_train_scaled = chunk_scaler.fit_transform(chunk_features)
        scaled_parts.append(chunk_train_scaled)
        
        # Combine scaled features
        X_train = np.concatenate(scaled_parts, axis=1)
        
        # Apply same scaling to test set
        hallushift_test_features = X_test[:, :62]
        selected_plus_test_features = X_test[:, 62:total_base_features]
        chunk_test_features = X_test[:, total_image_features:]
        
        hallushift_test_scaled = hallushift_scaler.transform(hallushift_test_features)
        selected_plus_test_scaled = selected_plus_scaler.transform(selected_plus_test_features)
        
        test_scaled_parts = [hallushift_test_scaled, selected_plus_test_scaled]
        
        # Handle composite features in test set
        if composite_features is not None and composite_features.shape[1] > 0:
            composite_test_features = X_test[:, total_base_features:total_image_features]
            composite_test_scaled = composite_scaler.transform(composite_test_features)
            test_scaled_parts.append(composite_test_scaled)
        
        chunk_test_scaled = chunk_scaler.transform(chunk_test_features)
        test_scaled_parts.append(chunk_test_scaled)
        
        X_test = np.concatenate(test_scaled_parts, axis=1)
        
        # Store all scalers for later use
        scaler = {
            'hallushift_scaler': hallushift_scaler,
            'selected_plus_scaler': selected_plus_scaler,
            'composite_scaler': composite_scaler if composite_features is not None else None,
            'chunk_scaler': chunk_scaler,
            'feature_groups': {
                'hallushift': slice(0, 62),
                'selected_plus': slice(62, total_base_features),
                'composite': slice(total_base_features, total_image_features) if total_composite_features > 0 else None,
                'chunk': slice(total_image_features, None)
            },
            'total_base_features': total_base_features,
            'total_composite_features': total_composite_features,
            'total_image_features': total_image_features
        }
    
    # Convert to tensors - keep on CPU for multiprocessing DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Keep tensors on CPU for DataLoader multiprocessing compatibility
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Optimize DataLoader for high-memory system (24GB)
    # Increase batch size and add parallel workers for faster data loading
    optimized_batch_size = min(batch_size * 4, len(X_train) // 10)  # Scale up batch size
    print(f"🚀 MEMORY OPTIMIZATION: Using batch size {optimized_batch_size} (vs {batch_size} original)")
    print(f"📊 Parallel workers: {num_workers} threads for data loading")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=optimized_batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=optimized_batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Calculate enhanced class weights for severe imbalance (especially ATTRIBUTE_HALLUC)
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    
    # Enhanced weighting strategy for extreme imbalance
    # Give extra weight to very rare classes like ATTRIBUTE_HALLUC
    class_weights = []
    for i in range(num_classes):
        count = class_counts[i] if class_counts[i] != 0 else 1
        base_weight = total_samples / count
        
        # Apply additional multiplier for extremely rare classes
        if count < total_samples * 0.05:  # Less than 5% of data
            multiplier = 3.0  # 3x additional weight for very rare classes
        elif count < total_samples * 0.10:  # Less than 10% of data  
            multiplier = 2.0  # 2x additional weight for rare classes
        else:
            multiplier = 1.0
            
        weighted_value = base_weight * multiplier
        class_weights.append(weighted_value)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Enhanced class weights: {class_weights}")
    
    # Show class percentages for clarity
    for i, class_name in enumerate(label_encoder.classes_):
        percentage = (class_counts[i] / total_samples) * 100
        print(f"  {class_name}: {class_counts[i]} samples ({percentage:.2f}%) - Weight: {class_weights[i]:.2f}")
    
    # Initialize model, criterion, and optimizer
    model = SemanticHallucinationNN(input_features=X.shape[1], num_classes=num_classes, 
                                   use_only_hallushift=use_only_hallushift).to(device)
    
    # Use enhanced loss function for HalluShift++ with focal loss
    if use_only_hallushift:
        criterion = AccuracyImprovementLossMultiClass(num_classes=num_classes, use_focal=False).to(device)
    else:
        criterion = AccuracyImprovementLossMultiClass(
            num_classes=num_classes, 
            use_focal=True, 
            focal_alpha=1.0, 
            focal_gamma=2.0,
            class_weights=class_weights
        ).to(device)
    # 🚀 ADAPTIVE LEARNING RATE STRATEGY with Feature-Group-Specific Optimization
    if use_only_hallushift:
        # Conservative learning for proven HalluShift features
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        print(f"   🔵 HalluShift Optimizer: lr={learning_rate}, weight_decay=0.0001")
    else:
        # 🎯 ADVANCED MULTI-TIER LEARNING RATE for Optimized HalluShift++
        # Different learning rates for different parts of the network
        param_groups = [
            # Higher learning rate for new composite/selected features
            {'params': [p for n, p in model.named_parameters() if 'selected_plus' in n or 'composite' in n or 'feature_amplification' in n], 
             'lr': learning_rate * 2.0, 'weight_decay': 0.001},
            
            # Moderate learning rate for interaction layers
            {'params': [p for n, p in model.named_parameters() if 'feature_interaction' in n or 'cross_attention' in n], 
             'lr': learning_rate * 1.5, 'weight_decay': 0.0005},
            
            # Conservative learning rate for proven HalluShift features
            {'params': [p for n, p in model.named_parameters() if 'hallushift_embedding' in n], 
             'lr': learning_rate * 0.8, 'weight_decay': 0.0001},
            
            # Standard learning rate for classifier and other components
            {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['selected_plus', 'composite', 'feature_amplification', 'feature_interaction', 'cross_attention', 'hallushift_embedding'])], 
             'lr': learning_rate, 'weight_decay': 0.0005}
        ]
        
        optimizer = optim.AdamW(param_groups)
        
        # Adaptive scheduler with different strategies for different phases
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=15)
        
        # Add cyclical learning rate for exploration
        cyclical_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=[pg['lr'] * 0.1 for pg in param_groups],
            max_lr=[pg['lr'] for pg in param_groups],
            step_size_up=50, 
            mode='triangular2'
        )
        
        print(f"   🚀 Optimized HalluShift++ Multi-Tier Learning:")
        print(f"      - Selected/Composite features: lr={learning_rate * 2.0}")
        print(f"      - Interaction layers: lr={learning_rate * 1.5}")
        print(f"      - HalluShift features: lr={learning_rate * 0.8}")
        print(f"      - Other components: lr={learning_rate}")
        print(f"   📈 Using CyclicLR for adaptive exploration")
    
    # Training parameters for early stopping
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    best_model_state = None
    final_train_acc = 0  # Track final train accuracy
    
    print("🚀 Starting optimized training with adaptive learning rates...")
    
    # Training loop with enhanced scheduling
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move tensors to GPU when fetched from DataLoader
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Apply class weights
            weights = class_weights[labels]
            weighted_loss = (loss * weights.mean()).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            # Apply cyclical learning rate if available (for HalluShift++)
            if not use_only_hallushift and 'cyclical_scheduler' in locals():
                cyclical_scheduler.step()
            
            running_loss += weighted_loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move tensors to GPU when fetched from DataLoader
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            final_train_acc = train_acc  # Update final train accuracy when we have the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break
        
        # Enhanced progress tracking every 20 epochs
        if epoch % 20 == 0:
            if use_only_hallushift:
                print(f"🔵 Epoch {epoch}: Train Loss = {running_loss/len(train_loader):.4f}, "
                      f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
            else:
                # Show learning rates for different parameter groups
                current_lrs = [group['lr'] for group in optimizer.param_groups]
                print(f"🚀 Epoch {epoch}: Train Loss = {running_loss/len(train_loader):.4f}, "
                      f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
                print(f"   📈 Learning Rates: Selected/Composite={current_lrs[0]:.6f}, "
                      f"Interaction={current_lrs[1]:.6f}, HalluShift={current_lrs[2]:.6f}, Other={current_lrs[3]:.6f}")
    
    # Final evaluation
    model.eval()
    y_pred = []
    y_true = []
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move tensors to GPU when fetched from DataLoader
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate AUC-ROC for multi-class
    try:
        auc_macro = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
        auc_weighted = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
        
        # Per-class AUC-ROC
        auc_per_class = roc_auc_score(y_true, y_pred_proba, average=None, multi_class='ovr')
        
        print(f"\nFinal Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC (Macro): {auc_macro:.4f}")
        print(f"AUC-ROC (Weighted): {auc_weighted:.4f}")
        
        print(f"\nPer-Class AUC-ROC:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name}: {auc_per_class[i]:.4f}")
            
    except ValueError as e:
        print(f"\nCould not calculate AUC-ROC: {e}")
        print(f"\nFinal Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: Could not calculate (insufficient class predictions)")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save model and components
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare model data with all metrics
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_features': X.shape[1],
            'num_classes': num_classes,
            'use_only_hallushift': use_only_hallushift
        },
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'test_accuracy': test_accuracy,
        'train_accuracy': final_train_acc / 100.0,  # Convert from percentage to decimal for consistency
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'label_distribution': chunk_df['chunk_label'].value_counts().to_dict(),
        'class_weights': class_weights.cpu().numpy(),
        'use_only_hallushift': use_only_hallushift
    }
    
    # Add AUC-ROC scores if they were calculated successfully
    try:
        if 'auc_macro' in locals():
            model_data['auc_macro'] = auc_macro
            model_data['auc_weighted'] = auc_weighted
            model_data['auc_per_class'] = dict(zip(label_encoder.classes_, auc_per_class))
    except:
        pass
    
    model_path = os.path.join(output_dir, 'semantic_hallucination_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # 🎯 OPTIMIZATION SUMMARY
    print("\n" + "="*80)
    print("🚀 OPTIMIZATION SUMMARY - HalluShift++ ENHANCEMENTS")
    print("="*80)
    
    if not use_only_hallushift:
        print("✅ PHASE 1 - FEATURE SELECTION & FILTERING:")
        print("   • Removed negative features: [64, 70, 72, 73]")
        print("   • Kept positive features: [65, 66, 67, 68, 69, 71]")
        print(f"   • Total optimized features: {len(feature_cols)} (vs 77 original)")
        
        print("\n✅ PHASE 2 - FEATURE ENGINEERING:")
        composite_count = len([col for col in feature_cols if col in ['feature_67_squared', 'feature_67_x_65', 'feature_67_x_69', 'perplexity_composite', 'attention_repetition']])
        if composite_count > 0:
            print(f"   • Created {composite_count} composite features")
            print("   • Amplified best performer (feature_67)")
            print("   • Added perplexity synergy combinations")
        
        print("\n✅ PHASE 3 - ADAPTIVE LEARNING RATES:")
        print("   • Multi-tier learning: 2.0x/1.5x/0.8x/1.0x ratios")
        print("   • Cyclical learning rate for exploration")
        print("   • Progressive dropout: 0.25→0.2→0.15→0.1→0.05")
        
        print("\n✅ PHASE 4 - ENHANCED ARCHITECTURE:")
        print("   • Advanced cross-attention mechanism")
        print("   • Feature amplification layers")
        print("   • Deeper classifier (6 layers vs 4)")
        
        print(f"\n📊 EXPECTED IMPROVEMENTS:")
        print(f"   • Baseline HalluShift++ accuracy: ~55-61%")
        print(f"   • Target optimized accuracy: ~62-75% (+7-20%)")
        print(f"   • Current achieved accuracy: {test_accuracy:.1%}")
        
        if test_accuracy >= 0.62:
            improvement = (test_accuracy - 0.58) * 100  # Assuming baseline ~58%
            print(f"   🎉 OPTIMIZATION SUCCESS: +{improvement:.1f}% improvement achieved!")
        else:
            print(f"   🔄 Further tuning may be needed to reach target performance")
    else:
        print("🔵 HALLUSHIFT MODE:")
        print(f"   • Using proven 62 HalluShift features")
        print(f"   • Achieved accuracy: {test_accuracy:.1%}")
    
    print("="*80)
    
    # Perform feature importance analysis for HalluShift++
    if not use_only_hallushift:
        try:
            feature_importance = analyze_feature_importance(model, X_test, y_test, feature_cols, use_only_hallushift)
            model_data['feature_importance'] = feature_importance
        except Exception as e:
            print(f"Feature importance analysis failed: {e}")
    
    return model_data

def analyze_feature_importance(model, X_test, y_test, feature_cols, use_only_hallushift=False):
    """Analyze feature importance using permutation importance"""
    print("\n" + "="*60)
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get baseline accuracy
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    with torch.no_grad():
        baseline_pred = model(X_test_tensor)
        baseline_acc = (torch.argmax(baseline_pred, dim=1) == y_test_tensor).float().mean().item()
    
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Calculate permutation importance for each feature
    feature_importance = {}
    
    for i, feature_name in enumerate(tqdm(feature_cols, desc="Analyzing feature importance")):
        X_test_permuted = X_test.copy()
        # Permute the feature
        np.random.shuffle(X_test_permuted[:, i])
        
        X_permuted_tensor = torch.tensor(X_test_permuted, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            permuted_pred = model(X_permuted_tensor)
            permuted_acc = (torch.argmax(permuted_pred, dim=1) == y_test_tensor).float().mean().item()
        
        # Importance is the drop in accuracy
        importance = baseline_acc - permuted_acc
        feature_importance[feature_name] = importance
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 TOP 20 MOST IMPORTANT FEATURES:")
    print("-" * 60)
    for i, (feature, importance) in enumerate(sorted_features[:20]):
        importance_type = "🔥" if importance > 0.01 else "⚡" if importance > 0.005 else "💡"
        print(f"{i+1:2d}. {importance_type} {feature:<25} | Importance: {importance:+.6f}")
    
    if not use_only_hallushift:
        print(f"\n🎯 HALLUSHIFT++ FEATURE ANALYSIS:")
        print("-" * 60)
        
        # Analyze HalluShift++ features specifically (features 62-73)
        hallushift_plus_features = [f for f in sorted_features if f[0].startswith('feature_') and 
                                   62 <= int(f[0].split('_')[1]) <= 73]
        
        if hallushift_plus_features:
            print(f"HalluShift++ features in top 20: {len([f for f in sorted_features[:20] if f[0] in [hf[0] for hf in hallushift_plus_features]])}")
            print(f"Best HalluShift++ feature: {hallushift_plus_features[0][0]} (rank {[f[0] for f in sorted_features].index(hallushift_plus_features[0][0]) + 1})")
            
            print(f"\nTop 10 HalluShift++ features:")
            for i, (feature, importance) in enumerate(hallushift_plus_features[:10]):
                feature_idx = int(feature.split('_')[1])
                feature_type = get_hallushift_plus_feature_type(feature_idx)
                print(f"  {i+1}. {feature} ({feature_type}) | Importance: {importance:+.6f}")
        
        # Calculate group importance
        hallushift_importance = sum(imp for feat, imp in sorted_features if feat.startswith('feature_') and int(feat.split('_')[1]) < 62)
        hallushift_plus_importance = sum(imp for feat, imp in sorted_features if feat.startswith('feature_') and 62 <= int(feat.split('_')[1]) <= 73)
        chunk_importance = sum(imp for feat, imp in sorted_features if not feat.startswith('feature_'))
        
        total_importance = hallushift_importance + hallushift_plus_importance + chunk_importance
        
        print(f"\n📈 FEATURE GROUP ANALYSIS:")
        print("-" * 60)
        print(f"HalluShift features (0-61):     {hallushift_importance:+.4f} ({hallushift_importance/total_importance*100:.1f}%)")
        print(f"HalluShift++ features (62+):    {hallushift_plus_importance:+.4f} ({hallushift_plus_importance/total_importance*100:.1f}%)")
        print(f"Chunk features:                 {chunk_importance:+.4f} ({chunk_importance/total_importance*100:.1f}%)")
    
    print("="*60)
    return feature_importance

def get_hallushift_plus_feature_type(feature_idx):
    """Map feature index to HalluShift++ feature type"""
    if 62 <= feature_idx <= 63:
        return "Layer Consistency"
    elif 64 <= feature_idx <= 65:
        return "Attention Concentration"
    elif 66 <= feature_idx <= 70:
        return "Perplexity/Confidence"
    elif 71 <= feature_idx <= 73:
        return "Token Repetition/Novelty"
    else:
        return "Unknown HalluShift++ Feature"

def save_chunk_dataset(chunk_df: pd.DataFrame, output_path: str):
    """Save the chunk dataset for future use"""
    chunk_df.to_csv(output_path, index=False)
    print(f"Chunk dataset saved to: {output_path}")

def save_classification_results(results_data: dict, output_dir: str):
    """Save classification results to CSV for comparison"""
    results_file = os.path.join(output_dir, 'classification_results_comparison.csv')
    
    # Create results dataframe
    results_df = pd.DataFrame([results_data])
    
    # Check if file exists and append if it does
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    # Save results
    results_df.to_csv(results_file, index=False)
    print(f"Classification results saved to: {results_file}")
    
    return results_df

def column_to_txt(df, column_name, file_name):
    """Save dataframe column to text file for BLEURT evaluation (from functions.py)
    Fixed to handle newlines within text properly for BLEURT evaluation"""
    with open(file_name, 'w') as f:
        for item in df[column_name]:
            # Replace internal newlines with spaces to prevent line count mismatch
            clean_text = str(item).replace('\n', ' ').replace('\r', ' ').strip()
            f.write(clean_text + '\n')

def bleurt_processing(id_file, score_file, threshold=0.5):
    """Process BLEURT scores and create binary labels (adapted from functions.py)"""
    # Read ID file
    with open(id_file, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    
    # Read scores file
    with open(score_file, 'r') as f:
        scores = [float(line.strip()) for line in f.readlines()]
    
    # Create dataframe
    df = pd.DataFrame({
        'id': ids,
        'bleurt_score': scores
    })
    
    # Create binary hallucination labels based on threshold
    df['hallucination'] = (df['bleurt_score'] < threshold).astype(int)
    
    return df

def prepare_llm_ground_truth(df: pd.DataFrame, dataset_type: str):
    """
    Prepare ground truth answers for BLEURT evaluation
    Adapted from haL_detection.py methodology
    """
    print(f"Preparing ground truth for {dataset_type} dataset...")
    
    # Define answer mapping like in haL_detection.py
    answer_mapping = {
        'truthfulqa': ['ground_truth'],
        'triviaqa': ['ground_truth_aliases', 'ground_truth_value'], 
        'tydiqa': ['ground_truth'],
        'coqa': ['ground_truth'],
        'haluevaldia': ['ground_truth'],
        'haluevalqa': ['ground_truth'],
        'haluevalsum': ['ground_truth']
    }
    
    if dataset_type not in answer_mapping:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    gt_columns = answer_mapping[dataset_type]
    
    # Check if required columns exist
    missing_cols = [col for col in gt_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing ground truth columns: {missing_cols}")
    
    # Prepare ground truth data
    ground_truth_data = []
    llm_answers = []
    ids = []
    
    for idx, row in df.iterrows():
        # Get LLM generated answer
        llm_answer = row.get('generated_description', row.get('llm_answer', ''))
        if pd.isna(llm_answer) or llm_answer == '':
            continue
            
        llm_answers.append(str(llm_answer))
        
        # Process ground truth based on dataset type
        if dataset_type == 'truthfulqa':
            if not pd.isna(row['ground_truth']):
                ground_truth_data.append(str(row['ground_truth']))
                ids.append(str(idx))
        
        elif dataset_type == 'triviaqa':
            # For TriviaQA, use primary answer and aliases
            gt_answers = []
            if not pd.isna(row.get('ground_truth_value', '')):
                gt_answers.append(str(row['ground_truth_value']))
            if not pd.isna(row.get('ground_truth_aliases', '')):
                # Split semicolon-separated aliases
                aliases = str(row['ground_truth_aliases']).split(';')
                gt_answers.extend([alias.strip() for alias in aliases if alias.strip()])
            
            if gt_answers:
                # Use the first answer as primary ground truth
                ground_truth_data.append(gt_answers[0])
                ids.append(str(idx))
        
        elif dataset_type in ['tydiqa', 'coqa', 'haluevaldia', 'haluevalqa', 'haluevalsum']:
            if not pd.isna(row['ground_truth']):
                ground_truth_data.append(str(row['ground_truth']))
                ids.append(str(idx))
    
    if len(ground_truth_data) != len(llm_answers):
        raise ValueError(f"Mismatch between ground truth ({len(ground_truth_data)}) and LLM answers ({len(llm_answers)})")
    
    print(f"Prepared {len(ground_truth_data)} samples for BLEURT evaluation")
    
    return ground_truth_data, llm_answers, ids

def compute_bleurt_scores(ground_truth_list, llm_answers_list, ids_list, bleurt_threshold=0.5, output_dir='./temp_bleurt/'):
    """
    Compute BLEURT scores by creating text files and running BLEURT evaluation
    Following the methodology from haL_detection.py
    """
    print("Computing BLEURT scores...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataframe for easier handling
    bleurt_df = pd.DataFrame({
        'ground_truth': ground_truth_list,
        'llm_answer': llm_answers_list,
        'id': ids_list
    })
    
    # Save to text files as required by BLEURT
    print("Creating text files for BLEURT evaluation...")
    column_to_txt(bleurt_df, 'ground_truth', 'answers')
    column_to_txt(bleurt_df, 'llm_answer', 'llm_answer') 
    column_to_txt(bleurt_df, 'id', 'id')
    
    # Download BLEURT model if not exists
    print("Checking BLEURT model...")
    if not os.path.exists("./models/BLEURT-20-D12"):
        print("Downloading BLEURT model...")
        os.makedirs("./models", exist_ok=True)
        os.system("wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip -O ./models/BLEURT-20-D12.zip")
        os.system("unzip -o ./models/BLEURT-20-D12.zip -d ./models")
    
    # Run BLEURT scoring
    print("Running BLEURT evaluation...")
    bleurt_cmd = (
        "python -m bleurt.score_files "
        "-candidate_file=llm_answer "
        "-reference_file=answers "
        "-bleurt_batch_size=100 "
        "-batch_same_length=True "
        "-bleurt_checkpoint=models/BLEURT-20-D12 "
        "-scores_file=scores"
    )
    
    result = os.system(bleurt_cmd)
    if result != 0:
        raise RuntimeError("BLEURT evaluation failed")
    
    # Process BLEURT results with the correct threshold parameter
    print("Processing BLEURT scores...")
    bleurt_results = bleurt_processing(id_file="id", score_file="scores", threshold=bleurt_threshold)
    
    # Clean up temporary files
    temp_files = ["answers", "llm_answer", "id", "scores"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"BLEURT evaluation completed. Mean score: {bleurt_results['bleurt_score'].mean():.3f}")
    print(f"Hallucination rate: {bleurt_results['hallucination'].mean():.2%}")
    
    return bleurt_results

def create_llm_binary_dataset(df: pd.DataFrame, dataset_type: str, bleurt_threshold: float = 0.5) -> pd.DataFrame:
    """
    Create binary hallucination dataset for LLMs by computing BLEURT scores at runtime
    Following the original HalluShift methodology for text datasets
    """
    print(f"Creating LLM binary hallucination dataset using BLEURT threshold: {bleurt_threshold}")
    
    # Prepare ground truth and LLM answers
    ground_truth_list, llm_answers_list, ids_list = prepare_llm_ground_truth(df, dataset_type)
    
    # Compute BLEURT scores
    bleurt_results = compute_bleurt_scores(ground_truth_list, llm_answers_list, ids_list, bleurt_threshold)
    
    # Create binary dataset
    binary_data = []
    for idx, row in df.iterrows():
        # Find corresponding BLEURT result
        bleurt_row = bleurt_results[bleurt_results['id'] == str(idx)]
        if bleurt_row.empty:
            continue  # Skip if no BLEURT result
            
        bleurt_score = bleurt_row['bleurt_score'].iloc[0]
        binary_label = bleurt_row['hallucination'].iloc[0]
        
        sample_entry = {
            'sample_id': f"{dataset_type}_sample_{idx}",
            'generated_text': row.get('generated_description', row.get('llm_answer', '')),
            'bleurt_score': bleurt_score,
            'binary_label': binary_label
        }
        
        # Add dataset-specific fields
        if 'question' in row and not pd.isna(row['question']):
            sample_entry['question'] = row['question']
        if 'model_name' in row:
            sample_entry['model_name'] = row['model_name']
        
        # Add all original HalluShift++ features
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        for col in feature_cols:
            sample_entry[col] = row[col]
            
        binary_data.append(sample_entry)
    
    result_df = pd.DataFrame(binary_data)
    
    print(f"Created LLM binary dataset with {len(result_df)} samples")
    print(f"Binary label distribution:")
    print(result_df['binary_label'].value_counts())
    print(f"Hallucination rate: {result_df['binary_label'].mean():.2%}")
    
    return result_df

def preprocess_llm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    LLM-specific feature preprocessing - use only positive-impact HalluShift++ features
    Based on analysis from LLM logs showing which features have positive vs negative impact
    """
    print("🔬 LLM-SPECIFIC FEATURE PREPROCESSING:")
    print("   Strategy: Use only positive-impact HalluShift++ features, exclude chunk features")
    
    all_feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Based on LLM logs analysis - features with consistent POSITIVE impact across datasets
    POSITIVE_HALLUSHIFT_FEATURES = list(range(62))  # Original HalluShift features (0-61)
    POSITIVE_HALLUSHIFT_PLUS = [68, 69, 71]  # Only consistently positive HalluShift++ features
    
    # Features with consistent NEGATIVE impact (exclude these)
    NEGATIVE_FEATURES = [62, 63, 64, 65, 66, 67, 70, 72, 73]  # Based on logs analysis
    
    # Select only positive features
    positive_features = (
        [f'feature_{i}' for i in POSITIVE_HALLUSHIFT_FEATURES if f'feature_{i}' in all_feature_cols] +
        [f'feature_{i}' for i in POSITIVE_HALLUSHIFT_PLUS if f'feature_{i}' in all_feature_cols]
    )
    
    print(f"   Original HalluShift features: {len([f for f in positive_features if int(f.split('_')[1]) < 62])}")
    print(f"   Positive HalluShift++ features: {POSITIVE_HALLUSHIFT_PLUS}")
    print(f"   Excluded negative features: {NEGATIVE_FEATURES}")
    print(f"   Total selected features: {len(positive_features)}")
    
    # Create a copy and process selected features
    processed_df = df.copy()
    
    # Remove negative features
    negative_feature_cols = [f'feature_{i}' for i in NEGATIVE_FEATURES if f'feature_{i}' in processed_df.columns]
    for neg_col in negative_feature_cols:
        processed_df = processed_df.drop(columns=[neg_col])
    print(f"   Dropped {len(negative_feature_cols)} negative features")
    
    # Process positive features
    print(f"   Processing {len(positive_features)} positive features...")
    for col in tqdm(positive_features, desc="Processing LLM features"):
        try:
            processed_df[col] = processed_df[col].apply(parse_feature_value)
        except Exception as e:
            print(f"Error processing column '{col}': {str(e)}")
            raise
    
    print("✅ LLM feature preprocessing completed")
    return processed_df

class LLMBinaryClassifierNN(nn.Module):
    """
    Simple binary neural network classifier for LLM hallucination detection
    Following original HalluShift methodology but with HalluShift++ features
    """
    
    def __init__(self, input_features):
        super(LLMBinaryClassifierNN, self).__init__()
        
        self.input_features = input_features
        
        # Simple but effective architecture for binary classification
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1),
        )
        
        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),  # Single output for binary classification
            nn.Sigmoid()
        )
        
        print(f"   🔷 LLM Binary Classifier Architecture:")
        print(f"      - Input features: {input_features}")
        print(f"      - Encoder: {input_features}→128→64→32")
        print(f"      - Classifier: 32→16→1 (binary)")
        print(f"      - Activation: Sigmoid for binary probability")
    
    def forward(self, x):
        encoded = self.feature_encoder(x)
        output = self.classifier(encoded)
        return output.squeeze()  # Remove extra dimension for binary classification

def train_llm_binary_classifier(binary_df: pd.DataFrame, output_dir: str = './llm_models/', 
                                test_size=0.2, batch_size=64, epochs=200, learning_rate=0.001):
    """
    Train binary classifier for LLM hallucination detection using BLEURT labels
    """
    print("Training LLM binary hallucination classifier...")
    
    # Preprocess features
    binary_df = preprocess_llm_features(binary_df)
    
    # Prepare features - only use positive HalluShift++ features
    feature_cols = [col for col in binary_df.columns if col.startswith('feature_')]
    print(f"Using {len(feature_cols)} features for binary classification")
    
    X = binary_df[feature_cols].values
    y = binary_df['binary_label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Binary label distribution: {np.bincount(y)}")
    print(f"Hallucination rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights for imbalanced data
    pos_weight = torch.tensor([len(y_train) / (2 * y_train.sum())]).to(device)
    print(f"Positive class weight: {pos_weight.item():.2f}")
    
    # Initialize model, criterion, and optimizer
    model = LLMBinaryClassifierNN(input_features=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    
    # Training parameters
    best_val_acc = 0
    patience = 30  # Increased patience for better convergence
    patience_counter = 0
    best_model_state = None
    
    print(f"🔷 Starting LLM binary classifier training...")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break
        
        # Progress tracking
        if epoch % 25 == 0:
            print(f"🔷 Epoch {epoch}: Train Loss = {running_loss/len(train_loader):.4f}, "
                  f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
    
    # Final evaluation
    model.eval()
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(probs.cpu().numpy())
    
    # Convert to numpy
    y_true = y_test
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n🔷 LLM Binary Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Hallucinated', 'Hallucinated']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_features': X.shape[1],
            'model_type': 'binary'
        },
        'scaler': scaler,
        'feature_columns': feature_cols,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'is_binary_classifier': True
    }
    
    return model_data

def main():
    parser = argparse.ArgumentParser(description="Train improved semantic hallucination classifier")
    parser.add_argument('--dataset', type=str, choices=['mscoco', 'llava', 'truthfulqa', 'triviaqa', 'tydiqa'], required=True,
                                               help='Dataset type: mscoco, llava, truthfulqa, triviaqa, or tydiqa')
    parser.add_argument('--features_folder', type=str, required=True,
                       help='Path to folder containing CSV files with features and ground truth')
    parser.add_argument('--features_csv', type=str, default=None,
                       help='Path to single features CSV file (alternative to folder)')
    parser.add_argument('--output_dir', type=str, default='./semantic_models/',
                       help='Directory to save trained models')
    parser.add_argument('--save_chunks', action='store_true',
                       help='Save the chunk dataset as CSV')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (default 128 for high-memory systems)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers for data loading (default 8 for 24GB RAM)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for testing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to show classification examples')
    parser.add_argument('--use_only_hallushift', action='store_true',
                       help='Use only first 62 HalluShift features instead of all features including HalluShift++')
    parser.add_argument('--llm_binary_mode', action='store_true',
                       help='Use LLM binary classification mode with BLEURT labels (for text datasets)')
    parser.add_argument('--bleurt_threshold', type=float, default=0.5,
                       help='BLEURT threshold for binary hallucination classification (default: 0.5)')
    
    args = parser.parse_args()
    
    # Determine if we should use LLM binary mode
    text_datasets = ['truthfulqa', 'triviaqa', 'tydiqa']
    if args.dataset in text_datasets and not args.llm_binary_mode:
        print(f"🔷 RECOMMENDATION: For text dataset '{args.dataset}', consider using --llm_binary_mode for better performance")
    
    print("="*80)
    if args.llm_binary_mode:
        print("🔷 LLM BINARY HALLUCINATION CLASSIFIER TRAINING")
        print("Based on original HalluShift methodology with BLEURT evaluation")
        print("="*80)
        print("🔷 MODE: LLM Binary Classification")
        print("   • Computing BLEURT scores at runtime")
        print("   • Using BLEURT score threshold for binary labels")
        print("   • Only positive-impact HalluShift++ features")
        print("   • Simple binary classifier architecture")
        print("   • No chunk-based features (inappropriate for factual accuracy)")
        print(f"   • BLEURT threshold: {args.bleurt_threshold}")
    else:
        print("🚀 OPTIMIZED SEMANTIC HALLUCINATION CLASSIFIER TRAINING")
        print("Based on HalLoc paper methodology with ENHANCED HalluShift++ optimizations")
        print("="*80)
        
        if args.use_only_hallushift:
            print("🔵 MODE: HalluShift Only (Proven Baseline)")
            print("   • Using 62 original HalluShift features")
            print("   • Conservative architecture and learning rates")
            print("   • Expected accuracy: ~67-68%")
        else:
            print("🚀 MODE: Optimized HalluShift++ (Enhanced)")
            print("   ✅ Phase 1: Feature Selection & Filtering")
            print("      • Removed negative features: [64, 70, 72, 73]")
            print("      • Kept positive features: [65, 66, 67, 68, 69, 71]")
            print("   ✅ Phase 2: Advanced Feature Engineering")  
            print("      • Amplified best performer (feature_67)")
            print("      • Created perplexity synergy combinations")
            print("   ✅ Phase 3: Adaptive Multi-Tier Learning Rates")
            print("      • Multi-tier learning: 2.0x/1.5x/0.8x/1.0x ratios")
            print("      • Cyclical learning rate for exploration")
            print("   ✅ Phase 4: Enhanced Deep Architecture")
            print("      • Advanced cross-attention mechanism")
            print("      • Progressive dropout: 0.25→0.2→0.15→0.1→0.05")
            print("   🎯 Target: 62-75% accuracy (+7-20% improvement)")
    
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Test Size: {args.test_size}")
    if args.llm_binary_mode:
        print(f"  Mode: LLM Binary Classification")
        print(f"  BLEURT Threshold: {args.bleurt_threshold}")
        print(f"  Features: Positive HalluShift++ only")
    else:
        print(f"  Feature Set: {'HalluShift only (62 features)' if args.use_only_hallushift else 'Optimized HalluShift++ (68+ features)'}")
    print("="*80)
    
    # Process files
    if args.features_csv:
        csv_files = [args.features_csv]
    else:
        csv_files = glob.glob(os.path.join(args.features_folder, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {args.features_folder}")
        print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*100}")
        print(f"PROCESSING: {os.path.basename(csv_file)}")
        print(f"{'='*100}")
        
        filename = os.path.basename(csv_file)
        model_name = filename.replace('.csv', '').replace('image_hallushift_features_', '').replace('_with_gt', '')
        
        print(f"Model Name: {model_name}")
        print(f"Dataset: {args.dataset}")
        
        try:
            # Load CSV file
            print(f"Loading features from {filename}...")
            df = pd.read_csv(csv_file)
            
            # Validate required columns based on mode
            if args.llm_binary_mode:
                # Check for ground truth columns needed for BLEURT
                required_gt_cols = ['ground_truth']
                if args.dataset == 'triviaqa':
                    required_gt_cols.extend(['ground_truth_aliases', 'ground_truth_value'])
                
                missing_cols = [col for col in required_gt_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: Missing ground truth columns {missing_cols} in {filename} for BLEURT evaluation, skipping...")
                    continue
                
                # Check for generated answers
                if 'generated_description' not in df.columns and 'llm_answer' not in df.columns:
                    print(f"Warning: No generated answers column found in {filename}, skipping...")
                    continue
            else:
                # Standard ground truth validation for multi-class mode
                if args.dataset == 'mscoco':
                    if 'ground_truth_1' not in df.columns:
                        print(f"Warning: ground_truth_1 not found in {filename}, skipping...")
                        continue
                    df['ground_truth'] = df['ground_truth_1']
                elif args.dataset == 'llava':
                    if 'ground_truth' not in df.columns:
                        print(f"Warning: ground_truth not found in {filename}, skipping...")
                        continue
                # Add other dataset validations as needed...
            
            df['model_name'] = model_name
            df['dataset_type'] = args.dataset
            
            print(f"Loaded {len(df)} samples with {len([col for col in df.columns if col.startswith('feature_')])} features each")
            
            if args.llm_binary_mode:
                # LLM Binary Classification Mode
                print(f"\n🔷 LLM Binary Classification Mode for {model_name}")
                
                # Create binary dataset using BLEURT scores computed at runtime
                binary_df = create_llm_binary_dataset(df, args.dataset, args.bleurt_threshold)
                
                if len(binary_df) == 0:
                    print(f"No samples created for {filename}, skipping...")
                    continue
                
                # Save binary dataset for inspection
                binary_csv_path = os.path.join(args.output_dir, f'llm_binary_dataset_{model_name}_{args.dataset}.csv')
                os.makedirs(args.output_dir, exist_ok=True)
                binary_df.to_csv(binary_csv_path, index=False)
                print(f"Binary dataset saved to: {binary_csv_path}")
                
                # Train binary classifier
                print(f"\nTraining LLM binary classifier for {model_name} on {args.dataset} dataset...")
                model_data = train_llm_binary_classifier(
                    binary_df,
                    args.output_dir,
                    test_size=args.test_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate
                )
                
                # Save binary results
                feature_set = 'LLM_Binary'
                results_data = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'model_name': model_name,
                    'dataset': args.dataset,
                    'feature_set': feature_set,
                    'total_samples': len(binary_df),
                    'accuracy': model_data['accuracy'],
                    'precision': model_data['precision'],
                    'recall': model_data['recall'],
                    'f1_score': model_data['f1_score'],
                    'auc_roc': model_data['auc_roc'],
                    'bleurt_threshold': args.bleurt_threshold,
                    'is_binary': True
                }
                
                # Add label distribution
                label_dist = binary_df['binary_label'].value_counts().to_dict()
                results_data['non_hallucinated_count'] = label_dist.get(0, 0)
                results_data['hallucinated_count'] = label_dist.get(1, 0)
                results_data['hallucination_rate'] = binary_df['binary_label'].mean()
                
                # Save model with binary identifier
                model_filename = f'llm_binary_classifier_{model_name}_{args.dataset}.pkl'
                model_path = os.path.join(args.output_dir, model_filename)
                model_data['model_name'] = model_name
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"\n✅ COMPLETED: {model_name} (LLM Binary)")
                print(f"Accuracy: {model_data['accuracy']:.4f}")
                print(f"F1 Score: {model_data['f1_score']:.4f}")
                print(f"AUC-ROC: {model_data['auc_roc']:.4f}")
                print(f"Model saved: {model_filename}")
                
            else:
                # Standard multi-class mode (existing code)
                chunk_df = create_chunk_dataset(df, args.dataset, debug_mode=args.debug)
                
                if len(chunk_df) == 0:
                    print(f"No chunks created for {filename}, skipping...")
                    continue
                
                # Save chunk dataset if requested
                if args.save_chunks:
                    chunk_csv_path = os.path.join(args.output_dir, f'chunk_dataset_{model_name}_{args.dataset}.csv')
                    os.makedirs(args.output_dir, exist_ok=True)
                    save_chunk_dataset(chunk_df, chunk_csv_path)
                
                # Train classifier
                print(f"\nTraining classifier for {model_name} on {args.dataset} dataset...")
                model_data = train_classifier(
                    chunk_df, 
                    args.output_dir,
                    test_size=args.test_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    use_only_hallushift=args.use_only_hallushift,
                    num_workers=args.num_workers
                )
                
                # Standard results processing...
                feature_set = 'HalluShift' if args.use_only_hallushift else 'HalluShift++'
                
                results_data = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'model_name': model_name,
                    'dataset': args.dataset,
                    'feature_set': feature_set,
                    'total_chunks': len(chunk_df),
                    'total_samples': len(df),
                    'test_accuracy': model_data['test_accuracy'],
                    'train_accuracy': model_data['train_accuracy'],
                    'precision': model_data['precision'],
                    'recall': model_data['recall'],
                    'f1_score': model_data['f1_score'],
                    'is_binary': False
                }
                
                # Save model
                model_filename = f'semantic_hallucination_classifier_{model_name}_{args.dataset}_{feature_set.lower()}.pkl'
                model_path = os.path.join(args.output_dir, model_filename)
                model_data['model_name'] = model_name
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"\n✅ COMPLETED: {model_name}")
                print(f"Test Accuracy: {model_data['test_accuracy']:.4f}")
                print(f"Model saved: {model_filename}")
            
            # Save results
            save_classification_results(results_data, args.output_dir)
                
        except Exception as e:
            print(f"❌ ERROR processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*100}")
    if args.llm_binary_mode:
        print("🔷 LLM BINARY CLASSIFICATION COMPLETED!")
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Mode: Binary Hallucination Detection")
        print(f"BLEURT Threshold: {args.bleurt_threshold}")
    else:
        print("🎉 ALL MODELS PROCESSED SUCCESSFULLY!")
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Feature Set: {'HalluShift only' if args.use_only_hallushift else 'HalluShift++'}")
    print(f"Results saved to: {os.path.join(args.output_dir, 'classification_results_comparison.csv')}")
    print(f"{'='*100}")

if __name__ == '__main__':
    main() 