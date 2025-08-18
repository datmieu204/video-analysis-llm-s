# backend/app/utils/nlp_utils.py

import re
import os
import torch
import asyncio
import numpy as np
import time
import logging

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    pipeline
)
from keybert import KeyBERT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Ultra-optimized model manager with aggressive caching and batching"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    async def initialize(self):
        """Async initialization to prevent blocking"""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            start_time = time.time()
            logger.info("Initializing ModelManager...")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device: {self.device}")
            
            self.SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
            self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            
            await self._load_models_async()
            
            self.cpu_executor = ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 1) * 2),
                thread_name_prefix="nlp-cpu"
            )
            self.io_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="nlp-io"
            )
   
            self._initialized = True
    
    async def _load_models_async(self):
        """Load models in parallel"""
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(None, self._load_sentiment_model),
            loop.run_in_executor(None, self._load_embedding_model),
            loop.run_in_executor(None, self._load_keyword_model)
        ]
        
        await asyncio.gather(*tasks)
    
    def _load_sentiment_model(self):
        """Load sentiment analysis model"""
        logger.info("Loading sentiment model...")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.SENTIMENT_MODEL)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.SENTIMENT_MODEL,
            low_cpu_mem_usage=False,
            device_map=None
        )

        if self.device == "cuda":
            self.sentiment_model = self.sentiment_model.half().to(self.device)
        else:
            self.sentiment_model = self.sentiment_model.to(self.device)
            
        self.sentiment_model.eval()
    
    def _load_embedding_model(self):
        """Load embedding model"""
        logger.info("Loading embedding model...")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL)
        self.embedding_model = AutoModel.from_pretrained(
            self.EMBEDDING_MODEL,
            low_cpu_mem_usage=False,
            device_map=None
        )

        if self.device == "cuda":
            self.embedding_model = self.embedding_model.half().to(self.device)
        else:
            self.embedding_model = self.embedding_model.to(self.device)
            
        self.embedding_model.eval()
    
    def _load_keyword_model(self):
        """Load keyword extraction model"""
        logger.info("Loading keyword model...")
        self.keyword_extractor = KeyBERT(model=self.EMBEDDING_MODEL)

# Global instance
model_manager = ModelManager()


@lru_cache(maxsize=2000)
def _cached_tokenize_sentiment(text: str) -> tuple:
    """Cache sentiment tokenization"""
    return tuple(model_manager.sentiment_tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True))

@lru_cache(maxsize=2000)
def _cached_tokenize_embedding(text: str) -> tuple:
    """Cache embedding tokenization"""
    return tuple(model_manager.embedding_tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True))

@lru_cache(maxsize=1000)
def _cached_keywords(text: str, top_n: int = 5) -> tuple:
    """Cache keyword extraction"""
    try:
        filtered_text = re.sub(r'\d{2}:\d{2}:\d{2}\s*', '', text)
        if len(filtered_text.strip()) < 10:  # Skip very short texts
            return tuple()
        keywords = model_manager.keyword_extractor.extract_keywords(
            filtered_text, 
            keyphrase_ngram_range=(1, 2), 
            top_n=top_n,
            stop_words='english'
        )
        return tuple(kw[0] for kw in keywords)
    except:
        return tuple()


def analyze_sentiment_ultra_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Ultra-fast batch sentiment analysis with optimized inference"""
    if not texts:
        return []
    
    try:
        batch_size = 32 if model_manager.device == "cuda" else 16
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = model_manager.sentiment_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(model_manager.device)
            
            with torch.no_grad():
                outputs = model_manager.sentiment_model(**encoded)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                for j, pred in enumerate(predictions):
                    pred_cpu = pred.cpu().numpy()
                    label_idx = np.argmax(pred_cpu)
                    score = float(pred_cpu[label_idx])
                    
                    labels = ['negative', 'neutral', 'positive']
                    label = labels[label_idx] if label_idx < len(labels) else 'neutral'
                    
                    results.append({
                        'label': label,
                        'score': score
                    })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        return [{'label': 'error', 'score': 0.0} for _ in texts]

def get_embeddings_ultra_batch(texts: List[str]) -> np.ndarray:
    """Ultra-fast batch embedding generation"""
    if not texts:
        return np.array([])
    
    try:
        batch_size = 64 if model_manager.device == "cuda" else 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = model_manager.embedding_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(model_manager.device)
            
            with torch.no_grad():
                outputs = model_manager.embedding_model(**encoded)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        return np.random.random((len(texts), 384))

def split_transcript_(transcript: str) -> List[Dict[str, str]]:
    """Optimized transcript splitting"""
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2})\s+(.*?)(?=\n\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL)
    
    sentences = []
    for match in pattern.finditer(transcript):
        timestamp, sentence = match.groups()
        clean_sentence = ' '.join(sentence.split())
        if len(clean_sentence) > 5:
            sentences.append({
                'timestamp': timestamp,
                'sentence': clean_sentence
            })
    
    return sentences

def group_sentences_(sentences: List[Dict[str, str]], max_tokens: int = 400) -> List[List[Dict[str, str]]]:
    """Optimized sentence grouping with smaller token limit for speed"""
    groups = []
    current_group = []
    current_length = 0
    
    for item in sentences:
        # Quick length estimation (characters / 4 ≈ tokens)
        estimated_tokens = len(item['sentence']) // 4
        
        if current_length + estimated_tokens > max_tokens and current_group:
            groups.append(current_group)
            current_group = [item]
            current_length = estimated_tokens
        else:
            current_group.append(item)
            current_length += estimated_tokens
    
    if current_group:
        groups.append(current_group)
    
    return groups

async def score_selection_ultra_fast(text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Ultra-optimized scoring with parallel processing"""
    start_time = time.time()
    
    await model_manager.initialize()
    
    sentences = split_transcript_(text)
    if not sentences:
        return []
    
    groups = group_sentences_(sentences, max_tokens=300)  # Smaller groups = faster
    if not groups:
        return []
    
    logger.info(f"Processing {len(groups)} groups from {len(sentences)} sentences")
    
    group_texts = [' '.join(item['sentence'] for item in group) for group in groups]
    
    overall_keywords_task = asyncio.get_event_loop().run_in_executor(
        model_manager.io_executor,
        lambda: list(_cached_keywords(text[:2000], 15))  # Use only first part for speed
    )
    
    sentiment_task = asyncio.get_event_loop().run_in_executor(
        model_manager.cpu_executor,
        analyze_sentiment_ultra_batch,
        group_texts
    )
    
    embedding_task = asyncio.get_event_loop().run_in_executor(
        model_manager.cpu_executor,
        get_embeddings_ultra_batch,
        group_texts
    )
    
    keyword_tasks = [
        asyncio.get_event_loop().run_in_executor(
            model_manager.io_executor,
            lambda t=text: list(_cached_keywords(t, 3))
        ) for text in group_texts
    ]
    
    try:
        overall_keywords, sentiments, embeddings = await asyncio.gather(
            overall_keywords_task,
            sentiment_task,
            embedding_task
        )
        all_keywords = await asyncio.gather(*keyword_tasks)
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        return []
    
    scored_groups = []
    prev_sentiment_score = 0
    
    for i, (group, group_text, sentiment, embedding, keywords) in enumerate(
        zip(groups, group_texts, sentiments, embeddings, all_keywords)
    ):
        sentiment_score = sentiment.get('score', 0.0)
        if sentiment.get('label') == 'negative':
            sentiment_score *= -1
        
        intensity_score = abs(sentiment_score)
        shift_score = abs(sentiment_score - prev_sentiment_score)
        prev_sentiment_score = sentiment_score
        
        if i > 0:
            novelty_score = 1 - cosine_similarity([embedding], [embeddings[i-1]])[0][0]
        else:
            novelty_score = 0.5
        
        keyword_score = len(set(keywords) & set(overall_keywords)) / max(len(keywords), 1)
        
        highlight_score = (
            0.4 * intensity_score +
            0.3 * shift_score +
            0.2 * novelty_score +
            0.1 * keyword_score
        )
        
        scored_groups.append({
            'group': group,
            'group_text': group_text,
            'keywords': keywords,
            'sentiment_score': sentiment_score,
            'highlight_score': highlight_score,
            'reason': f"I:{intensity_score:.2f} S:{shift_score:.2f} N:{novelty_score:.2f}"
        })
    
    result = sorted(scored_groups, key=lambda x: x['highlight_score'], reverse=True)[:top_n]
    
    return result

def analyze_sentiment(text: str, **kwargs) -> Dict[str, Any]:
    """Legacy compatibility for single text sentiment analysis"""
    results = analyze_sentiment_ultra_batch([text])
    return results[0] if results else {'label': 'neutral', 'score': 0.0}

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Legacy compatibility for keyword extraction"""
    return list(_cached_keywords(text, top_n))

async def score_selection(text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Legacy compatibility for scoring selection"""
    return await score_selection_ultra_fast(text, top_n)

def prompt_highlights(highlights: List[Dict[str, Any]]) -> str:
    """Generate highlight prompts"""
    if not highlights:
        return "No highlights found."
    
    blocks = []
    for i, item in enumerate(highlights):
        transcript_group = '\n'.join([
            f"[{s['timestamp']}] {s['sentence']}"
            for s in item['group']
        ])
        
        blocks.append(
            f"### Highlight {i+1} (Score: {item.get('highlight_score', 0):.2f})\n"
            f"{transcript_group}\n"
            f"Keywords: {', '.join(item.get('keywords', []))}\n"
            f"Analysis: {item.get('reason', 'N/A')}"
        )
    
    return '\n\n'.join(blocks)

def clear_all_caches():
    """Clear all caches"""
    _cached_tokenize_sentiment.cache_clear()
    _cached_tokenize_embedding.cache_clear()
    _cached_keywords.cache_clear()
    logger.info("All caches cleared")

# Initialize on import
async def init_models():
    """Initialize models on startup"""
    await model_manager.initialize()

# if __name__ == "__main__":
#     asyncio.run(init_models())