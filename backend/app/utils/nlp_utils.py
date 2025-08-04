# backend/app/utils/nlp_utils.py

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from transformers import AutoTokenizer, pipeline
from keybert import KeyBERT

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL)

EXTRACTKEY_MODEL = "sentence-transformers/LaBSE"
keyword_extractor = KeyBERT(model=EXTRACTKEY_MODEL)

tokenizer = AutoTokenizer.from_pretrained(EXTRACTKEY_MODEL)
embedding_pipeline = pipeline("feature-extraction", model=EXTRACTKEY_MODEL, tokenizer=EXTRACTKEY_MODEL)

def analyze_sentiment(text: str , window_size: int = 512, overlap: int = 100) -> Dict[str, Any]:
    """
    Analyze sentiment of a text using a sliding window (token-based).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= window_size:
        try:
            response = sentiment_pipeline(text, truncation=True, max_length=window_size)
            return response[0]
        except Exception as e:
            return {
                "label": "error",
                "score": 0.0,
                "message": str(e)
            }

    parts = []

    for i in range(0, len(tokens), window_size - overlap):
        token_chunk = tokens[i:i + window_size]
        chunk_text = tokenizer.decode(token_chunk)
        parts.append(chunk_text)

    results = []

    for part in parts:
        try:
            response = sentiment_pipeline(part)
            results.append(response[0])
        except Exception as e:
            results.append({
                "label": "error",
                "score": 0.0,
                "message": str(e)
            })

    return results

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract keywords from a text using KeyBERT.
    """
    try:
        filtered_text = re.sub(r'\d{2}:\d{2}:\d{2}\s*', '', text)
        keywords = keyword_extractor.extract_keywords(filtered_text, keyphrase_ngram_range=(1, 2), top_n=top_n)
        return [kw[0] for kw in keywords]
    except Exception as e:
        return ["Error extracting keywords: " + str(e)]

def split_transcript_into_sentences(transcript: str) -> List[Dict[str, str]]:
    """
    Analyze a video transcript and split it into sentences with timestamps.
    Each sentence is represented as a dictionary with 'timestamp' and 'sentence' keys.
    """

    pattern = r'(\d{2}:\d{2}:\d{2}) (.*?)(?=\n\d{2}:\d{2}:\d{2}|$)'
    matches = re.findall(pattern, transcript, re.DOTALL)

    sentences = []
    for timestamp, sentence in matches:
        clean_sentence = sentence.strip().replace('\n', ' ')
        if clean_sentence:
            sentences.append({
                'timestamp': timestamp,
                'sentence': clean_sentence
            })
    return sentences

def group_sentences(sentences: List[Dict[str, str]], max_tokens=512) -> List[List[Dict[str, str]]]:
    """
    Group sentences into chunks that fit within a specified token limit.
    """

    groups = []
    current_group = []
    current_tokens = 0

    for item in sentences:
        token_count = len(tokenizer.encode(item['sentence'], add_special_tokens=False))
        
        if current_tokens + token_count > max_tokens:
            groups.append(current_group)
            current_group = []
            current_tokens = 0

        current_group.append(item)
        current_tokens += token_count

    if current_group:
        groups.append(current_group)

    return groups

def analyze_group_sentiment(group: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze sentiment of a group of sentences.
    """
    text = " ".join([item['sentence'] for item in group])
    result = analyze_sentiment(text)

    if isinstance(result, list):
        non_neutral_results = [r for r in result if r.get('label') != 'neutral']
        if non_neutral_results:
            return sorted(non_neutral_results, key=lambda x: x.get('score', 0), reverse=True)[0]
        return result[0]
    return result

def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding of a text using the specified model.
    """
    try:
        embeddings = embedding_pipeline(text, truncation=True, max_length=512)
        return np.array(embeddings[0][0])
    except Exception as e:
        raise ValueError(f"Failed to get embedding: {str(e)}")

def score_selection(text: str, top_n: int=5) -> List[Dict[str, Any]]:
    """
    Analyzes transcript to find candidate highlights using a composite scoring system.
    """
    sentences = split_transcript_into_sentences(text)
    if not sentences:
        return []
    
    groups = group_sentences(sentences)
    if not groups:
        return []

    group_features = []
    overall_keywords = [kw[0] for kw in keyword_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=20)]

    for i, group in enumerate(groups):
        group_text = " ".join(item['sentence'] for item in group)
        
        sentiment = analyze_group_sentiment(group)
        sentiment_score = sentiment.get('score', 0.0)
        if sentiment.get('label') == 'negative':
            sentiment_score *= -1

        embedding = get_embedding(group_text)
        
        group_keywords = extract_keywords(group_text, top_n=5)
        keyword_score = sum(1 for kw in group_keywords if kw in overall_keywords)
        
        group_features.append({
            "group": group,
            "group_text": group_text,
            "keywords": group_keywords,
            "sentiment_score": sentiment_score,
            "embedding": embedding,
            "keyword_score": keyword_score
        })

    scored_groups = []
    for i, features in enumerate(group_features):
        intensity_score = abs(features['sentiment_score'])
        
        prev_sentiment_score = group_features[i-1]['sentiment_score'] if i > 0 else 0
        shift_score = abs(features['sentiment_score'] - prev_sentiment_score)
        
        prev_embedding = group_features[i-1]['embedding'] if i > 0 else np.zeros_like(features['embedding'])
        novelty_score = 1 - cosine_similarity([features['embedding']], [prev_embedding])[0][0]
        
        w_intensity, w_shift, w_novelty, w_keyword = 0.3, 0.4, 0.2, 0.1
        highlight_score = (
            w_intensity * intensity_score +
            w_shift * shift_score +
            w_novelty * novelty_score +
            w_keyword * features['keyword_score']
        )
        
        scored_groups.append({
            **features,
            "highlight_score": highlight_score,
            "reason": f"Intensity: {intensity_score:.2f}, Shift: {shift_score:.2f}, Novelty: {novelty_score:.2f}"
        })

    sorted_groups = sorted(scored_groups, key=lambda x: x['highlight_score'], reverse=True)
    
    return sorted_groups[:top_n]

def prompt_highlights(highlights: List[Dict[str, Any]]) -> str:
    """
    Write prompt highlights from a list of scored groups.
    """
    transcript_block = []

    for i, group_item in enumerate(highlights):
        transcript_group = "\n".join([f"[{item['timestamp']}] {item['sentence']}" for item in group_item['group']])
        keywords_group = group_item['keywords']
        reason = group_item.get('reason', 'N/A')

        transcript_block.append(
            f"""
### Candidate Group {i+1} (Score: {group_item.get('highlight_score', 0):.2f})
{transcript_group}
Keywords: {', '.join(keywords_group)}
Reason for consideration: {reason}
            """.strip()
        )

    return "\n\n".join(transcript_block)

