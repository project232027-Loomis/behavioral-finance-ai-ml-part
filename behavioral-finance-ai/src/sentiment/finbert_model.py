import pandas as pd
from dataclasses import dataclass
from loguru import logger
import torch
import warnings

@dataclass
class SentimentResult:
    text: str
    label: str
    positive_score: float
    negative_score: float
    neutral_score: float
    confidence: float

class FinBERTSentiment:
    def __init__(self, config: dict):
        self.model_name = config.get('model_name', 'ProsusAI/finbert')
        self.batch_size = config.get('batch_size', 16)
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipeline = None
        self._transformers_available = True
        try:
            from transformers import pipeline
        except ImportError:
            self._transformers_available = False
            logger.warning("transformers library not found. Falling back to neutral default.")

    def _load_model(self):
        if self.pipeline is None and self._transformers_available:
            try:
                from transformers import pipeline
                logger.info(f"Loading FinBERT model: {self.model_name}...")
                self.pipeline = pipeline("sentiment-analysis", model=self.model_name, return_all_scores=True, device=self.device)
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                self._transformers_available = False

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        self._load_model()
        if not self._transformers_available or not texts:
            return [self._get_neutral_default(t) for t in texts]

        results = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                out = self.pipeline(batch)
                
                for text, scores in zip(batch, out):
                    score_dict = {s['label']: s['score'] for s in scores}
                    max_label = max(score_dict, key=score_dict.get)
                    confidence = score_dict[max_label]
                    
                    results.append(SentimentResult(
                        text=text,
                        label=max_label,
                        positive_score=score_dict.get('positive', 0.0),
                        negative_score=score_dict.get('negative', 0.0),
                        neutral_score=score_dict.get('neutral', 0.0),
                        confidence=confidence
                    ))
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            results.extend([self._get_neutral_default(t) for t in texts[len(results):]])
            
        return results

    def analyze_single(self, text: str) -> SentimentResult:
        return self.analyze([text])[0]

    def _get_neutral_default(self, text: str) -> SentimentResult:
        return SentimentResult(text, 'neutral', 0.0, 0.0, 1.0, 1.0)

    def aggregate_sentiment(self, results: list[SentimentResult]) -> dict:
        if not results:
            return {'avg_positive': 0, 'avg_negative': 0, 'avg_neutral': 1, 'overall_sentiment': 'neutral'}
            
        pos = sum(r.positive_score for r in results) / len(results)
        neg = sum(r.negative_score for r in results) / len(results)
        neu = sum(r.neutral_score for r in results) / len(results)
        
        labels = [r.label for r in results]
        majority = max(set(labels), key=labels.count)
        
        return {
            'avg_positive': pos,
            'avg_negative': neg,
            'avg_neutral': neu,
            'overall_sentiment': majority
        }

    def filter_negative(self, stock_headlines: dict[str, list[str]], threshold: float = 0.6) -> dict:
        filtered = {}
        for symbol, texts in stock_headlines.items():
            res = self.analyze(texts)
            agg = self.aggregate_sentiment(res)
            if agg['avg_negative'] < threshold:
                filtered[symbol] = texts
            else:
                logger.info(f"Filtered out {symbol} due to high negative sentiment ({agg['avg_negative']:.2f})")
        return filtered

    def score_stock_universe(self, stock_headlines: dict[str, list[str]]) -> pd.DataFrame:
        data = []
        for symbol, texts in stock_headlines.items():
            res = self.analyze(texts)
            agg = self.aggregate_sentiment(res)
            data.append({
                'Symbol': symbol,
                'avg_positive': agg['avg_positive'],
                'avg_negative': agg['avg_negative'],
                'sentiment_label': agg['overall_sentiment'],
                'recommendation': 'Avoid' if agg['avg_negative'] > 0.5 else 'Consider'
            })
        return pd.DataFrame(data)

if __name__ == '__main__':
    logger.info("Demo FinBERT Sentiment")
    bert = FinBERTSentiment({})
    samples = {
        'AAPL': ['Apple reports record profits and strong iPhone sales', 'New product launch successful'],
        'TSLA': ['Tesla faces bankruptcy rumors and factory shutdowns', 'CEO steps down amid scandal'],
        'MSFT': ['Microsoft acquires new gaming studio', 'Cloud revenue meets expectations']
    }
    df = bert.score_stock_universe(samples)
    print(df)
    
    filtered = bert.filter_negative(samples, threshold=0.5)
    print("\nFiltered (Negative < 0.5):", list(filtered.keys()))
