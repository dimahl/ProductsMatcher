import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from thefuzz.fuzz import ratio, partial_ratio, token_sort_ratio
import logging
from pathlib import Path
import pickle
from scipy import sparse

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self, debug_mode=False, debug_dir='debug_data'):
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'[a-zA-Z0-9]+',
            stop_words='english',
            min_df=1  # Изменено с 2 на 1 для работы с малыми наборами данных
        )
        self.debug_mode = debug_mode
        self.debug_dir = Path(debug_dir)

        if debug_mode:
            self.debug_dir.mkdir(exist_ok=True)
            logger.debug(f"TextAnalyzer debug mode enabled. Saving data to {self.debug_dir}")

    def _save_debug_data(self, data, name):
        """Save intermediate data for debugging"""
        if self.debug_mode:
            file_path = self.debug_dir / f"text_{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {name} to {file_path}")

            # Log data preview for quick inspection
            if isinstance(data, dict):
                for k, v in data.items():
                    logger.debug(f"{name} - {k}: {type(v)}")
                    if isinstance(v, (pd.Series, pd.DataFrame)):
                        logger.debug(f"\n{v.head()}")
                    elif isinstance(v, (sparse.csr_matrix, sparse.csc_matrix)):
                        logger.debug(f"Shape: {v.shape}, Non-zero elements: {v.nnz}")

    def calculate_title_similarity(self, df, title_v='title_v', title_m='title_m'):
        """Calculate similarity between product titles using multiple metrics"""
        logger.info("Calculating title similarities")

        df = df.copy()

        # Basic string metrics
        df['title_len_ratio'] = df.apply(
            lambda x: min(len(str(x[title_v])), len(str(x[title_m]))) / 
                     max(len(str(x[title_v])), len(str(x[title_m]))),
            axis=1
        )

        # Fuzzy string matching
        df['title_ratio'] = df.apply(
            lambda x: ratio(str(x[title_v]), str(x[title_m])) / 100,
            axis=1
        )

        df['title_partial_ratio'] = df.apply(
            lambda x: partial_ratio(str(x[title_v]), str(x[title_m])) / 100,
            axis=1
        )

        df['title_token_sort_ratio'] = df.apply(
            lambda x: token_sort_ratio(str(x[title_v]), str(x[title_m])) / 100,
            axis=1
        )

        if self.debug_mode:
            self._save_debug_data({
                'length_ratios': df['title_len_ratio'].describe(),
                'similarity_ratios': df[['title_ratio', 'title_partial_ratio', 'title_token_sort_ratio']].describe()
            }, 'title_metrics')

        return df

    def calculate_tfidf_similarity(self, df, text_columns=['title_v', 'title_m']):
        """Calculate TF-IDF vectors and cosine similarity for text columns"""
        logger.info("Calculating TF-IDF similarities")

        try:
            # Prepare text data
            titles_v = df[text_columns[0]].fillna('').astype(str)
            titles_m = df[text_columns[1]].fillna('').astype(str)

            # Calculate TF-IDF for both columns
            tfidf_v = self.vectorizer.fit_transform(titles_v)
            vocabulary = self.vectorizer.vocabulary_  # Save vocabulary from first fit

            # Use same vocabulary for second column
            vectorizer_m = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'[a-zA-Z0-9]+',
                stop_words='english',
                vocabulary=vocabulary  # Use same vocabulary
            )
            tfidf_m = vectorizer_m.fit_transform(titles_m)

            if self.debug_mode:
                self._save_debug_data({
                    'vocabulary': vocabulary,
                    'feature_names': self.vectorizer.get_feature_names_out(),
                    'tfidf_shape': tfidf_v.shape,
                    'sample_vectors': {
                        'first_title_v': pd.DataFrame(
                            tfidf_v.toarray()[0],
                            index=self.vectorizer.get_feature_names_out()
                        )
                    }
                }, 'tfidf_metadata')

            # Calculate cosine similarity between corresponding rows
            numerator = np.array((tfidf_v.multiply(tfidf_m)).sum(axis=1)).flatten()
            v_norm = np.sqrt(np.array(tfidf_v.power(2).sum(axis=1)).flatten())
            m_norm = np.sqrt(np.array(tfidf_m.power(2).sum(axis=1)).flatten())
            denominator = v_norm * m_norm

            # Handle zero denominators
            mask = denominator != 0
            similarities = np.zeros_like(denominator)
            similarities[mask] = numerator[mask] / denominator[mask]

            df['tfidf_similarity'] = similarities

            if self.debug_mode:
                self._save_debug_data({
                    'similarity_stats': pd.Series(similarities).describe(),
                    'vectors': {
                        'v_norm': v_norm,
                        'm_norm': m_norm,
                        'numerator': numerator,
                        'denominator': denominator
                    }
                }, 'tfidf_results')

        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            df['tfidf_similarity'] = np.nan

        return df

    def analyze_matches(self, df):
        """Run all text analysis on matching results"""
        logger.info("Starting text analysis of matches")

        df = self.calculate_title_similarity(df)
        df = self.calculate_tfidf_similarity(df)

        # Calculate composite similarity score
        similarity_cols = ['title_ratio', 'title_partial_ratio', 
                         'title_token_sort_ratio', 'tfidf_similarity']
        df['text_similarity_score'] = df[similarity_cols].mean(axis=1)

        logger.info("Text analysis complete")
        return df