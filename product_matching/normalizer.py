import pandas as pd
import re
from functools import reduce
import operator
from .constants import terms_by_type, terms_by_country, ALL_PATTERNS, PATTERNS_LOW

class DataNormalizer:
    def __init__(self):
        # Create stop words set from business terms
        ts = reduce(operator.iconcat, terms_by_type.values(), [])
        cs = reduce(operator.iconcat, terms_by_country.values(), [])
        self.stop_words = set(ts + cs)

    def remove_stop_words(self, string):
        """Remove stop words from string"""
        if pd.isna(string):
            return pd.NA
        return ' '.join(filter(lambda x: x not in self.stop_words, string.split()))

    def normalize_name(self, df, name_columns=['raw_manuf_v', 'manuf_m'], suffix='_cl'):
        """Normalize manufacturer/brand names by removing special chars and stop words"""
        df = df.copy()

        for name_col in name_columns:
            name_column_clean = name_col + suffix
            df[name_column_clean] = df[name_col].str.lower()
            df[name_column_clean].replace(to_replace=r'[\W]+', value=' ', regex=True, inplace=True)

            # Apply stop words removal twice to catch nested patterns
            for _ in range(2):
                df.loc[df[name_column_clean].notna(), name_column_clean] = df.loc[
                    df[name_column_clean].notna(), name_column_clean].apply(self.remove_stop_words)
                df[name_column_clean].replace(to_replace=r'[\W]+', value=' ', regex=True, inplace=True)

            df.loc[df[name_column_clean] == ' ', name_column_clean] = pd.NA

        return df

    def normalize_upc(self, df, col_upc, new_col_name):
        """Normalize UPC codes by removing leading zeros and spaces"""
        df = df.copy()
        df[new_col_name] = df[col_upc].str.strip()
        df[new_col_name] = df[new_col_name].str.lower()
        df[new_col_name].replace(to_replace='^s{0,20}', value='', regex=True, inplace=True)
        df[new_col_name].replace(to_replace='^0{0,20}', value='', regex=True, inplace=True)
        df[df[new_col_name] == ''] = pd.NA
        return df

    def replace_pattern(self, string, modify_degree='low'):
        """Replace specified patterns in string based on modification degree"""
        if pd.isna(string):
            return pd.NA

        string = str(string).lower()
        patterns = PATTERNS_LOW if modify_degree == 'low' else ALL_PATTERNS

        for key, pattern in patterns.items():
            match = re.search(pattern, string)
            if match:
                string = re.sub(pattern, lambda match_obj: re.sub(key, '', match_obj[0]), string)

        return re.sub(r'[\W]+', '', string)

    def filter_short_numeric_mpn(self, df, name_col='mpn_mod', length=4, length_num=6):
        """Filter out short numeric MPNs"""
        df_length = df.loc[df[name_col].str.len() >= length].copy()

        alpha_numeric = df_length.loc[~df_length[name_col].str.isnumeric()]
        numeric = df_length.loc[df_length[name_col].str.isnumeric()]
        numeric_len = numeric.loc[numeric[name_col].str.len() >= length_num]

        return pd.concat([alpha_numeric, numeric_len])

    def fix_short_upc(self, df, name_col='UPC', length=12, min_length=9, debug=False):
        """Add leading zeros to UPC codes to achieve standard length"""
        df = df.copy()
        flt_non_length_idx = df.loc[(df[name_col].str.len() != length) & df['UPC'].notna()].index

        for n in range(min_length, length):
            n_idx = df.loc[df[name_col].str.len() == n].index
            df.loc[n_idx, name_col] = '0' * (length - n) + df.loc[n_idx, name_col]

        if not df.loc[(df[name_col].str.len() != length) & df['UPC'].notna()].empty:
            print('Alert! Exist uncorrected upc')

        return df.loc[flt_non_length_idx] if debug else df