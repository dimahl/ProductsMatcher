import pandas as pd
from sqlalchemy import create_engine
import logging
import pdb
import pickle
from pathlib import Path
from .normalizer import DataNormalizer
from .conflict_finder import ConflictFinder
from .constants import BASE_COLUMNS, MPN_V_MOD, UPC_V_MOD, UPC_M_MOD
from .utils import Timer, safe_concat, safe_merge
from .text_analyzer import TextAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductMatcher:
    def __init__(self, sql_string=None, debug_mode=False, debug_dir='debug_data'):
        self.normalizer = DataNormalizer()
        self.conflict_finder = ConflictFinder(debug_mode=debug_mode, debug_dir=debug_dir)
        self.text_analyzer = TextAnalyzer(debug_mode=debug_mode, debug_dir=debug_dir)
        self.sql_string = sql_string
        self.timer = Timer()
        self.debug_mode = debug_mode
        self.debug_dir = Path(debug_dir)

        if debug_mode:
            logger.setLevel(logging.DEBUG)
            self.debug_dir.mkdir(exist_ok=True)
            logger.debug(f"Debug mode enabled. Saving data to {self.debug_dir}")

    def _save_debug_data(self, data, name):
        """Save intermediate data for debugging"""
        if self.debug_mode:
            file_path = self.debug_dir / f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {name} to {file_path}")

            # Log first few rows for quick inspection
            if isinstance(data, pd.DataFrame):
                logger.debug(f"\n{name} preview:\n{data.head()}")

    def create_mapping_conflicts(self, slave, master, vendor_id, dict_DKH_mfg, 
                             dict_DKH_brand, dict_uom_DKH, ratio_price=0.2,
                             ratio_price_manuf=0.5, trust2='check_existing',
                             source=None, set_trace=False):
        """Main method to create product mapping conflicts"""
        self.timer.reset()
        logger.info('Starting mapping creation')

        if self.debug_mode:
            self._save_debug_data(slave, 'initial_slave')
            self._save_debug_data(master, 'initial_master')

        if set_trace:
            logger.debug("Setting breakpoint for interactive debugging")
            pdb.set_trace()

        if trust2 not in ['check_existing', 'expand']:
            raise ValueError('Wrong choice parameter trust2, only: check_existing or expand')

        # Step 1: Normalize UPCs
        logger.info('Starting UPC modification')
        slave = self.normalizer.normalize_upc(slave, 'upc_v', UPC_V_MOD)
        master = self.normalizer.normalize_upc(master, 'upc_m', UPC_M_MOD)

        if self.debug_mode:
            self._save_debug_data(slave, 'normalized_upc_slave')
            self._save_debug_data(master, 'normalized_upc_master')

        # Step 2: First round of conflict finding with low modification
        logger.info('Starting MPN low modification')
        slave[MPN_V_MOD] = slave['mpn_v'].apply(self.normalizer.replace_pattern, args=('low',))
        master[MPN_M_MOD] = master['mpn_m'].apply(self.normalizer.replace_pattern, args=('low',))

        if self.debug_mode:
            self._save_debug_data(slave, 'low_mod_mpn_slave')
            self._save_debug_data(master, 'low_mod_mpn_master')

        result_1 = self.conflict_finder.find_conflicts(
            slave, master, dict_DKH_mfg, dict_DKH_brand, dict_uom_DKH,
            ratio_price, ratio_price_manuf
        )

        if not result_1.empty:
            result_1['trust'] = 1
            if self.debug_mode:
                self._save_debug_data(result_1, 'conflicts_round1')

        # Step 3: Second round with high modification
        logger.info('Starting MPN high modification')
        if trust2 == 'check_existing':
            slave = slave[~slave['prod_id_v'].isin(result_1['prod_id_v'])].copy()
            if self.debug_mode:
                self._save_debug_data(slave, 'filtered_slave_round2')

        slave[MPN_V_MOD] = slave['mpn_v'].apply(self.normalizer.replace_pattern, args=('high',))
        master[MPN_M_MOD] = master['mpn_m'].apply(self.normalizer.replace_pattern, args=('high',))

        if self.debug_mode:
            self._save_debug_data(slave, 'high_mod_mpn_slave')
            self._save_debug_data(master, 'high_mod_mpn_master')

        result_2 = self.conflict_finder.find_conflicts(
            slave, master, dict_DKH_mfg, dict_DKH_brand, dict_uom_DKH,
            ratio_price, ratio_price_manuf
        )

        if not result_2.empty:
            result_2['trust'] = 2
            if self.debug_mode:
                self._save_debug_data(result_2, 'conflicts_round2')

        if result_1.empty and result_2.empty:
            logger.info('No matching products found')
            return pd.DataFrame()

        # Combine and process results
        cols = BASE_COLUMNS if source != 'DK' else [c for c in BASE_COLUMNS if c not in ['item_code_v', 'raw_manuf_id_v']]
        result_conf = safe_concat([result_1, result_2], ignore_index=True)
        if not result_conf.empty:
            result_conf = result_conf.loc[:, cols]
            if self.debug_mode:
                self._save_debug_data(result_conf, 'combined_results')

            # Add text analysis results
            logger.info("Adding text similarity analysis")
            result_conf = self.text_analyzer.analyze_matches(result_conf)

            if self.debug_mode:
                self._save_debug_data(result_conf, 'text_analysis_results')

        # Prioritize matches by type
        type_priority = {
            11: 1, 12: 2, 10: 3, 1: 4, 2: 5, 21: 6
        }
        result_conf['temp'] = result_conf['type'].map(type_priority)
        result_conf.sort_values('temp', inplace=True)
        result_conf.drop_duplicates(subset=['prod_id_v', 'prod_id_m'], keep='first', inplace=True)
        result_conf.drop(columns='temp', inplace=True)

        result_conf.insert(2, 'master', pd.NA)
        result_conf['prod_id_m'] = result_conf['prod_id_m'].astype('int')

        if self.debug_mode:
            self._save_debug_data(result_conf, 'final_results')

        logger.info(f'Finished mapping creation after {self.timer.delta()} seconds')
        return result_conf

    def handle_equal_manuf(self, df, what_do):
        """Handle manufacturer equivalence data in database"""
        if not self.sql_string:
            raise ValueError("SQL connection string not provided")

        self.timer.reset()
        logger.info(f'Starting handle_equal_manuf with action: {what_do}')

        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={self.sql_string}")
        table = 'equal_manuf_matching_products'

        try:
            if what_do == 'read':
                df_equal_manuf = pd.read_sql(f"SELECT * FROM {table}", con=engine)
                return df_equal_manuf.drop_duplicates()

            elif what_do == 'add':
                if df is not None:
                    df.drop_duplicates().to_sql(table, con=engine, schema='dbo', 
                                                if_exists='append', index=False)

            elif what_do == 'creat_new':
                if df is not None:
                    df.drop_duplicates().to_sql(table, con=engine, schema='dbo',
                                                if_exists='replace', index=False)
            else:
                raise ValueError('Wrong choice what_do')

        finally:
            engine.dispose()

        logger.info(f'Finished handle_equal_manuf after {self.timer.delta()} seconds')

    def reconflict(self, df, equal_manuf_from_prod, equal_prod, not_same_prod,
                 raw_equal_manuf_from_prod, raw_manuf_from_equivalents_prod,
                 ratio_price_manuf=0.5):
        """Process and update conflicts based on equivalence data"""
        self.timer.reset()
        logger.info('Starting reconflict processing')

        df = df.copy()

        try:
            df_equal_manuf = self.handle_equal_manuf(None, 'read')
        except Exception as e:
            logger.warning(f'Failed to read equal manufacturers: {e}')
            df_equal_manuf = pd.DataFrame(columns=['manuf_id_v', 'manuf_id_m', 
                                                    'manuf_v', 'manuf_m'])

        # Process manufacturer equivalences
        for products in equal_manuf_from_prod:
            manufs = df.loc[(df['prod_id_v'] == products[0]) & 
                             (df['prod_id_m'] == products[1]),
                             ['manuf_id_v', 'manuf_id_m', 'manuf_v', 'manuf_m']]
            if not manufs.empty:
                df_equal_manuf = safe_concat([df_equal_manuf, manufs])

        if not df_equal_manuf.empty:
            df_equal_manuf.drop_duplicates(inplace=True)
            self.handle_equal_manuf(df_equal_manuf, 'creat_new')

        # Process conflicts based on equivalences
        TEMP_LABEL = 777
        for manufs in df_equal_manuf.itertuples():
            for cond in [
                (df['manuf_id_v'] == manufs.manuf_id_v) & 
                (df['manuf_id_m'] == manufs.manuf_id_m),
                (df['manuf_id_v'] == manufs.manuf_id_m) & 
                (df['manuf_id_m'] == manufs.manuf_id_v)
            ]:
                df.loc[cond, 'type'] = TEMP_LABEL

        # Update types based on price ratios
        ratio_cols = ['ratio_rp', 'ratio_dkp', 'ratio_cp']
        df.loc[(df['type'] == TEMP_LABEL) & 
              (df[ratio_cols] <= ratio_price_manuf).any(axis=1),
              'type'] = 11
        df.loc[(df['type'] == TEMP_LABEL) & 
              (df[ratio_cols] > ratio_price_manuf).any(axis=1),
              'type'] = 10
        df.loc[(df['type'] == TEMP_LABEL) & 
              (df['uom_v'] == df['uom_m']),
              'type'] = 11

        # Process equal products
        for products in equal_prod:
            df.loc[(df['prod_id_v'] == products[0]) & 
                 (df['prod_id_m'] == products[1]),
                 'type'] = 3

        # Process not same products
        for products in not_same_prod:
            df.loc[(df['prod_id_v'] == products[0]) & 
                 (df['prod_id_m'] == products[1]),
                 ['type', 'master']] = (0, False)

        # Calculate letter count in MPNs
        df['cnt_abc'] = df[MPN_V_MOD].apply(
            lambda x: len([i for i in str(x) if i.isalpha()]) if pd.notna(x) else 0
        )

        # Remove lower priority conflicts
        df.drop_duplicates(subset=['prod_id_v', 'prod_id_m', 'type'],
                            keep='first', inplace=True)
        reliable_mask = df['type'] == 11
        if reliable_mask.any():
            reliable_ids = df.loc[reliable_mask, 'prod_id_v'].unique()
            df = df[~((df['type'] != 11) & df['type'].notna() & 
                      df['prod_id_v'].isin(reliable_ids))]

        logger.info(f'Finished reconflict after {self.timer.delta()} seconds')
        return df