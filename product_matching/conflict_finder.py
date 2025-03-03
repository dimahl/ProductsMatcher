import pandas as pd
from thefuzz.fuzz import partial_token_sort_ratio as ptsr
from itertools import product
import logging
import pickle
from pathlib import Path
from .constants import UPC_V_MOD, UPC_M_MOD, MPN_V_MOD, MPN_M_MOD
from .utils import Timer, safe_concat, safe_merge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConflictFinder:
    def __init__(self, debug_mode=False, debug_dir='debug_data'):
        self.timer = Timer()
        self.debug_mode = debug_mode
        self.debug_dir = Path(debug_dir)

        if debug_mode:
            logger.setLevel(logging.DEBUG)
            self.debug_dir.mkdir(exist_ok=True)
            logger.debug(f"ConflictFinder debug mode enabled. Saving data to {self.debug_dir}")

    def _save_debug_data(self, data, name):
        """Save intermediate data for debugging"""
        if self.debug_mode:
            file_path = self.debug_dir / f"cf_{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {name} to {file_path}")

            # Log first few rows for quick inspection
            if isinstance(data, pd.DataFrame):
                logger.debug(f"\n{name} preview:\n{data.head()}")

    def calc_ratio_prices(self, df, couple_price_col=['price_v', 'ret_price_m'], new_col_ratio='ratio_rp'):
        """Calculate price ratio between two columns"""
        logger.info(f'Calculating ratio prices {new_col_ratio}')

        if self.debug_mode:
            logger.debug(f"Input shape: {df.shape}")
            logger.debug(f"Price columns: {couple_price_col}")
            self._save_debug_data(df, f'before_ratio_{new_col_ratio}')

        df.replace({couple_price_col[0]:{0: pd.NA}, couple_price_col[1]:{0: pd.NA}}, inplace=True)
        if df.loc[df[couple_price_col].notna().all(axis=1)].empty:
            df.loc[:, new_col_ratio] = pd.NA
            logger.warning(f'Failed calculating ratio prices {new_col_ratio}, all rows contain NaN or zero')

        df[new_col_ratio] = (df[couple_price_col[0]] - df[couple_price_col[1]]) / df[couple_price_col[1]]

        if self.debug_mode:
            logger.debug(f"Ratio statistics:\n{df[new_col_ratio].describe()}")
            self._save_debug_data(df, f'after_calc_ratio_{new_col_ratio}')

        return df

    def find_conflicts(self, slave, master, dict_DKH_mfg, dict_DKH_brand, dict_uom_DKH, ratio_price=0.25, ratio_price_manuf=0.3):
        """Find conflicts between slave and master data"""
        self.timer.reset()
        logger.info('Starting find_conflicts')

        if self.debug_mode:
            logger.debug(f"Slave shape: {slave.shape}, Master shape: {master.shape}")
            self._save_debug_data({'slave': slave, 'master': master}, 'input_data')

        # Match by UPC and MPN
        matching_upc = safe_merge(
            slave.dropna(subset=[UPC_V_MOD]), 
            master, 
            how='inner', 
            left_on=UPC_V_MOD, 
            right_on=UPC_M_MOD
        )

        matching_mpn = safe_merge(
            slave.dropna(subset=[MPN_V_MOD]), 
            master, 
            how='inner', 
            left_on=MPN_V_MOD, 
            right_on=MPN_M_MOD
        )

        if self.debug_mode:
            logger.debug(f"UPC matches: {len(matching_upc)}, MPN matches: {len(matching_mpn)}")
            self._save_debug_data({'upc': matching_upc, 'mpn': matching_mpn}, 'initial_matches')

        if not matching_upc.empty:
            matching_upc['key'] = 'upc'
        if not matching_mpn.empty:
            matching_mpn['key'] = 'mpn'

        matching = safe_concat([matching_mpn, matching_upc], ignore_index=True)
        if matching.empty:
            logger.info('No matches found')
            return matching

        # Calculate price ratios
        for ratio_cols, ratio_name in [
            (['price_v', 'ret_price_m'], 'ratio_rp'),
            (['price_v', 'price_m'], 'ratio_dkp'),
            (['cost_v', 'cost_m'], 'ratio_cp')
        ]:
            matching = self.calc_ratio_prices(matching, ratio_cols, ratio_name)

        if self.debug_mode:
            self._save_debug_data(matching, 'after_price_ratios')

        # Replace UOM codes
        matching.replace({'uom_m': dict_uom_DKH.to_dict()['unit_measure']}, inplace=True)
        matching.replace({'uom_v': dict_uom_DKH.set_index('uom').to_dict()['unit_measure']}, inplace=True)

        matching['type'] = pd.NA

        # Process matches
        self._process_manufacturer_matches(matching, dict_DKH_mfg, dict_DKH_brand, ratio_price_manuf)
        self._process_remaining_matches(matching, ratio_price)

        if self.debug_mode:
            self._save_debug_data(matching, 'final_conflicts')
            logger.debug(f"Final results shape: {matching.shape}")
            logger.debug(f"Type distribution:\n{matching['type'].value_counts()}")

        logger.info(f'Finished find_conflicts after {self.timer.delta()} seconds')
        return matching

    def _process_manufacturer_matches(self, matching, dict_DKH_mfg, dict_DKH_brand, ratio_price_manuf):
        """Process matches based on manufacturer data"""
        logger.info('Processing manufacturer matches')

        # Normalize names
        for df in [dict_DKH_brand, dict_DKH_mfg]:
            df['_cl'] = df.apply(lambda x: ' '.join(x.str.lower().str.split()), axis=1)

        # Merge manufacturer and brand data
        for prefix, df in [('manuf', dict_DKH_mfg), ('brand', dict_DKH_brand)]:
            for suffix in ['v', 'm']:
                col = f'{prefix}_id_{suffix}'
                if col in matching.columns:
                    matching = safe_merge(
                        matching,
                        df[[f'{prefix}_ID', f'{prefix}_name_dkh', f'{prefix}_name_dkh_cl']].drop_duplicates(),
                        how='left',
                        left_on=col,
                        right_on=f'{prefix}_ID'
                    )
                    matching.rename(columns={
                        f'{prefix}_name_dkh': f'{prefix}_{suffix}',
                        f'{prefix}_name_dkh_cl': f'{prefix}_{suffix}_cl'
                    }, inplace=True)

        # Mark exact matches
        names_v = [i for i in matching.columns if 'v_cl' in i]
        names_m = [i for i in matching.columns if 'm_cl' in i]

        for name_v in names_v:
            try:
                matching.loc[matching[names_m].eq(matching[name_v], axis=0).any(axis=1), 'type'] = 1
            except ValueError as e:
                logger.warning(f'Error comparing names: {e}')
                continue

        # Process price ratios for type 1 matches
        ratio_cols = ['ratio_rp', 'ratio_dkp', 'ratio_cp']
        matching.loc[
            (matching['type'] == 1) & 
            (matching[ratio_cols] <= ratio_price_manuf).any(axis=1),
            'type'
        ] = 11

        matching.loc[
            (matching['type'] == 1) & 
            (matching['pack_size_v'] == matching['pack_size_m']) & 
            (matching['pack_size_m'] != 1),
            'type'
        ] = 11

        matching.loc[
            (matching['type'] == 1) & 
            (matching[ratio_cols] > ratio_price_manuf).any(axis=1),
            'type'
        ] = 10

    def _process_remaining_matches(self, matching, ratio_price):
        """Process remaining matches based on various criteria"""
        logger.info('Processing remaining matches')

        # Process exact matches
        for key in ['mpn', 'upc']:
            mask = (
                (matching['type'].isna()) & 
                (matching['key'] == key) & 
                (matching[f'{key}_v_mod'] == matching[f'{key}_m_mod'])
            )
            matching.loc[mask, 'type'] = 12

        # Process long MPN matches
        matching_long_mpn = self._filter_short_numeric_mpn(
            matching[matching['type'].isna() & (matching['key'] == 'mpn')],
            name_col=MPN_V_MOD
        )

        # Set type 2 for matches with similar prices
        ratio_cols = ['ratio_rp', 'ratio_dkp', 'ratio_cp']
        if not matching_long_mpn.empty:
            matching.loc[
                matching_long_mpn.index[
                    matching_long_mpn[ratio_cols].le(ratio_price).any(axis=1)
                ],
                'type'
            ] = 2

        # Set type 2 for exact UPC matches
        matching.loc[
            (matching[UPC_V_MOD] == matching[UPC_M_MOD]) & 
            matching['type'].isna(),
            'type'
        ] = 2

        # Process UPC matches with manufacturer differences
        matching.loc[
            (matching['type'].isna()) & 
            (matching[UPC_V_MOD] == matching[UPC_M_MOD]) & 
            (matching[ratio_cols] <= ratio_price).any(axis=1),
            'type'
        ] = 21

    def _filter_short_numeric_mpn(self, df, name_col=MPN_V_MOD, length=4, length_num=6):
        """Filter out short numeric MPNs"""
        if df.empty:
            return df

        df = df[df[name_col].str.len() >= length].copy()

        alpha_numeric = df[~df[name_col].str.isnumeric()]
        numeric = df[df[name_col].str.isnumeric()]
        numeric = numeric[numeric[name_col].str.len() >= length_num]

        return safe_concat([alpha_numeric, numeric])