"""
Process Dashboard Application
===========================
This application creates a Dash web dashboard for analyzing CD (Critical Dimension) 
measurement data from semiconductor manufacturing processes.

Features:
- Interactive charts and tables
- Chamber and tool analysis
- Date range filtering
- Statistical analysis with control limits
"""

# Standard library imports
import sys
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Third-party imports
print("Loading required libraries...")
try:
    import pandas as pd
    import numpy as np
    print("✓ Core data libraries loaded")
except ImportError as e:
    print(f"✗ Error importing core libraries: {e}")
    sys.exit(1)

try:
    import statsmodels.stats.multicomp as multi
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.interpolate import interp1d
    print("✓ Statistical libraries loaded")
except ImportError as e:
    print(f"⚠ Warning: Some statistical libraries not available: {e}")

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Dash, dcc, html, State, callback
    from dash import dash_table as dt
    from dash.dependencies import Input, Output
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    print("✓ Dash and plotting libraries loaded")
except ImportError as e:
    print(f"✗ Error importing Dash libraries: {e}")
    sys.exit(1)

# Try to import PyUber (database connection)
try:
    import PyUber
    PYUBER_AVAILABLE = True
    print("✓ PyUber database library loaded")
except ImportError:
    PYUBER_AVAILABLE = False
    print("⚠ Warning: PyUber not available, will try to use existing CSV files")

print("✓ All available libraries loaded successfully")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# DATABASE CONFIGURATION AND SQL QUERIES
# =============================================================================

print("Setting up database queries...")

# Note: Not using rework flag since we want post rework data if lot was reworked
# Also sorting CD df by descending to get last CD (after rework if there was one)

# SQL to get the CD data
SQL_CD = '''
SELECT 
        a1.entity AS entity
        ,To_Char(a1.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS entity_data_collect_date
        ,a0.operation AS spc_operation
        ,a0.lot AS lot
        ,(SELECT lrc99.last_pass FROM F_LOT_RUN_CARD lrc99 where lrc99.lot =a0.lot AND lrc99.operation = a0.operation AND lrc99.site_prevout_date=a0.prev_moveout_time and rownum<=1) AS last_pass
        ,a4.wafer3 AS raw_wafer3
        ,a4.parameter_name AS raw_parameter_name
        ,a4.value AS raw_value
        ,a3.measurement_set_name AS measurement_set_name
        ,a3.valid_flag as valid_flag
        ,a3.standard_flag as standard_flag
        ,a4.native_x_col AS native_x_col
        ,a4.native_y_row AS native_y_row
        ,a0.route AS route
        ,a0.product AS product
        ,a0.process_operation AS process_operation
FROM 
P_SPC_LOT a0
LEFT JOIN P_SPC_ENTITY a1 ON a1.spcs_id = a0.spcs_id AND a1.entity_sequence=1
INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a0.spcs_id AND a2.data_collection_time = a0.data_collection_time
INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
LEFT JOIN P_SPC_MEASUREMENT a4 ON a4.spcs_id = a3.spcs_id AND a4.measurement_set_name = a3.measurement_set_name
WHERE
a1.data_collection_time >= TRUNC(SYSDATE) -  7
AND      (a3.measurement_set_name = 'CD.DCCD_MEASUREMENTS.5051')
AND      (a4.parameter_name LIKE '%CA%')
AND      (a0.lot LIKE  '%')
AND      a3.standard_flag = 'Y'
AND      a3.valid_flag = 'V'
'''
# Include valid flags to make sure only good data is used

# SQL to get the limits for the CD charts
SQL_LIMITS = '''
SELECT 
          a4.parameter_name AS raw_parameter_name
         ,a10.lo_control_lmt AS lo_control_lmt
         ,a10.up_control_lmt AS up_control_lmt
         ,a0.route AS route
         ,a0.product AS product
         ,a0.operation AS spc_operation
         ,a0.process_operation AS process_operation
FROM 
P_SPC_LOT a0
LEFT JOIN P_SPC_ENTITY a1 ON a1.spcs_id = a0.spcs_id AND a1.entity_sequence=1
INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a0.spcs_id AND a2.data_collection_time = a0.data_collection_time
INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
INNER JOIN P_SPC_CHART_POINT a5 ON a5.spcs_id = a3.spcs_id AND a5.measurement_set_name = a3.measurement_set_name
LEFT JOIN P_SPC_CHARTPOINT_MEASUREMENT a7 ON a7.spcs_id = a3.spcs_id and a7.measurement_set_name = a3.measurement_set_name
AND a5.spcs_id = a7.spcs_id AND a5.chart_id = a7.chart_id AND a5.chart_point_seq = a7.chart_point_seq AND a5.measurement_set_name = a7.measurement_set_name
LEFT JOIN P_SPC_CHART_LIMIT a10 ON a10.chart_id = a5.chart_id AND a10.limit_id = a5.limit_id
LEFT JOIN P_SPC_MEASUREMENT a4 ON a4.spcs_id = a3.spcs_id AND a4.measurement_set_name = a3.measurement_set_name
AND a4.spcs_id = a7.spcs_id AND a4.measurement_id = a7.measurement_id
WHERE a1.data_collection_time >= TRUNC(SYSDATE) - 7
AND (a3.MEASUREMENT_SET_NAME = 'CD.DCCD_STATISTICS.5051')
'''

# SQL to get the chamber or unit data from the track
SQL_CHAMBER = '''
SELECT 
          leh.lot AS lot1
         ,wch.waf3 AS waf3
         ,leh.entity AS lot_entity
         ,wch.slot AS slot
         ,wch.chamber AS chamber
         ,wch.state AS state
         ,To_Char(wch.start_time,'yyyy-mm-dd hh24:mi:ss') AS start_date
         ,To_Char(wch.end_time,'yyyy-mm-dd hh24:mi:ss') AS end_date
         ,lwr2.recipe AS lot_recipe
         ,leh.lot_abort_flag AS lot_abort_flag
         ,leh.load_port AS lot_load_port
         ,leh.processed_wafer_count AS lot_processed_wafer_count
         ,leh.reticle AS lot_reticle
         ,lrc.rework AS rework
         ,leh.operation AS operation1
         ,leh.route AS lot_route
         ,leh.product AS lot_product
         ,To_Char(leh.introduce_txn_time,'yyyy-mm-dd hh24:mi:ss') AS lot_introduce_txn_date
FROM 
F_LotEntityHist leh
INNER JOIN
F_WaferChamberHist wch
ON leh.runkey = wch.runkey
INNER JOIN F_Lot_Wafer_Recipe lwr2 ON lwr2.recipe_id=leh.lot_recipe_id
INNER JOIN F_Lot_Run_card lrc ON lrc.lotoperkey = wch.lotoperkey
WHERE
              (wch.chamber LIKE  '%ADH%'
OR wch.chamber LIKE '%PHP%'
OR wch.chamber LIKE '%CPHG%'
OR wch.chamber LIKE '%RGCH%'
OR wch.chamber LIKE '%CGCH%'
OR wch.chamber LIKE '%ITC%'
OR wch.chamber LIKE '%BCT%'
OR wch.chamber LIKE '%COT%'
OR wch.chamber LIKE '%PCT%'
OR wch.chamber LIKE '%DEV%') 
 AND      (leh.entity LIKE 'SBH20%')
 AND      (leh.lot LIKE  'W%') 
 AND      (lwr2.recipe Like '%')
 AND      wch.start_time >= TRUNC(SYSDATE) -  7
'''

# Note: Can use either last X days or specific time period using syntax:
# >= TRUNC(SYSDATE) -  50
# between '2024-08-05 00:00:00.0' and '2024-08-07 23:59:59.999'

# =============================================================================
# DATA EXTRACTION AND INITIAL PROCESSING
# =============================================================================

def load_data_from_database() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from the database and save to CSV files.
    
    Returns:
        Tuple of (df_cd, df_chamber, df_limits) DataFrames
    """
    print("Connecting to database...")
    
    try:
        conn = PyUber.connect(datasource='F21_PROD_XEUS')
        print("✓ Database connection established")
        
        print("Executing CD data query...")
        df_cd = pd.read_sql(SQL_CD, conn)
        print(f"✓ CD data loaded: {len(df_cd)} rows")
        
        print("Executing chamber data query...")
        df_chamber = pd.read_sql(SQL_CHAMBER, conn)
        print(f"✓ Chamber data loaded: {len(df_chamber)} rows")
        
        print("Executing limits data query...")
        df_limits = pd.read_sql(SQL_LIMITS, conn)
        print(f"✓ Limits data loaded: {len(df_limits)} rows")
        
        conn.close()
        print("✓ Database connection closed")
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("⚠ Consider connecting to VPN or check network connectivity")
        # Return empty DataFrames as fallback
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Save data to CSV files for backup/debugging
    try:
        print("Saving data to CSV files...")
        df_chamber.to_csv('data-chamber.csv', index=False)
        df_cd.to_csv('data-cd.csv', index=False)
        df_limits.to_csv('data-limits.csv', index=False)
        print("✓ Data saved to CSV files")
    except Exception as e:
        print(f"⚠ Warning: Could not save CSV files: {e}")
    
    return df_cd, df_chamber, df_limits

# Load the data
df_cd, df_chamber, df_limits = load_data_from_database()
# =============================================================================
# DATA PROCESSING AND TRANSFORMATION FUNCTIONS
# =============================================================================

def process_limits_data(df_limits: pd.DataFrame) -> pd.DataFrame:
    """
    Process and transform the limits dataframe.
    
    Args:
        df_limits: Raw limits dataframe
        
    Returns:
        Processed limits dataframe
    """
    if df_limits.empty:
        print("⚠ Warning: Limits dataframe is empty, skipping processing")
        return df_limits
        
    print("Processing limits data...")
    
    try:
        # Rename and transform columns
        df_limits = df_limits.rename(columns={'PROCESS_OPERATION': 'OPN'})
        df_limits['OPN'] = df_limits['OPN'].astype(str)
        df_limits = df_limits.rename(columns={'LO_CONTROL_LMT': 'LCL'})
        df_limits = df_limits.rename(columns={'UP_CONTROL_LMT': 'UCL'})
        
        # Split parameter name into components
        df_limits[['LAYER', 'STRUCTURE', 'STAT']] = df_limits['RAW_PARAMETER_NAME'].str.split(';', expand=True)
        
        # Remove PERCENT_MEASURED (STAT is WAFER_SIGMA, WAFER_MEAN, and PERCENT_MEASURED)
        df_limits = df_limits[df_limits['STAT'] != 'PERCENT_MEASURED']
        
        # Extract technology from route
        df_limits['TECH'] = df_limits['ROUTE'].str[:4].str[-2:]
        df_limits = df_limits.drop(columns=['RAW_PARAMETER_NAME', 'PRODUCT', 'ROUTE'])
        
        # Create new columns to reshape UCL, LCL from tall to wide format
        df_limits['WAFER_SIGMA_LCL'] = None
        df_limits['WAFER_SIGMA_UCL'] = None
        df_limits['WAFER_MEAN_LCL'] = None
        df_limits['WAFER_MEAN_UCL'] = None
        
        # Move values based on STAT
        df_limits.loc[df_limits['STAT'] == 'WAFER_SIGMA', 'WAFER_SIGMA_LCL'] = df_limits['LCL']
        df_limits.loc[df_limits['STAT'] == 'WAFER_SIGMA', 'WAFER_SIGMA_UCL'] = df_limits['UCL']
        df_limits.loc[df_limits['STAT'] == 'WAFER_MEAN', 'WAFER_MEAN_LCL'] = df_limits['LCL']
        df_limits.loc[df_limits['STAT'] == 'WAFER_MEAN', 'WAFER_MEAN_UCL'] = df_limits['UCL']
        
        # Drop the original columns
        df_limits = df_limits.drop(columns=['LCL', 'UCL', 'STAT'])
        df_limits = df_limits.drop_duplicates()
        
        # Replace None with NaN
        columns_to_replace = ['WAFER_SIGMA_LCL', 'WAFER_SIGMA_UCL', 'WAFER_MEAN_LCL', 'WAFER_MEAN_UCL']
        df_limits.loc[:, columns_to_replace] = df_limits[columns_to_replace].replace({None: np.nan}).infer_objects(copy=False)
        
        # Group by TECH, OPN, STRUCTURE and aggregate
        df_limits = df_limits.groupby(['TECH', 'OPN', 'SPC_OPERATION', 'STRUCTURE']).agg({
            'WAFER_SIGMA_LCL': 'mean',
            'WAFER_SIGMA_UCL': 'mean',
            'WAFER_MEAN_LCL': 'mean',
            'WAFER_MEAN_UCL': 'mean',
            'LAYER': 'first'  # Assuming LAYER is the same for identical TECH, OPN, STRUCTURE
        }).reset_index()
        
        print(f"✓ Limits data processed: {len(df_limits)} rows")
        return df_limits
        
    except Exception as e:
        print(f"✗ Error processing limits data: {e}")
        return pd.DataFrame()

def process_cd_data(df_cd: pd.DataFrame) -> pd.DataFrame:
    """
    Process and transform the CD dataframe.
    
    Args:
        df_cd: Raw CD dataframe
        
    Returns:
        Processed CD dataframe
    """
    if df_cd.empty:
        print("⚠ Warning: CD dataframe is empty, skipping processing")
        return df_cd
        
    print("Processing CD data...")
    
    try:
        # Split the 'RAW_PARAMETER_NAME' column into multiple columns
        df_cd[['PARAM1', 'LAYER', 'STRUCTURE', 'PARAM4']] = df_cd['RAW_PARAMETER_NAME'].str.split(';', expand=True)
        df_cd = df_cd.drop(columns=['RAW_PARAMETER_NAME', 'PARAM1', 'PARAM4'])
        
        # Rename columns
        df_cd = df_cd.rename(columns={'PROCESS_OPERATION': 'OPN'})
        df_cd['OPN'] = df_cd['OPN'].astype(str)
        df_cd = df_cd.rename(columns={'RAW_WAFER3': 'WAF3'})
        df_cd = df_cd.rename(columns={'ENTITY': 'Tool'})
        
        # Extract technology from route
        df_cd['TECH'] = df_cd['ROUTE'].str[:4].str[-2:]
        
        # Remove rows where 'CD' is 0
        initial_count = len(df_cd)
        df_cd = df_cd[df_cd['RAW_VALUE'] != 0]
        removed_zeros = initial_count - len(df_cd)
        if removed_zeros > 0:
            print(f"  • Removed {removed_zeros} rows with zero values")
        
        # Convert RAW_VALUE to float64, non-convertible values will be set to NaN
        df_cd['RAW_VALUE'] = pd.to_numeric(df_cd['RAW_VALUE'], errors='coerce')
        
        # Drop rows with NaN values after converting RAW_VALUE to float64
        initial_count = len(df_cd)
        df_cd = df_cd.dropna(subset=['RAW_VALUE'])
        removed_nan = initial_count - len(df_cd)
        if removed_nan > 0:
            print(f"  • Removed {removed_nan} rows with invalid numeric values")
        
        # Sort by date to get newest, post rework, data first
        df_cd = df_cd.sort_values('ENTITY_DATA_COLLECT_DATE', ascending=False)
        
        # Group by 'LOT' and 'OPN' and find the maximum 'ENTITY_DATA_COLLECT_DATE' for each group
        newest_dates = df_cd.groupby(['LOT', 'OPN'])['ENTITY_DATA_COLLECT_DATE'].max().reset_index()
        
        # Merge to keep the newest date CD data (helps keep post-rework data)
        df_cd = df_cd.merge(newest_dates, on=['LOT', 'OPN', 'ENTITY_DATA_COLLECT_DATE'])
        
        print(f"✓ CD data processed: {len(df_cd)} rows")
        return df_cd
        
    except Exception as e:
        print(f"✗ Error processing CD data: {e}")
        return pd.DataFrame()

def process_chamber_data(df_chamber: pd.DataFrame) -> pd.DataFrame:
    """
    Process and transform the chamber dataframe.
    
    Args:
        df_chamber: Raw chamber dataframe
        
    Returns:
        Processed chamber dataframe
    """
    if df_chamber.empty:
        print("⚠ Warning: Chamber dataframe is empty, skipping processing")
        return df_chamber
        
    print("Processing chamber data...")
    
    try:
        # Rename columns
        df_chamber = df_chamber.rename(columns={'LOT1': 'LOT'})
        df_chamber = df_chamber.rename(columns={'OPERATION1': 'OPN'})
        df_chamber = df_chamber.rename(columns={'LOT_INTRODUCE_TXN_DATE': 'INTRO_DATE'})
        df_chamber['OPN'] = df_chamber['OPN'].astype(str)
        df_chamber = df_chamber.rename(columns={'LOT_ROUTE': 'ROUTE'})
        
        # Extract technology from route
        df_chamber['TECH'] = df_chamber['ROUTE'].str[:4].str[-2:]
        
        # Sort the data
        df_chamber = df_chamber.sort_values(by=['OPN', 'LOT','INTRO_DATE', 'WAF3'], ascending=[True, True, True, True])
        
        print(f"✓ Chamber data processed: {len(df_chamber)} rows")
        return df_chamber
        
    except Exception as e:
        print(f"✗ Error processing chamber data: {e}")
        return pd.DataFrame()

# Process all dataframes
print("\n" + "="*50)
print("PROCESSING DATA")
print("="*50)

df_limits = process_limits_data(df_limits)
df_cd = process_cd_data(df_cd)
df_chamber = process_chamber_data(df_chamber)

# =============================================================================
# CHAMBER DATA FLATTENING
# =============================================================================

def flatten_chamber_data(df_chamber: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten chamber data from tall to wide format.
    
    Args:
        df_chamber: Processed chamber dataframe
        
    Returns:
        Flattened chamber dataframe
    """
    if df_chamber.empty:
        print("⚠ Warning: Chamber dataframe is empty, skipping flattening")
        return df_chamber
        
    print("Flattening chamber data...")
    
    try:
        # Create columns for chamber data to go from tall to wide format
        bake_num = 3
        chamberl = ['ADH', 'COT', 'ITC', 'BCT', 'PCT', 'DEV']
        ch_columns = [f'{stage}-CH' for stage in chamberl + [f'BAKE{i}' for i in range(1, bake_num + 1)]]
        
        for col in ch_columns:
            df_chamber[col] = None

        # Flatten the CHAMBER column
        def flatten_chamber(group):
            """Helper function to flatten chamber data for each group."""
            adh = cot = itc = bct = pct = dvlp = None
            bake = [None] * bake_num
            bake_count = 0
            
            for idx, row in group.iterrows():
                ch = row['CHAMBER']
                if ch.startswith(('ADH', 'CADH')):
                    adh = ch
                elif ch.startswith('COT'):
                    cot = ch
                elif ch.startswith('ITC'):
                    itc = ch
                elif ch.startswith('BCT'):
                    bct = ch
                elif ch.startswith('PCT'):
                    pct = ch
                elif ch.startswith('DEV'):
                    dvlp = ch
                elif ch.startswith(('RGCH', 'CGCH', 'CPHG', 'CPHP', 'PHP')):
                    if bake_count < 3:
                        bake[bake_count] = ch
                        bake_count += 1
                        
            group['ADH-CH'] = adh
            group['COT-CH'] = cot
            group['ITC-CH'] = itc
            group['BCT-CH'] = bct
            group['PCT-CH'] = pct
            group['DEV-CH'] = dvlp
            for i in range(bake_num):
                group[f'BAKE{i+1}-CH'] = bake[i]
            return group.iloc[0]

        df_chamber = df_chamber.groupby(['INTRO_DATE', 'OPN', 'LOT', 'WAF3'], group_keys=False).apply(flatten_chamber).reset_index(drop=True)
        
        # Keep only the most recent chamber data based on INTRO_DATE
        newest_chamber_dates = df_chamber.groupby(['LOT', 'OPN', 'WAF3'])['INTRO_DATE'].max().reset_index()
        df_chamber = df_chamber.merge(newest_chamber_dates, on=['LOT', 'OPN', 'WAF3', 'INTRO_DATE'])
        df_chamber = df_chamber.drop(columns=['CHAMBER'])

        # Process recipe data
        split_columns = df_chamber['LOT_RECIPE'].str.split(' ', n=1, expand=True)
        df_chamber['RESIST_RCP'] = split_columns[0]
        df_chamber['SCANNER'] = split_columns[1]
        df_chamber = df_chamber.drop(columns=['LOT_RECIPE'])
        df_chamber['RESIST'] = df_chamber['RESIST_RCP'].str.split('-').str[1]
        df_chamber['RESIST'] = df_chamber['RESIST'].astype(str)
        
        print(f"✓ Chamber data flattened: {len(df_chamber)} rows")
        return df_chamber
        
    except Exception as e:
        print(f"✗ Error flattening chamber data: {e}")
        return df_chamber

# Flatten chamber data and merge with CD data
df_chamber = flatten_chamber_data(df_chamber)

# Merge df_limits with df_cd by matching TECH, OPN, STRUCTURE
print("Merging limits data with CD data...")
try:
    df_cd = pd.merge(df_cd, df_limits, on=['TECH', 'OPN', 'STRUCTURE'], how='left')
    print(f"✓ Limits merged with CD data: {len(df_cd)} rows")
except Exception as e:
    print(f"✗ Error merging limits with CD data: {e}")

# Merge chamber data with CD data
print("Merging chamber data with CD data...")
try:
    df_cd = pd.merge(df_cd, df_chamber, on=['LOT', 'OPN', 'WAF3'], how='left')
    print(f"✓ Chamber data merged with CD data: {len(df_cd)} rows")
except Exception as e:
    print(f"✗ Error merging chamber data with CD data: {e}")
# =============================================================================
# FINAL DATA CLEANUP AND PREPARATION
# =============================================================================

def finalize_dataframe(df_cd: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up and finalize the main dataframe.
    
    Args:
        df_cd: Merged dataframe
        
    Returns:
        Final cleaned dataframe
    """
    if df_cd.empty:
        print("⚠ Warning: Dataframe is empty, skipping finalization")
        return df_cd
        
    print("Finalizing dataframe...")
    
    try:
        # Clean up after merging the dataframes
        columns_to_drop = ['STANDARD_FLAG', 'VALID_FLAG', 'MEASUREMENT_SET_NAME', 'ROUTE_y', 
                          'TECH_y', 'LAYER_y', 'SPC_OPERATION_y', 'LOT_ENTITY', 'STATE']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_cd.columns]
        df = df_cd.drop(columns=existing_columns_to_drop)
        
        # Rename columns for better readability
        column_renames = {
            'WAFER_MEAN_UCL': 'UCL',
            'WAFER_MEAN_LCL': 'LCL',
            'WAFER_SIGMA_UCL': 'SIGMA_UCL',
            'WAFER_SIGMA_LCL': 'SIGMA_LCL',
            'RAW_VALUE': 'CD',
            'NATIVE_X_COL': 'X-COL',
            'NATIVE_Y_ROW': 'Y-ROW',
            'LOT_RETICLE': 'RETICLE',
            'LOT_PROCESSED_WAFER_COUNT': 'QTY',
            'LOT_LOAD_PORT': 'PORT',
            'LOT_ABORT_FLAG': 'ABORT',
            'TECH_x': 'TECH',
            'LAYER_x': 'LAYER',
            'WAF3': 'WFR',
            'ROUTE_x': 'ROUTE',
            'SPC_OPERATION_x': 'SPC_OPN',
            'ENTITY_DATA_COLLECT_DATE': 'DATETIME'
        }
        
        # Only rename columns that exist
        existing_renames = {old: new for old, new in column_renames.items() if old in df.columns}
        df = df.rename(columns=existing_renames)
        
        # Create X-Y coordinate column
        if 'X-COL' in df.columns and 'Y-ROW' in df.columns:
            df['X-Y'] = df['X-COL'].astype(str) + '-' + df['Y-ROW'].astype(str)
        
        # Assign MP values based on unique STRUCTURE values for each TECH, OPN grouping
        df['MP'] = None
        
        def assign_mp(group):
            """Assign MP values to structures."""
            unique_structures = group['STRUCTURE'].unique()
            structure_map = {structure: f'MP{i+1}' for i, structure in enumerate(unique_structures)}
            group['MP'] = group['STRUCTURE'].map(structure_map)
            return group

        if 'TECH' in df.columns and 'OPN' in df.columns and 'STRUCTURE' in df.columns:
            df = df.groupby(['TECH', 'OPN']).apply(assign_mp).reset_index(drop=True)
        
        # Create TECH_LAYER column
        if 'TECH' in df.columns and 'LAYER' in df.columns:
            df['TECH_LAYER'] = df['TECH'] + '_' + df['LAYER']
        
        print(f"✓ Dataframe finalized: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"✗ Error finalizing dataframe: {e}")
        return df_cd

# Finalize the main dataframe
df = finalize_dataframe(df_cd)

# =============================================================================
# STATISTICAL ANALYSIS AND CONTROL LIMITS
# =============================================================================

def calculate_statistical_limits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistical control limits and merge with the main dataframe.
    
    Args:
        df: Main dataframe
        
    Returns:
        Dataframe with statistical limits
    """
    if df.empty:
        print("⚠ Warning: Dataframe is empty, skipping statistical calculations")
        return df
        
    print("Calculating statistical control limits...")
    
    try:
        # Group by TECH_LAYER and aggregate unique values for Tool, LOT, and WFR
        if 'TECH_LAYER' in df.columns:
            grouped = df.groupby('TECH_LAYER').agg({
                'Tool': lambda x: list(x.unique()),
                'LOT': lambda x: list(x.unique()),
                'WFR': lambda x: list(x.unique())
            }).reset_index()

            # Convert the grouped DataFrame to a dictionary for easier access
            tech_layer_dict = grouped.set_index('TECH_LAYER').T.to_dict('list')
            unique_tech_layer = df['TECH_LAYER'].unique()
        else:
            tech_layer_dict = {}
            unique_tech_layer = []

        # Convert datetime column
        if 'DATETIME' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y-%m-%d %H:%M:%S')
            df['DATE'] = df['DATETIME'].dt.normalize()

        # Calculate statistical limits
        if all(col in df.columns for col in ['TECH', 'LAYER', 'MP', 'CD']):
            grouped = df.groupby(['TECH', 'LAYER', 'MP'])['CD']
            mean = grouped.mean()
            std = grouped.std()
            ave_p3s = (mean + 3 * std).round(1)
            ave_m3s = (mean - 3 * std).round(1)

            # Reset index to avoid ambiguity during merge
            df = df.reset_index(drop=True)

            # Merge ave_p3s and ave_m3s back to the original DataFrame
            df = df.merge(ave_p3s.rename('ave_p3s'), on=['TECH', 'LAYER', 'MP'])
            df = df.merge(ave_m3s.rename('ave_m3s'), on=['TECH', 'LAYER', 'MP'])

            # Fill NaN, None, or blank values in UCL and LCL
            if 'UCL' in df.columns and 'LCL' in df.columns:
                df['UCL'] = df['UCL'].replace('', np.nan).fillna(df['ave_p3s']).infer_objects(copy=False)
                df['LCL'] = df['LCL'].replace('', np.nan).fillna(df['ave_m3s']).infer_objects(copy=False)

            # Drop the temporary columns if not needed
            df.drop(columns=['ave_p3s', 'ave_m3s'], inplace=True)

        print(f"✓ Statistical limits calculated")
        return df, tech_layer_dict, unique_tech_layer
        
    except Exception as e:
        print(f"✗ Error calculating statistical limits: {e}")
        return df, {}, []

# Calculate statistical limits
df, tech_layer_dict, unique_tech_layer = calculate_statistical_limits(df)

# Save final dataframe to Excel
try:
    print("Saving final dataframe to Excel...")
    df.to_excel('cd_df.xlsx', index=False)
    print("✓ Data saved to cd_df.xlsx")
except Exception as e:
    print(f"⚠ Warning: Could not save Excel file: {e}")

print(f"\n✓ Data processing complete! Final dataset contains {len(df)} rows")

# =============================================================================
# DASH APPLICATION CONFIGURATION
# =============================================================================

print("\n" + "="*50)
print("SETTING UP DASHBOARD")
print("="*50)

# Setup colors and themes
print("Configuring dashboard theme and colors...")

# Define chamber columns for the UI
bake_num = 3
chamberl = ['ADH', 'COT', 'ITC', 'BCT', 'PCT', 'DEV']
ch_columns = [f'{stage}-CH' for stage in chamberl + [f'BAKE{i}' for i in range(1, bake_num + 1)]]

color_map = {'EQ101': '#636efa', 'EQ102': '#ef553b', 'EQ103': '#00cc96', 'EQ104': '#ab63fa'}
table_font_size = 10
OOS_highlight = '#cfb974'

# Create theme configuration
theme_bright = dbc.themes.SANDSTONE
theme_chart_bright = pio.templates['seaborn']  # available plotly themes: simple_white, plotly, plotly_dark, ggplot2, seaborn, plotly_white, none

tukey_table_cell_highlight = '#cfb974'  # Color for highlighting a tukeyHSD flagged cell

print("✓ Theme and colors configured")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE]) # external_stylesheets=[theme_bright]

# =============================================================================
# DASH APPLICATION SETUP AND UI COMPONENTS
# =============================================================================

print("Initializing Dash application...")

try:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
    print("✓ Dash application initialized")
except Exception as e:
    print(f"✗ Error initializing Dash application: {e}")
    sys.exit(1)

print("Creating UI components...")

try:
    # Application title
    title = html.H1("PROCESS DASHBOARD", style={'font-size': '18px'})

    # Chart theme configuration
    chart_theme = theme_chart_bright
    max_table_rows = 30

    # Dropdown options for tech layer selection
    dropdown_options = []
    if len(unique_tech_layer) > 0:
        dropdown_options = [{'label': value, 'value': value} for value in unique_tech_layer]

    # Tech layer dropdown
    tech_layer_dd = html.Div([
            dcc.Dropdown(
                id='tech_layer_dd',
                options=dropdown_options,
                value='51_ODT' if '51_ODT' in unique_tech_layer else (unique_tech_layer[0] if unique_tech_layer else None)
            )])

    # MP radio buttons
    chart_MP_radio = html.Div(
        dcc.RadioItems(
            id='chart_MP_radio', 
            options=[{'value': x, 'label': x} for x in ['MP1', 'MP2', 'MP3', 'MP4', 'MP5']],
            value='MP1',
            labelStyle={'display': 'inline-block'}
            ))

    # Range slider for chart limits
    chart_range_slider = html.Div([dcc.RangeSlider(
            id = 'limit-slider',
            min = 100,
            max = 200,
            step = 10,
            value=[110, 190],
            tooltip={"placement": "bottom", "always_visible": False})])

    # Date picker
    if 'DATE' in df.columns and not df.empty:
        min_date = df['DATE'].min().date()
        max_date = df['DATE'].max().date()
    else:
        from datetime import date, timedelta
        max_date = date.today()
        min_date = max_date - timedelta(days=7)
        
    calendar = html.Div(["Date Range ",
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
            )])

    print("✓ UI components created successfully")

except Exception as e:
    print(f"✗ Error creating UI components: {e}")
    # Create minimal fallback components
    title = html.H1("PROCESS DASHBOARD - Error Loading Data")
    tech_layer_dd = html.Div("Error loading data")
    chart_MP_radio = html.Div("Error loading data")
    chart_range_slider = html.Div("Error loading data")
    calendar = html.Div("Error loading data")

summary_tableh = html.Div(["Summary Table", dt.DataTable(id='summary-table',
        columns=[
            {'name': ['Summary Table', 'No'], 'id': 'no_rows'},
            {'name': ['Summary Table','Tool'], 'id': 'tool'},
            {'name': ['Summary Table','Chamber'], 'id': 'chamber'},
            {'name': ['Summary Table','Count'], 'id': 'count'},
            {'name': ['Summary Table','Mean'], 'id': 'mean'},
            {'name': ['Summary Table','Sigma'], 'id': 'sigma'}
        ],
        data=[{'no_rows': i} for i in range(1,max_table_rows)],
        sort_action='native',
        sort_mode='multi',
        editable=False,
        merge_duplicate_headers=True,
        style_cell={'textAlign': 'center'},
        style_header={          
            'fontWeight': 'bold'
        })
        ])  # style={'display': 'inline-table', 'margin':'10px', 'width': '20%'}

tool_checklist = html.Div(dcc.Checklist(
        id="tool_list",  # id names will be used by the callback to identify the components
        options=[],  # List of tools updated by the callback
        value=[],  
        inline=True))

chamber_list_radio = html.Div(
        dcc.RadioItems(
        id='chamber', 
        options=[{'value': x, 'label': x[:-3]}  
                for x in ch_columns],  
        value='DEV-CH',   # Default
        labelStyle={'display': 'inline-block'}
        ))

line_chart1 = html.Div([dcc.Graph(figure={}, id='line-chart1')])  # figure is blank dict because created in callback below

boxplot1 = html.Div([dcc.Graph(figure={}, id='box-plot1')])

line_chart2 = html.Div([dcc.Graph(figure={}, id='line-chart2')]) 
line_chartXcol = html.Div([dcc.Graph(figure={}, id='line_chartXcol')]) 
line_chartYrow = html.Div([dcc.Graph(figure={}, id='line_chartYrow')]) 

boxplot_lot = html.Div([dcc.Graph(figure={}, id='boxplot_lot')])
line_chartXcol_lot = html.Div([dcc.Graph(figure={}, id='line_chartXcol_lot')]) 
line_chartYrow_lot = html.Div([dcc.Graph(figure={}, id='line_chartYrow_lot')]) 

line_chartXcol_lot_tool = html.Div([dcc.Graph(figure={}, id='line_chartXcol_lot_tool')]) 
line_chartYrow_lot_tool = html.Div([dcc.Graph(figure={}, id='line_chartYrow_lot_tool')]) 

# =============================================================================
# LOT SUMMARY DATA PREPARATION
# =============================================================================

def create_lot_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lot summary dataframe for the dashboard.
    
    Args:
        df: Main dataframe
        
    Returns:
        Lot summary dataframe
    """
    if df.empty:
        print("⚠ Warning: Main dataframe is empty, creating empty lot summary")
        return pd.DataFrame()
        
    print("Creating lot summary data...")
    
    try:
        # Group by LOT, OPN, and MP and calculate mean and std of CD
        required_columns = ['LOT', 'LAYER', 'MP', 'STRUCTURE', 'Tool', 'DATETIME', 'CD']
        if not all(col in df.columns for col in required_columns):
            print(f"⚠ Warning: Missing required columns for lot summary")
            return pd.DataFrame()
            
        df_lot_summary = df.groupby(['LOT', 'LAYER', 'MP']).agg(
            STRUCTURE=('STRUCTURE', 'first'),
            Tool=('Tool', 'first'),
            DATETIME=('DATETIME', 'first'),
            CD_mean=('CD', 'mean'),
            CD_std=('CD', 'std'),
            SIGMA_UCL=('SIGMA_UCL', 'first') if 'SIGMA_UCL' in df.columns else None,
            REWORK=('REWORK', 'first') if 'REWORK' in df.columns else None,
            RESIST_RCP=('RESIST_RCP', 'first') if 'RESIST_RCP' in df.columns else None,
            QTY=('QTY', 'first') if 'QTY' in df.columns else None,
            SPC_OPN=('SPC_OPN', 'first') if 'SPC_OPN' in df.columns else None,
            ROUTE=('ROUTE', 'first') if 'ROUTE' in df.columns else None,
            ABORT=('ABORT', 'first') if 'ABORT' in df.columns else None,
            OPN=('OPN', 'first') if 'OPN' in df.columns else None,
            SCANNER=('SCANNER', 'first') if 'SCANNER' in df.columns else None,
            TECH_LAYER=('TECH_LAYER', 'first') if 'TECH_LAYER' in df.columns else None,
            DATE=('DATE', 'first') if 'DATE' in df.columns else None,
        ).reset_index()
        
        # Clean up the aggregation - remove None entries
        df_lot_summary = df_lot_summary.loc[:, ~df_lot_summary.columns.isin([None])]
        
        df_lot_summary['CD_mean'] = df_lot_summary['CD_mean'].round(1)
        df_lot_summary['CD_std'] = df_lot_summary['CD_std'].round(1)
        
        print(f"✓ Lot summary created: {len(df_lot_summary)} lots")
        return df_lot_summary
        
    except Exception as e:
        print(f"✗ Error creating lot summary: {e}")
        return pd.DataFrame()

# Create lot summary
df_lot_summary = create_lot_summary(df)

# =============================================================================
# DASHBOARD UI COMPONENTS CREATION
# =============================================================================

print("Creating all dashboard UI components...")

try:
    # Summary table
    summary_tableh = html.Div(["Summary Table", dt.DataTable(id='summary-table',
            columns=[
                {'name': ['Summary Table', 'No'], 'id': 'no_rows'},
                {'name': ['Summary Table','Tool'], 'id': 'tool'},
                {'name': ['Summary Table','Chamber'], 'id': 'chamber'},
                {'name': ['Summary Table','Count'], 'id': 'count'},
                {'name': ['Summary Table','Mean'], 'id': 'mean'},
                {'name': ['Summary Table','Sigma'], 'id': 'sigma'}
            ],
            data=[{'no_rows': i} for i in range(1,max_table_rows)],
            sort_action='native',
            sort_mode='multi',
            editable=False,
            merge_duplicate_headers=True,
            style_cell={'textAlign': 'center'},
            style_header={'fontWeight': 'bold'})
            ])

    # Tool checklist
    tool_checklist = html.Div(dcc.Checklist(
            id="tool_list",
            options=[],
            value=[],  
            inline=True))

    # Chamber list radio buttons
    chamber_list_radio = html.Div(
            dcc.RadioItems(
            id='chamber', 
            options=[{'value': x, 'label': x[:-3]} for x in ch_columns],  
            value='DEV-CH',
            labelStyle={'display': 'inline-block'}
            ))

    # Chart components
    line_chart1 = html.Div([dcc.Graph(figure={}, id='line-chart1')])
    boxplot1 = html.Div([dcc.Graph(figure={}, id='box-plot1')])
    line_chart2 = html.Div([dcc.Graph(figure={}, id='line-chart2')]) 
    line_chartXcol = html.Div([dcc.Graph(figure={}, id='line_chartXcol')]) 
    line_chartYrow = html.Div([dcc.Graph(figure={}, id='line_chartYrow')]) 
    boxplot_lot = html.Div([dcc.Graph(figure={}, id='boxplot_lot')])
    line_chartXcol_lot = html.Div([dcc.Graph(figure={}, id='line_chartXcol_lot')]) 
    line_chartYrow_lot = html.Div([dcc.Graph(figure={}, id='line_chartYrow_lot')]) 
    line_chartXcol_lot_tool = html.Div([dcc.Graph(figure={}, id='line_chartXcol_lot_tool')]) 
    line_chartYrow_lot_tool = html.Div([dcc.Graph(figure={}, id='line_chartYrow_lot_tool')]) 

    print("✓ All UI components created successfully")

except Exception as e:
    print(f"✗ Error creating UI components: {e}")
    # Create minimal fallback components
    summary_tableh = html.Div("Error loading summary table")
    tool_checklist = html.Div("Error loading tool checklist")
    chamber_list_radio = html.Div("Error loading chamber selection")
    line_chart1 = html.Div("Error loading charts")
    boxplot1 = html.Div("Error loading charts")
    line_chart2 = html.Div("Error loading charts")
    line_chartXcol = html.Div("Error loading charts")
    line_chartYrow = html.Div("Error loading charts")
    boxplot_lot = html.Div("Error loading charts")
    line_chartXcol_lot = html.Div("Error loading charts")
    line_chartYrow_lot = html.Div("Error loading charts")
    line_chartXcol_lot_tool = html.Div("Error loading charts")
    line_chartYrow_lot_tool = html.Div("Error loading charts")

lot_list_table = html.Div([
    html.Button("Unselect All", id="unselect-button"),
    dt.DataTable(
        id='lot_list_table',
        columns=[{'name': col, 'id': col} for col in df_lot_summary.columns],
        data=df_lot_summary.to_dict('records'),
        style_table={'height': '200px', 'overflowX': 'auto', 'overflowY': 'auto'},
        row_selectable='multi',
        fixed_rows={'headers': True, 'data': 0},
        sort_action='native',
        sort_mode='multi',
        filter_action='native',  # Enable column filtering
        style_cell={'textAlign': 'center'},
        tooltip_header={col: {'value': col, 'type': 'markdown'} for col in df_lot_summary.columns}
    )])


# Layout of the dash graphs, tables, drop down menus, etc
# Using dbc container for styling/formatting
app.layout = dbc.Container([
    # Summary and Tukey Tables
    dbc.Row([
        dbc.Col(calendar, width={"size":5, "justify":"left"}),
        dbc.Col(tech_layer_dd, width={"size":3, "justify":"between"}),
        dbc.Col(chart_MP_radio, width={"size":4})]),
    # Charts and Boxplot
    dbc.Row([
        dbc.Col(chart_range_slider, width={"size":12})]),
    dbc.Row([
        dbc.Col(tool_checklist, width={"size":6}),
        dbc.Col(chamber_list_radio, width={"size":6})]),
    dbc.Row([
        dbc.Col(line_chart1, width={"size":6}),
        dbc.Col(boxplot1, width={"size":6})]),
    dbc.Row([
        dbc.Col(line_chart2, width={"size":6}),
        dbc.Col(line_chartXcol, width={"size":3}),
        dbc.Col(line_chartYrow, width={"size":3})]),
    dbc.Row([
        dbc.Col(lot_list_table, width={"size":12})]),
    dbc.Row([
        dbc.Col(boxplot_lot, width={"size":6}),
        dbc.Col(line_chartXcol_lot, width={"size":3}),
        dbc.Col(line_chartYrow_lot, width={"size":3})]),
    dbc.Row([
        dbc.Col(summary_tableh, width={"size":6}),
        dbc.Col(line_chartXcol_lot_tool, width={"size":3}),
        dbc.Col(line_chartYrow_lot_tool, width={"size":3})]),
    ], fluid=True, className="dbc dbc-row-selectable")

#=====CREATE INTERACTIVE GRAPHS=============
# Callbacks are used to update the graphs and tables when the user changes the inputs 
# chart upper lower limit slider
@app.callback(
    Output('limit-slider', 'min'),
    Output('limit-slider', 'max'),
    Output('limit-slider', 'step'),
    Output('limit-slider', 'value'),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value')
)
def update_slider(selected_tech_layer, selected_chart_mpx):
    # Filter the DataFrame based on selected_tech_layer and selected_chart_mpx
    filtered_df = df[(df['TECH_LAYER'] == selected_tech_layer) & (df['MP'] == selected_chart_mpx)]
    # Ensure there is data after filtering
    if filtered_df.empty:
        raise ValueError("No data found for the selected TECH_LAYER and MP combination.")
    # Get the LCL and UCL values
    lcl = filtered_df['LCL'].values[0]
    ucl = filtered_df['UCL'].values[0]
    # Calculate the limits and step
    lower_limit = lcl - (0.1 * abs(lcl))
    upper_limit = ucl + (0.1 * abs(ucl))
    step = round((upper_limit - lower_limit) / 40, 1)
    value = [lcl, ucl]
    return lower_limit, upper_limit, step, value

# Define the callback to update the tool_checklist based on which tools have been selected
@app.callback(
    Output('tool_list', 'options'),
    Input('tech_layer_dd', 'value')
)
def update_tool_checklist(selected_tech_layer):
    # tech_layer_dict is a dictionary with keys as tech_layer values
    # and values as lists of tool options
    tool_list = tech_layer_dict.get(selected_tech_layer, [])
    return tool_list[0] if tool_list else []

# Create plotly express by Xcol for lot selection with tool color
@app.callback(
    Output("line_chartXcol_lot_tool", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def update_line_chart_tool_xcol(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):    # callback function arg 'tool' refers to the component property of the input or "value" above
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows if i < len(data)]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    
    # Calculate the middle value of Y-ROW from ALL data (not just selected tools)
    # This ensures consistent filtering regardless of which tools are selected
    all_data_for_median = filtered_data.copy()  # Use the same lot filtering but before tool filtering
    middle_y_row = all_data_for_median['Y-ROW'].median()
    
    mask = filtered_data.Tool.isin(tool) 
    filtered_data = filtered_data[mask]
    # Filter data to include only rows where Y-ROW is within ±1 of the middle value
    filtered_data = filtered_data[(filtered_data['Y-ROW'] >= middle_y_row - 1) & (filtered_data['Y-ROW'] <= middle_y_row + 1)]
    # Sort the filtered data by X-COL
    sorted_data = filtered_data.sort_values(by='X-COL')
    sorted_tools = sorted([str(tl) for tl in filtered_data['Tool'].unique().tolist() if tl is not None])
    title = "Selected Lots by Tool"                                  
    fig = px.line(sorted_data,   
        x='X-COL', y='CD', color='Tool'
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme,
        category_orders={'Tool': sorted_tools})
    fig.update_traces(mode="markers")
    
    # Add trend lines for each tool (connecting means at each X-COL)
    colors = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
    for i, tl in enumerate(sorted_tools):
        tool_data = sorted_data[sorted_data['Tool'] == tl]
        if len(tool_data) > 0:
            # Calculate mean CD at each X-COL position for this tool
            tool_means = tool_data.groupby('X-COL')['CD'].mean().reset_index()
            # Sort by X-COL to ensure proper line connection
            tool_means = tool_means.sort_values('X-COL')
            
            # Add trend line connecting the means
            fig.add_trace(go.Scatter(
                x=tool_means['X-COL'], 
                y=tool_means['CD'],
                mode='lines',
                name=f'Trend {tl}',
                line=dict(color=colors[i % len(colors)], dash='dot', width=3),
                showlegend=False
            ))
    
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Yrow for lot selection with tool color
@app.callback(
    Output("line_chartYrow_lot_tool", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def update_line_chart_tool_yrow(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):    # callback function arg 'tool' refers to the component property of the input or "value" above
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows if i < len(data)]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    
    # Calculate the middle value of X-COL from ALL data (not just selected tools)
    # This ensures consistent filtering regardless of which tools are selected
    all_data_for_median = filtered_data.copy()  # Use the same lot filtering but before tool filtering
    middle_x_col = all_data_for_median['X-COL'].median()
    
    mask = filtered_data.Tool.isin(tool) 
    filtered_data = filtered_data[mask]
    # Filter data to include only rows where X-COL is within ±1 of the middle value
    filtered_data = filtered_data[(filtered_data['X-COL'] >= middle_x_col - 1) & (filtered_data['X-COL'] <= middle_x_col + 1)]
    # Sort the filtered data by Y-ROW
    sorted_data = filtered_data.sort_values(by='Y-ROW')
    sorted_tools = sorted([str(tl) for tl in filtered_data['Tool'].unique().tolist() if tl is not None])
    title = "Across Wafer by Tool"                                  
    fig = px.line(sorted_data,   
        x='Y-ROW', y='CD', color='Tool'
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme,
        category_orders={'Tool': sorted_tools})
    fig.update_traces(mode="markers")
    
    # Add trend lines for each tool (connecting means at each Y-ROW)
    colors = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
    for i, tl in enumerate(sorted_tools):
        tool_data = sorted_data[sorted_data['Tool'] == tl]
        if len(tool_data) > 0:
            # Calculate mean CD at each Y-ROW position for this tool
            tool_means = tool_data.groupby('Y-ROW')['CD'].mean().reset_index()
            # Sort by Y-ROW to ensure proper line connection
            tool_means = tool_means.sort_values('Y-ROW')
            
            # Add trend line connecting the means
            fig.add_trace(go.Scatter(
                x=tool_means['Y-ROW'], 
                y=tool_means['CD'],
                mode='lines',
                name=f'Trend {tl}',
                line=dict(color=colors[i % len(colors)], dash='dot', width=3),
                showlegend=False
            ))
    
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Summary table update 
@app.callback(
    Output('summary-table', 'data'),     # args are component id and then component property. component property is passed
    Input('date-range', 'start_date'),  # in order to the chart function below
    Input('date-range', 'end_date'),
    State('summary-table', 'data'),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'),
    Input('chamber', 'value'),  # Add chamber input
    Input('lot_list_table', 'selected_rows'),
    State('lot_list_table', 'data'))
def summary_table(start_date, end_date, rows, selected_tech_layer, selected_chart_mpx, selected_chamber, selected_rows, data):
    # Filter the data based on selected rows
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows if i < len(data)]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots"
        )
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx"
        )
    
    # Group by Tool and selected_chamber and aggregate
    dfsummary = filtered_data.groupby(['Tool', selected_chamber]).agg(
        count=('CD', 'size'),
        mean=('CD', 'mean'),
        std=('CD', 'std')
    ).reset_index()
    
    # Format mean and std columns
    dfsummary['mean'] = dfsummary['mean'].map('{:.1f}'.format)
    dfsummary['std'] = dfsummary['std'].map('{:.1f}'.format)
    
    # Map summary data to rows
    summaryd = {'tool': 'Tool', 'chamber': selected_chamber, 'count': 'count', 'mean': 'mean', 'sigma': 'std'}
    for i, row in enumerate(rows):
        for key, value in summaryd.items():
            try:
                row[key] = dfsummary.at[i, value]
            except KeyError:
                row[key] = ''
    
    return rows

# Lot list table
@app.callback(
    Output('lot_list_table', 'data'),     # args are component id and then component property. component property is passed
    Input('date-range', 'start_date'),  # in order to the chart function below
    Input('date-range', 'end_date'),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'))
def lot_list_table(start_date, end_date, selected_tech_layer, selected_chart_mpx):
    filtered_data = df_lot_summary.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx") 
    return filtered_data.to_dict('records')

# Create plotly express line chart 1
@app.callback(
    Output("line-chart1", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    mask = filtered_data.Tool.isin(tool) 
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"                                  # Create a panda series with True/False of only tools selected 
    fig = px.line(filtered_data[mask],   
        x='DATETIME', y='CD', color='Tool'
        ,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR', 'RESIST_RCP', 'PRODUCT', 'RETICLE', 'SLOT', 'ITC-CH', 'BCT-CH', 'COT-CH', 'PCT-CH', 'BAKE1-CH', 'BAKE1-CH','BAKE2-CH','DEV-CH','BAKE3-CH','X-Y']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express box plot
@app.callback(
    Output("box-plot1", "figure"), 
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'))
def generate_bx_chamber(start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    sorted_chambers = sorted([str(ch) for ch in filtered_data[chamber].unique().tolist() if ch is not None])
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"
    fig = px.box(filtered_data, x="Tool", y='CD', color=chamber, notched=True, template=chart_theme, hover_data=[filtered_data['LOT'], filtered_data['WFR'],  filtered_data['RESIST_RCP']], category_orders={"Tool": tooll, chamber: sorted_chambers}, title=title)
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express box plot for lot selections
@app.callback(
    Output("boxplot_lot", "figure"), 
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def generate_bx_chamber(start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows if i < len(data)]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    sorted_chambers = sorted([str(ch) for ch in filtered_data[chamber].unique().tolist() if ch is not None])
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"
    fig = px.box(filtered_data, x="Tool", y='CD', color=chamber, notched=True, template=chart_theme, hover_data=[filtered_data['LOT'], filtered_data['WFR'],  filtered_data['RESIST_RCP']], category_orders={"Tool": tooll, chamber: sorted_chambers}, title=title)
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Callback to unselect all rows from the lot table
@app.callback(
    Output('lot_list_table', 'selected_rows'),
    Input('unselect-button', 'n_clicks')
)
def unselect_all(n_clicks):
    if n_clicks:
        return []
    return dash.no_update

# Create plotly express line chart 2 - CD by chamber
@app.callback(
    Output("line-chart2", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    mask = filtered_data.Tool.isin(tool) 
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    sorted_chambers = sorted([str(ch) for ch in filtered_data[chamber].unique().tolist() if ch is not None])
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"                                  # Create a panda series with True/False of only tools selected 
    fig = px.line(filtered_data[mask],   
        x='DATETIME', y='CD', color=chamber
        ,category_orders={'Tool': tooll, chamber: sorted_chambers}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR', 'RESIST_RCP', 'PRODUCT', 'RETICLE', 'SLOT', 'ITC-CH', 'BCT-CH', 'COT-CH', 'PCT-CH', 'BAKE1-CH', 'BAKE1-CH','BAKE2-CH','DEV-CH','BAKE3-CH','X-Y']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Xcol
@app.callback(
    Output("line_chartXcol", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    mask = filtered_data.Tool.isin(tool) 
    filtered_data = filtered_data[mask]
    # Calculate the middle value of Y-ROW
    middle_y_row = filtered_data['Y-ROW'].median()
    # Filter data to include only rows where Y-ROW is within ±1 of the middle value
    filtered_data = filtered_data[(filtered_data['Y-ROW'] >= middle_y_row - 1) & (filtered_data['Y-ROW'] <= middle_y_row + 1)]
    # Sort the filtered data by X-COL
    sorted_data = filtered_data.sort_values(by='X-COL')
    sorted_chambers = sorted([str(ch) for ch in filtered_data[chamber].unique().tolist() if ch is not None])
    title = "Across Lots"                                  
    fig = px.line(sorted_data,   
        x='X-COL', y='CD', color=chamber
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme,
        category_orders={chamber: sorted_chambers})
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Yrow
@app.callback(
    Output("line_chartYrow", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    mask = filtered_data.Tool.isin(tool) 
    filtered_data = filtered_data[mask]
    # Calculate the middle value of X-COL
    middle_x_col = filtered_data['X-COL'].median()
    # Filter data to include only rows where X-COL is within ±1 of the middle value
    filtered_data = filtered_data[(filtered_data['X-COL'] >= middle_x_col - 1) & (filtered_data['X-COL'] <= middle_x_col + 1)]
    # Sort the filtered data by Y-ROW
    sorted_data = filtered_data.sort_values(by='Y-ROW')
    sorted_chambers = sorted([str(ch) for ch in filtered_data[chamber].unique().tolist() if ch is not None])
    title = "Across Wafer"                                  
    fig = px.line(sorted_data,   
        x='Y-ROW', y='CD', color=chamber
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme,
        category_orders={chamber: sorted_chambers})
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Xcol for lot selection
@app.callback(
    Output("line_chartXcol_lot", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):    # callback function arg 'tool' refers to the component property of the input or "value" above
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows if i < len(data)]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    mask = filtered_data.Tool.isin(tool) 
    filtered_data = filtered_data[mask]
    # Calculate the middle value of Y-ROW
    middle_y_row = filtered_data['Y-ROW'].median()
    # Filter data to include only rows where Y-ROW is within ±1 of the middle value
    filtered_data = filtered_data[(filtered_data['Y-ROW'] >= middle_y_row - 1) & (filtered_data['Y-ROW'] <= middle_y_row + 1)]
    # Sort the filtered data by X-COL
    sorted_data = filtered_data.sort_values(by='X-COL')
    sorted_chambers = sorted([str(ch) for ch in filtered_data[chamber].unique().tolist() if ch is not None])
    title = "Selected Lots"                                  
    fig = px.line(sorted_data,   
        x='X-COL', y='CD', color=chamber
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme,
        category_orders={chamber: sorted_chambers})
    fig.update_traces(mode="markers")
    
    # Add trend lines for each chamber (connecting means at each X-COL)
    colors = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
    for i, ch in enumerate(sorted_chambers):
        chamber_data = sorted_data[sorted_data[chamber] == ch]
        if len(chamber_data) > 0:
            # Calculate mean CD at each X-COL position for this chamber
            chamber_means = chamber_data.groupby('X-COL')['CD'].mean().reset_index()
            # Sort by X-COL to ensure proper line connection
            chamber_means = chamber_means.sort_values('X-COL')
            
            # Add trend line connecting the means
            fig.add_trace(go.Scatter(
                x=chamber_means['X-COL'], 
                y=chamber_means['CD'],
                mode='lines',
                name=f'Trend {ch}',
                line=dict(color=colors[i % len(colors)], dash='dot', width=3),
                showlegend=False
            ))
    
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Yrow for lot selection
@app.callback(
    Output("line_chartYrow_lot", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):    # callback function arg 'tool' refers to the component property of the input or "value" above
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows if i < len(data)]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    mask = filtered_data.Tool.isin(tool) 
    filtered_data = filtered_data[mask]
    # Calculate the middle value of X-COL
    middle_x_col = filtered_data['X-COL'].median()
    # Filter data to include only rows where X-COL is within ±1 of the middle value
    filtered_data = filtered_data[(filtered_data['X-COL'] >= middle_x_col - 1) & (filtered_data['X-COL'] <= middle_x_col + 1)]
    # Sort the filtered data by Y-ROW
    sorted_data = filtered_data.sort_values(by='Y-ROW')
    sorted_chambers = sorted([str(ch) for ch in filtered_data[chamber].unique().tolist() if ch is not None])
    title = "Across Wafer"                                  
    fig = px.line(sorted_data,   
        x='Y-ROW', y='CD', color=chamber
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme,
        category_orders={chamber: sorted_chambers})
    fig.update_traces(mode="markers")
    
    # Add trend lines for each chamber (connecting means at each Y-ROW)
    colors = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
    for i, ch in enumerate(sorted_chambers):
        chamber_data = sorted_data[sorted_data[chamber] == ch]
        if len(chamber_data) > 0:
            # Calculate mean CD at each Y-ROW position for this chamber
            chamber_means = chamber_data.groupby('Y-ROW')['CD'].mean().reset_index()
            # Sort by Y-ROW to ensure proper line connection
            chamber_means = chamber_means.sort_values('Y-ROW')
            
            # Add trend line connecting the means
            fig.add_trace(go.Scatter(
                x=chamber_means['Y-ROW'], 
                y=chamber_means['CD'],
                mode='lines',
                name=f'Trend {ch}',
                line=dict(color=colors[i % len(colors)], dash='dot', width=3),
                showlegend=False
            ))
    
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("STARTING DASHBOARD APPLICATION")
    print("="*50)
    
    try:
        print("Dashboard is starting...")
        print("✓ Open your browser and navigate to: http://127.0.0.1:8050")
        print("✓ Press Ctrl+C to stop the application")
        app.run(debug=True)
    except Exception as e:
        print(f"✗ Error starting application: {e}")
        print("Please check the error messages above and ensure all data is loaded correctly.")
    finally:
        print("\n✓ Application stopped")
        print("="*50)