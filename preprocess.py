import pandas as pd
import re

# Function to clean up data before preprocessing
def clean_data(df):
    # Trim spaces for string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convert text to lowercase for string columns
        df[col] = df[col].map(lambda x: x.lower() if isinstance(x, str) else x)
        
        # Handle missing values - fill NaN in string columns with ''
    df = df.fillna('')

    return df


def preprocess_alm_data(df):
    df = clean_data(df)

    # Extract package size (e.g., '700ml', '330ml') and remove it from description
    df['ITEM_SIZE'] = df['ITEM_DESCRIPTION'].str.extract(r'(\d+\s?ml|\d+\s?l|\d+ml|\d+L|\d+\s?g|\d+g)', expand=False)
    df['ITEM_DESCRIPTION'] = df['ITEM_DESCRIPTION'].str.replace(r'(\d+\s?ml|\d+\s?l|\d+ml|\d+L|\d+\s?g|\d+g)', '', regex=True)

    # Extract packaging type (e.g., 'bottle', 'can') from description
    df['ITEM_DESCRIPTION'] = df['ITEM_DESCRIPTION'].str.replace(r'(bottle|can|pack|box|carton)', '', regex=True)

    # Clean up extra whitespace
    df['ITEM_DESCRIPTION'] = df['ITEM_DESCRIPTION'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return df

def preprocess_danmurphys_data(df):
    df = clean_data(df)

    # Extract alcohol volume (e.g., '40%', '%40') and remove it from product name
    df['ALCOHOL_VOLUME'] = df['PRODUCT_NAME'].str.extract(r'(\d+%|%\s?\d+)', expand=False)
    df['PRODUCT_NAME'] = df['PRODUCT_NAME'].str.replace(r'(\d+%|%\s?\d+)', '', regex=True)

    # Remove packaging type (e.g., 'bottle', 'can') from product name
    df['PRODUCT_NAME'] = df['PRODUCT_NAME'].str.replace(r'(bottle|can|pack|box|carton)', '', regex=True)

    # Remove the package size only if it exists in PACKAGE_SIZE to avoid redundancy
    if 'PACKAGE_SIZE' in df.columns:
        df['PRODUCT_NAME'] = df.apply(
            lambda row: re.sub(r'\b' + re.escape(row['PACKAGE_SIZE']) + r'\b', '', row['PRODUCT_NAME']) 
            if pd.notnull(row['PACKAGE_SIZE']) else row['PRODUCT_NAME'], axis=1
        )

    # Clean up extra whitespace
    df['PRODUCT_NAME'] = df['PRODUCT_NAME'].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df

# Dictionary to map product companies to their preprocessing functions
PREPROCESS_FUNCTIONS = {
    'ALM': preprocess_alm_data,
    'DANMURPHYS': preprocess_danmurphys_data,
}

def preprocess_product_data(df, company_name):
    """Preprocess data based on the company."""
    if company_name not in PREPROCESS_FUNCTIONS:
        raise ValueError(f"No preprocessing function defined for company: {company_name}")
    return PREPROCESS_FUNCTIONS[company_name](df)
