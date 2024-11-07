import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

sample_size = 1000
similarity_threshold = 0.8

# Function to concatenate multiple fields into a single description
def concatenate_fields(row, fields):
    return ' '.join([str(row[field]) for field in fields if pd.notnull(row[field])])

# Function to save matching result into a .csv file
def save_matching_result(matches):
    try:
        timestamp = datetime.today().strftime('%Y-%m-%dT%H%M%S')
        if not os.path.exists('matchings'):
            os.makedirs('matchings')
        output_file_path = os.path.join('matchings', 'matching_' + timestamp + '.csv')
            
        # Convert matches to a DataFrame
        matches_df = pd.DataFrame(matches)

        # Display matched products
        print(f"Found {len(matches_df)} matches above the similarity threshold of {similarity_threshold}.\n")
        print(matches_df.head())

        # Save matches to a CSV file
        matches_df.to_csv(output_file_path, index=False)

        print(f"Matching saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving matching: {e}")

def main():
    # Load data
    alm_data = pd.read_csv('products/20241101_ALM_PRODUCTS.csv', delimiter='|')
    danmurphys_data = pd.read_csv('products/20241101_DANMURPHYS_PRODUCTS.csv', delimiter='|')

    alm_sample = alm_data.sample(n=min(sample_size, len(alm_data)), random_state=42).reset_index(drop=True)
    danmurphys_sample = danmurphys_data.sample(n=min(sample_size, len(danmurphys_data)), random_state=42).reset_index(drop=True)

    # Define fields to concatenate for embedding
    alm_fields = ['ITEM_DESCRIPTION', 'ITEM_BRAND', 'ITEM_SIZE', 'RETAIL_UNIT_LUC_PACK', 'CATEGORY', 'ALCOHOL_STRENGTH_PERC']
    danmurphys_fields = ['PRODUCT_NAME', 'BRAND', 'PACKAGE_SIZE', 'PACK_FORMAT', 'CATEGORY', 'ALCOHOL_VOLUME']

    # Create concatenated descriptions
    alm_sample['full_description'] = alm_sample.apply(lambda row: concatenate_fields(row, alm_fields), axis=1)
    danmurphys_sample['full_description'] = danmurphys_sample.apply(lambda row: concatenate_fields(row, danmurphys_fields), axis=1)

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    print("Generating embeddings for ALM products...")
    alm_embeddings = model.encode(alm_sample['full_description'].tolist(), convert_to_tensor=False)
    
    print("Generating embeddings for Dan Murphy's products...")
    danmurphys_embeddings = model.encode(danmurphys_sample['full_description'].tolist(), convert_to_tensor=False)

    # Compute cosine similarity
    print("Computing cosine similarity...")
    cosine_similarities = cosine_similarity(alm_embeddings, danmurphys_embeddings)

    # Find matches above the threshold
    matches = []
    for i, alm_similarities in enumerate(cosine_similarities):
        match_indices = [j for j, score in enumerate(alm_similarities) if score >= similarity_threshold]
        for j in match_indices:
            matches.append({
                'ALM Product': alm_sample.iloc[i]['ITEM_DESCRIPTION'],
                'ALM Brand': alm_sample.iloc[i]['ITEM_BRAND'],
                'ALM Pack Size': alm_sample.iloc[i]['ITEM_SIZE'],
                'ALM Pack Format': alm_sample.iloc[i]['RETAIL_UNIT_LUC_PACK'],
                'Dan Murphy\'s Product': danmurphys_sample.iloc[j]['PRODUCT_NAME'],
                'Dan Murphy\'s Brand': danmurphys_sample.iloc[j]['BRAND'],
                'Dan Murphy\'s Pack Size': danmurphys_sample.iloc[j]['PACKAGE_SIZE'],
                'Dan Murphy\'s Pack Format': danmurphys_sample.iloc[j]['PACK_FORMAT'],
                'Dan Murphy\'s Price': danmurphys_sample.iloc[j]['PRICE'],
                'Similarity Score': alm_similarities[j]
            })

    save_matching_result(matches)

if __name__ == "__main__":
    main()
