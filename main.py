import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from preprocess import preprocess_product_data

load_dotenv(override=True)

alm_file = os.getenv('ALM_FILE_PATH')
danmurphys_file = os.getenv('DANMURPHYS_FILE_PATH')
qdrant_host = os.getenv('QDRANT_HOST')
qdrant_port = int(os.getenv('QDRANT_PORT'))
alm_collection_name = os.getenv('ALM_COLLECTION_NAME')
dan_murphys_collection_name = os.getenv('DANMURPHYS_COLLECTION_NAME')
sample_size = int(os.getenv('SAMPLE_SIZE'))
similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD'))
top_k_results = int(os.getenv('TOP_K_RESULTS'))

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
    alm_data = pd.read_csv('products/20241101_ALM_PRODUCTS.csv', delimiter='|', low_memory=False)
    danmurphys_data = pd.read_csv('products/20241101_DANMURPHYS_PRODUCTS.csv', delimiter='|', low_memory=False)

    # Use whole datasets if the sample_size is set to zero
    if sample_size > 0:
        alm_data = alm_data.sample(n=min(sample_size, len(alm_data)), random_state=42).reset_index(drop=True)
        danmurphys_data = danmurphys_data.sample(n=min(sample_size, len(danmurphys_data)), random_state=42).reset_index(drop=True)

    # Preprocess product data 
    print("Preprocessing datas...")
    alm_data = preprocess_product_data(alm_data, 'ALM')
    danmurphys_data = preprocess_product_data(danmurphys_data, 'DANMURPHYS')

    # Define fields to concatenate for embedding
    alm_fields = ['ITEM_DESCRIPTION', 'ITEM_BRAND', 'ITEM_SIZE', 'RETAIL_UNIT_LUC_PACK', 'CATEGORY', 'ALCOHOL_STRENGTH_PERC']
    danmurphys_fields = ['PRODUCT_NAME', 'BRAND', 'PACKAGE_SIZE', 'PACK_FORMAT', 'CATEGORY', 'ALCOHOL_VOLUME']

    # Create concatenated descriptions
    alm_data['full_description'] = alm_data.apply(lambda row: concatenate_fields(row, alm_fields), axis=1)
    danmurphys_data['full_description'] = danmurphys_data.apply(lambda row: concatenate_fields(row, danmurphys_fields), axis=1)

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    print("Generating embeddings for ALM products...")
    alm_embeddings = model.encode(alm_data['full_description'].tolist(), convert_to_tensor=False)
    
    print("Generating embeddings for Dan Murphy's products...")
    danmurphys_embeddings = model.encode(danmurphys_data['full_description'].tolist(), convert_to_tensor=False)

    # Initialize Qdrant client
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Create collections in Qdrant for each dataset
    if not client.collection_exists(alm_collection_name):
        client.create_collection(
            collection_name=alm_collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    if not client.collection_exists(dan_murphys_collection_name):
        client.create_collection(
            collection_name=dan_murphys_collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Upload embeddings to Qdrant
    print("Uploading ALM embeddings to Qdrant...")
    client.upload_collection(
        collection_name=alm_collection_name,
        vectors=alm_embeddings,
        payload=[{"full_description": desc, "ITEM_NUMBER": str(item)} for desc, item in zip(alm_data['full_description'], alm_data['ITEM_NUMBER'])],
        ids=[i for i in range(len(alm_embeddings))]
    )

    print("Uploading Dan Murphy's embeddings to Qdrant...")
    client.upload_collection(
        collection_name=dan_murphys_collection_name,
        vectors=danmurphys_embeddings,
        payload=[{"full_description": desc, "STOCKCODE": str(item)} for desc, item in zip(danmurphys_data['full_description'], danmurphys_data['STOCKCODE'])],
        ids=[i for i in range(len(danmurphys_embeddings))]
    )

    # Perform matching: Search for similar products
    matches = []
    for i, alm_embedding in enumerate(alm_embeddings):
        search_results = client.search(
            collection_name=dan_murphys_collection_name,
            query_vector=alm_embedding,
            limit=top_k_results
        )

        for result in search_results:
            if result.score >= similarity_threshold:
                matches.append({
                    'ALM Product': alm_data.iloc[i]['ITEM_DESCRIPTION'],
                    'ALM Brand': alm_data.iloc[i]['ITEM_BRAND'],
                    'ALM Pack Size': alm_data.iloc[i]['ITEM_SIZE'],
                    'ALM Pack Format': alm_data.iloc[i]['RETAIL_UNIT_LUC_PACK'],
                    'ALM ID': alm_data.iloc[i]['ITEM_NUMBER'],
                    "Dan Murphy's Product": result.payload['full_description'],
                    'Dan Murphy\'s Price': danmurphys_data.loc[result.id, 'PRICE'],
                    'Dan Murphy\'s ID': danmurphys_data.loc[result.id, 'STOCKCODE'],
                    'Similarity Score': result.score
                })

    save_matching_result(matches)

if __name__ == "__main__":
    main()
