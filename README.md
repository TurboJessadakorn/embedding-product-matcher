# Product Matcher using Vector Embeddings and Qdrant

This repository contains a script for matching product data between two datasets (e.g., ALM and Dan Murphy's) using sentence embeddings and vector search with [Qdrant](https://qdrant.tech/). The script helps identify similar products based on various fields and can be used as a Proof of Concept (PoC) for efficient product matching in large datasets.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)

## Overview
The script loads product data from two sources, generates sentence embeddings for product descriptions, and uploads them to Qdrant for vector-based similarity matching. It checks for existing embeddings to avoid duplication and stores match results in CSV format for easy reference.

## Requirements
- [Python 3.7+](https://www.python.org/downloads/) (Reccomend version 3.11 for compability)
- [Sentence Transformers](https://www.sbert.net/)
- [Qdrant Client for Python](https://qdrant.tech/documentation/quick-start/)
- [Pandas](https://pandas.pydata.org/)
- [Dotenv](https://pypi.org/project/python-dotenv/)

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/embedding-product-matcher.git
   cd embedding-product-matcher
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Qdrant: Make sure you have a Qdrant instance running locally or on a server. You can also use Qdrant Cloud if preferred. Below is the script to run Qdrant locally using docker**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   
4. **Copy .env.example file and rename it to .env:**
   
   ```bash
   cp .env.example .env
   ```
   
   In the .env file set the following variables.
   
   ```plaintext
    ALM_FILE_PATH=path/to/alm_products.csv
    DANMURPHYS_FILE_PATH=path/to/danmurphys_products.csv
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    ALM_COLLECTION_NAME=alm_products_collection
    DANMURPHYS_COLLECTION_NAME=danmurphys_products_collection
    SAMPLE_SIZE=1000
    SIMILARITY_THRESHOLD=0.8
    TOP_K_RESULTS=5
    ```
      - ALM_FILE_PATH and DANMURPHYS_FILE_PATH should point to the ALM and Dan Murphy's data files, respectively.
      - QDRANT_HOST and QDRANT_PORT define the Qdrant instance location.
      - ALM_COLLECTION_NAME and DANMURPHYS_COLLECTION_NAME are the Qdrant collection names.
      - SAMPLE_SIZE controls the number of samples used from each dataset (set to 0 to use the entire dataset).
      - SIMILARITY_THRESHOLD defines the minimum similarity score for a match.
      - TOP_K_RESULTS is the number of top matches to retrieve for each product.

## Usage

  **Run the product-matching script as follows:**
  
  ```bash
  python main.py
  ```

  **Output**
  
  - The script will output matched products in the console and save them in a timestamped CSV file in the matchings/ folder.

## How It Works

  1. **Data Loading**: Loads product data from ALM and Dan Murphy's files.
  2. **Data Preprocessing**: Concatenates relevant fields for each product to create a unified text description.
  3. **Embeddings**: Generates sentence embeddings using SentenceTransformer.
  4. **Qdrant Integration**:
      - Checks if embeddings are already in Qdrant collections.
      - Uploads embeddings for new products to their respective collections.
  5. **Product Matching**: Uses Qdrant to find similar products from Dan Murphy's collection for each product in ALM.
  6. **Output**: Saves matches with similarity scores to a CSV file.
     
## Project Structure

```plaintext
embedding-product-matcher/
├── .env                 # Environment configuration file
├── main.py              # Main script for product matching
├── matchings/           # Folder for storing match result CSV files
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
