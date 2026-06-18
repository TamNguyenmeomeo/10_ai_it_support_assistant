import os
import sqlite3
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_NAME = "tickets_knowledge_base.db"
CSV_PATH = "downloaded_datasets/all_tickets_processed_improved_v3.csv"
OUTPUT_FILE = "it_tickets_dataset.json"

def augment_data():
    if not os.path.exists(DB_NAME):
        print(f"Error: Database {DB_NAME} not found. Please run ingest_dataset.py first.")
        return
        
    if not os.path.exists(CSV_PATH):
        print(f"Error: Kaggle CSV file {CSV_PATH} not found. Downloading now...")
        # Try to download if missing
        os.system("kaggle datasets download -d adisongoh/it-service-ticket-classification-dataset --unzip -p downloaded_datasets")
        if not os.path.exists(CSV_PATH):
            print("Failed to download or extract the dataset.")
            return

    print("Loading seed tickets from CSDL SQLite...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT category, description, resolution, script FROM tickets")
    seed_rows = cursor.fetchall()
    conn.close()
    
    # Group seed tickets by category
    seeds_by_cat = {}
    for cat, desc, res, script in seed_rows:
        if cat not in seeds_by_cat:
            seeds_by_cat[cat] = []
        seeds_by_cat[cat].append({
            "description": desc,
            "resolution": res,
            "script": script
        })
        
    print(f"Loaded {len(seed_rows)} seed tickets across {len(seeds_by_cat)} categories.")

    print("Reading first 10,000 rows from Kaggle CSV dataset...")
    df = pd.read_csv(CSV_PATH, nrows=10000, on_bad_lines='skip', engine='python')
    
    # Categories mapping
    # We will pick 100 random samples for each of our 4 target categories from the CSV
    target_samples = {
        "Network & Internet": [],
        "Hardware & Peripherals": [],
        "Software & OS": [],
        "Access & Security": []
    }
    
    print("Filtering and mapping Kaggle tickets...")
    # Group the CSV data to make it easier to extract
    for _, row in df.iterrows():
        doc = str(row["Document"]).strip()
        topic = str(row["Topic_group"]).strip()
        
        if not doc or len(doc) < 15:
            continue
            
        # Map Kaggle categories to our categories
        mapped_cat = None
        if topic == "Hardware":
            mapped_cat = "Hardware & Peripherals"
        elif topic in ["Access", "Administrative rights"]:
            mapped_cat = "Access & Security"
        elif topic == "Storage":
            mapped_cat = "Network & Internet"
        else:
            # Fallback keyword matching for Network & Software
            doc_lower = doc.lower()
            if any(w in doc_lower for w in ["vpn", "wifi", "wi-fi", "internet", "dns", "connection", "router", "network"]):
                mapped_cat = "Network & Internet"
            elif any(w in doc_lower for w in ["excel", "outlook", "windows", "blue screen", "crash", "freeze", "bug", "office"]):
                mapped_cat = "Software & OS"
                
        if mapped_cat and len(target_samples[mapped_cat]) < 100:
            target_samples[mapped_cat].append(doc)
            
    # Combine all samples
    augmented_dataset = []
    
    system_prompt = (
        "You are an expert IT Helpdesk and Systems Engineer. "
        "Classify the issue, provide troubleshooting steps, and write a recovery script. "
        "Respond in professional Vietnamese. Keep script code in English but comments in Vietnamese."
    )
    
    print("Generating augmented responses using TF-IDF matching...")
    total_augmented = 0
    for cat, docs in target_samples.items():
        if cat not in seeds_by_cat:
            continue
            
        seeds = seeds_by_cat[cat]
        seed_descriptions = [s["description"] for s in seeds]
        
        # Initialize TF-IDF Vectorizer for this category's seeds
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_seeds = vectorizer.fit_transform(seed_descriptions)
        
        for doc in docs:
            # Find the most similar seed ticket in this category
            query_vector = vectorizer.transform([doc])
            similarities = cosine_similarity(query_vector, tfidf_seeds).flatten()
            best_idx = similarities.argmax()
            matched_seed = seeds[best_idx]
            
            # Format the output using matched seed's resolution and script
            gpt_response = (
                f"Danh mục: {cat}\n\n"
                f"Giải pháp đề xuất:\n{matched_seed['resolution']}\n\n"
                f"Kịch bản khắc phục sự cố:\n"
                f"```powershell\n{matched_seed['script']}\n```"
            )
            
            # Add to augmented dataset
            item = {
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": f"Sự cố hệ thống: {doc}"},
                    {"from": "gpt", "value": gpt_response}
                ]
            }
            augmented_dataset.append(item)
            total_augmented += 1
            
    # Include the original seed tickets as well to preserve the baseline
    print("Adding original seed tickets to dataset...")
    for cat, seeds in seeds_by_cat.items():
        for seed in seeds:
            gpt_response = (
                f"Danh mục: {cat}\n\n"
                f"Giải pháp đề xuất:\n{seed['resolution']}\n\n"
                f"Kịch bản khắc phục sự cố:\n"
                f"```powershell\n{seed['script']}\n```"
            )
            item = {
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": f"Sự cố hệ thống: {seed['description']}"},
                    {"from": "gpt", "value": gpt_response}
                ]
            }
            augmented_dataset.append(item)
            total_augmented += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(augmented_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully created augmented dataset with {total_augmented} examples at {OUTPUT_FILE}!")

if __name__ == "__main__":
    augment_data()
