import os
import sqlite3
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

DB_NAME = "tickets_knowledge_base.db"
CSV_PATH_KAGGLE = "downloaded_datasets/all_tickets_processed_improved_v3.csv"
OUTPUT_FILE = "it_tickets_dataset.json"

def augment_data_all():
    if not os.path.exists(DB_NAME):
        print(f"Error: Database {DB_NAME} not found. Please run ingest_dataset.py first.")
        return

    print("Loading seed tickets from SQLite...")
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
            "category": cat,
            "description": desc,
            "resolution": res,
            "script": script
        })
        
    all_seeds = [s for cat_seeds in seeds_by_cat.values() for s in cat_seeds]
    print(f"Loaded {len(all_seeds)} seed tickets across {len(seeds_by_cat)} categories.")

    raw_tickets = []

    # --- 1. Load Console-AI HuggingFace Dataset (500 tickets) ---
    print("\nLoading HuggingFace dataset: Console-AI/IT-helpdesk-synthetic-tickets...")
    try:
        ds_console = load_dataset('Console-AI/IT-helpdesk-synthetic-tickets', split='train')
        for item in ds_console:
            desc = item.get("description", "") or item.get("subject", "")
            cat_raw = item.get("category", "").lower()
            
            # Map raw category
            mapped_cat = "Software & OS"
            if "network" in cat_raw or "internet" in cat_raw:
                mapped_cat = "Network & Internet"
            elif "hardware" in cat_raw or "printer" in cat_raw or "device" in cat_raw:
                mapped_cat = "Hardware & Peripherals"
            elif "access" in cat_raw or "security" in cat_raw or "login" in cat_raw or "password" in cat_raw:
                mapped_cat = "Access & Security"
                
            raw_tickets.append({"description": desc, "category": mapped_cat, "source": "Console-AI"})
        print(f"Loaded {len(ds_console)} tickets from Console-AI.")
    except Exception as e:
        print(f"Warning: Could not load Console-AI dataset: {e}")

    # --- 2. Load 6StringNinja HuggingFace Dataset (500 tickets) ---
    print("\nLoading HuggingFace dataset: 6StringNinja/synthetic-servicenow-incidents...")
    try:
        ds_servicenow = load_dataset('6StringNinja/synthetic-servicenow-incidents', split='train')
        for item in ds_servicenow:
            desc = item.get("description", "") or item.get("short_description", "")
            cat_raw = str(item.get("category", "")).lower()
            
            mapped_cat = "Software & OS"
            if "network" in cat_raw:
                mapped_cat = "Network & Internet"
            elif "hardware" in cat_raw or "printer" in cat_raw:
                mapped_cat = "Hardware & Peripherals"
            elif "access" in cat_raw or "security" in cat_raw or "login" in cat_raw:
                mapped_cat = "Access & Security"
                
            raw_tickets.append({"description": desc, "category": mapped_cat, "source": "ServiceNow"})
        print(f"Loaded {len(ds_servicenow)} tickets from 6StringNinja.")
    except Exception as e:
        print(f"Warning: Could not load 6StringNinja dataset: {e}")

    # --- 3. Load Kaggle Dataset (adisongoh/it-service-ticket-classification-dataset) ---
    if os.path.exists(CSV_PATH_KAGGLE):
        print(f"\nLoading and mapping Kaggle CSV dataset: {CSV_PATH_KAGGLE}...")
        try:
            df = pd.read_csv(CSV_PATH_KAGGLE, nrows=3000, on_bad_lines='skip', engine='python')
            kaggle_count = 0
            for _, row in df.iterrows():
                doc = str(row["Document"]).strip()
                topic = str(row["Topic_group"]).strip()
                
                if not doc or len(doc) < 15:
                    continue
                    
                mapped_cat = None
                if topic == "Hardware":
                    mapped_cat = "Hardware & Peripherals"
                elif topic in ["Access", "Administrative rights"]:
                    mapped_cat = "Access & Security"
                elif topic == "Storage":
                    mapped_cat = "Network & Internet"
                else:
                    doc_lower = doc.lower()
                    if any(w in doc_lower for w in ["vpn", "wifi", "wi-fi", "internet", "dns", "connection", "router"]):
                        mapped_cat = "Network & Internet"
                    elif any(w in doc_lower for w in ["excel", "outlook", "windows", "blue screen", "crash", "freeze"]):
                        mapped_cat = "Software & OS"
                        
                if mapped_cat and kaggle_count < 200:
                    raw_tickets.append({"description": doc, "category": mapped_cat, "source": "Kaggle"})
                    kaggle_count += 1
            print(f"Loaded {kaggle_count} tickets from Kaggle.")
        except Exception as e:
            print(f"Warning: Could not read Kaggle CSV: {e}")

    # --- PART 4: Perform TF-IDF Mapping & Generate Prompts ---
    print(f"\nTotal raw tickets compiled: {len(raw_tickets)}")
    if not raw_tickets:
        print("No raw tickets loaded. Aborting.")
        return

    augmented_dataset = []
    system_prompt = (
        "You are an expert IT Helpdesk and Systems Engineer. "
        "Classify the issue, provide troubleshooting steps, and write a recovery script. "
        "Respond in professional Vietnamese. Keep script code in English but comments in Vietnamese."
    )

    print("Generating training dataset using TF-IDF similarity matcher...")
    
    # Pre-train vectorizers for each category to match within category
    category_vectorizers = {}
    for cat, seeds in seeds_by_cat.items():
        seed_descs = [s["description"] for s in seeds]
        vec = TfidfVectorizer(stop_words='english')
        tfidf_seeds = vec.fit_transform(seed_descs)
        category_vectorizers[cat] = (vec, tfidf_seeds, seeds)

    for i, item in enumerate(raw_tickets):
        desc = item["description"]
        cat = item["category"]
        
        if cat not in category_vectorizers:
            # Fallback to general category matching using global tf-idf
            cat = "Software & OS"
            
        vec, tfidf_seeds, seeds = category_vectorizers[cat]
        
        # Match using TF-IDF cosine similarity
        query_vector = vec.transform([desc])
        similarities = cosine_similarity(query_vector, tfidf_seeds).flatten()
        best_idx = similarities.argmax()
        matched_seed = seeds[best_idx]
        
        # Construct response
        gpt_response = (
            f"Danh mục: {cat}\n\n"
            f"Giải pháp đề xuất:\n{matched_seed['resolution']}\n\n"
            f"Kịch bản khắc phục sự cố:\n"
            f"```powershell\n{matched_seed['script']}\n```"
        )
        
        # Add training chat format
        convo = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": f"Sự cố hệ thống: {desc}"},
                {"from": "gpt", "value": gpt_response}
            ]
        }
        augmented_dataset.append(convo)

    # Include original seeds
    print("Appending original seeds as baselines...")
    for seed in all_seeds:
        gpt_response = (
            f"Danh mục: {seed['category']}\n\n"
            f"Giải pháp đề xuất:\n{seed['resolution']}\n\n"
            f"Kịch bản khắc phục sự cố:\n"
            f"```powershell\n{seed['script']}\n```"
        )
        convo = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": f"Sự cố hệ thống: {seed['description']}"},
                {"from": "gpt", "value": gpt_response}
            ]
        }
        augmented_dataset.append(convo)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(augmented_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Success! Output dataset written to {OUTPUT_FILE} with {len(augmented_dataset)} samples.")

if __name__ == "__main__":
    augment_data_all()
