import sqlite3
import json
import os

DB_NAME = "tickets_knowledge_base.db"
OUTPUT_FILE = "it_tickets_dataset.json"

def export_data():
    if not os.path.exists(DB_NAME):
        print(f"Error: Database {DB_NAME} not found. Please run ingest_dataset.py first.")
        return
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT category, description, resolution, script FROM tickets")
    rows = cursor.fetchall()
    conn.close()
    
    dataset = []
    
    # System instruction for the model
    system_prompt = (
        "You are an expert IT Helpdesk and Systems Engineer. "
        "Classify the issue, provide troubleshooting steps, and write a recovery script. "
        "Respond in professional Vietnamese. Keep script code in English but comments in Vietnamese."
    )
    
    for row in rows:
        category, description, resolution, script = row
        
        # Format the response in a structured way
        gpt_response = (
            f"Danh mục: {category}\n\n"
            f"Giải pháp đề xuất:\n{resolution}\n\n"
            f"Kịch bản khắc phục sự cố:\n"
            f"```powershell\n{script}\n```"
        )
        
        # Unsloth/ShareGPT chat format
        item = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": f"Sự cố hệ thống: {description}"},
                {"from": "gpt", "value": gpt_response}
            ]
        }
        dataset.append(item)
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully exported {len(dataset)} tickets to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_data()
