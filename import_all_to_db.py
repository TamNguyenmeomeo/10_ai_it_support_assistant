import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

DB_NAME = "tickets_knowledge_base.db"
CSV_PATH_KAGGLE = "downloaded_datasets/all_tickets_processed_improved_v3.csv"

# Seed tickets
SEED_TICKETS = [
    # Network Issues
    ("Cannot connect to office Wi-Fi network on my laptop.", 
     "Network & Internet", 
     "Verify SSID name, toggle Wi-Fi adapter, forget the network and reconnect with corporate credentials.", 
     "netsh wlan show interfaces\ninterface show interface"),
    
    ("Internet connection is extremely slow, web pages taking minutes to load.", 
     "Network & Internet", 
     "Clear browser cache, flush local DNS, release and renew IP address from DHCP server.", 
     "ipconfig /flushdns\nipconfig /release\nipconfig /renew"),
    
    ("Cannot access the shared department network drive (Z:).", 
     "Network & Internet", 
     "Check VPN status if remote, verify credential manager permissions, attempt manual mapping via net use.", 
     "net use Z: \\\\fileserver\\department /persistent:yes"),
    
    ("VPN client disconnected and refuses to reconnect with error 800.", 
     "Network & Internet", 
     "Check internet connection, verify VPN server IP, restart the routing and remote access service.", 
     "Restart-Service -Name RasMan -Force"),

    # Hardware Issues
    ("Office printer is offline and not responding to print jobs.", 
     "Hardware & Peripherals", 
     "Check printer power, verify spooler service is running on the computer, delete stuck queue files.", 
     "Restart-Service -Name Spooler -Force"),
    
    ("External monitor is black and screen won't turn on or detect input.", 
     "Hardware & Peripherals", 
     "Inspect HDMI/DisplayPort cable connection, verify input source, reload graphics drivers.", 
     "pnputil /scan-devices"),
    
    ("Laptop is overheating and making loud fan noises.", 
     "Hardware & Peripherals", 
     "Clean dust from vents, check task manager for high CPU processes and terminate them.", 
     "Get-Process | Sort-Object CPU -Descending | Select-Object -First 5"),

    # Software Issues
    ("Windows displays Blue Screen of Death (BSOD) after update.", 
     "Software & OS", 
     "Boot in Safe Mode, perform a System File Checker scan, restore the system image.", 
     "sfc /scannow\nDISM /Online /Cleanup-Image /RestoreHealth"),
    
    ("Microsoft Excel crashes or freezes every time I open a large spreadsheet.", 
     "Software & OS", 
     "Disable Excel add-ins, run Office repair utility, clear temporary files.", 
     "Remove-Item -Path $env:TEMP\\* -Recurse -Force -ErrorAction SilentlyContinue"),
    
    ("Outlook client is stuck on loading profile screen on startup.", 
     "Software & OS", 
     "Open Outlook in safe mode, repair email profile, disable hardware graphics acceleration.", 
     "outlook.exe /safe"),

    # Access & Security
    ("Forgot my login password and account is locked out of active directory.", 
     "Access & Security", 
     "Verify domain connectivity, unlock account via active directory domain controller.", 
     "Unlock-ADAccount -Identity $env:USERNAME"),
    
    ("Access denied permission error when opening project shared folder.", 
     "Access & Security", 
     "Verify security permissions group membership, clear stored credentials in Credential Manager.", 
     "cmdkey /list"),
    
    ("Received a suspicious phishing email looking link in my inbox.", 
     "Access & Security", 
     "Do not click links. Report the email to the security operations center (SOC) and delete the mail.", 
     "Write-Host 'Phishing report generated.' -ForegroundColor Red")
]

def main():
    print("Preparing DB connection...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Recreate tables to be clean
    cursor.execute("DROP TABLE IF EXISTS tickets")
    cursor.execute("""
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            category TEXT,
            resolution TEXT,
            script TEXT
        )
    """)
    conn.commit()

    # Load and categorize seed tickets
    seeds_by_cat = {}
    for desc, cat, res, script in SEED_TICKETS:
        if cat not in seeds_by_cat:
            seeds_by_cat[cat] = []
        seeds_by_cat[cat].append({
            "category": cat,
            "description": desc,
            "resolution": res,
            "script": script
        })

    # Group all raw tickets to match
    raw_tickets = []

    # 1. HuggingFace: Console-AI
    print("Loading Console-AI/IT-helpdesk-synthetic-tickets...")
    try:
        ds = load_dataset('Console-AI/IT-helpdesk-synthetic-tickets', split='train')
        for item in ds:
            desc = item.get("description", "") or item.get("subject", "")
            cat_raw = str(item.get("category", "")).lower()
            mapped_cat = "Software & OS"
            if "network" in cat_raw or "internet" in cat_raw:
                mapped_cat = "Network & Internet"
            elif "hardware" in cat_raw or "printer" in cat_raw or "device" in cat_raw:
                mapped_cat = "Hardware & Peripherals"
            elif "access" in cat_raw or "security" in cat_raw or "login" in cat_raw or "password" in cat_raw:
                mapped_cat = "Access & Security"
            raw_tickets.append({"description": desc, "category": mapped_cat})
    except Exception as e:
        print(f"HF Console-AI error: {e}")

    # 2. HuggingFace: 6StringNinja
    print("Loading 6StringNinja/synthetic-servicenow-incidents...")
    try:
        ds = load_dataset('6StringNinja/synthetic-servicenow-incidents', split='train')
        for item in ds:
            desc = item.get("description", "") or item.get("short_description", "")
            cat_raw = str(item.get("category", "")).lower()
            mapped_cat = "Software & OS"
            if "network" in cat_raw:
                mapped_cat = "Network & Internet"
            elif "hardware" in cat_raw or "printer" in cat_raw:
                mapped_cat = "Hardware & Peripherals"
            elif "access" in cat_raw or "security" in cat_raw or "login" in cat_raw:
                mapped_cat = "Access & Security"
            raw_tickets.append({"description": desc, "category": mapped_cat})
    except Exception as e:
        print(f"HF ServiceNow error: {e}")

    # 3. Kaggle CSV
    if os.path.exists(CSV_PATH_KAGGLE):
        print("Loading Kaggle CSV dataset...")
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
                    raw_tickets.append({"description": doc, "category": mapped_cat})
                    kaggle_count += 1
        except Exception as e:
            print(f"Kaggle CSV error: {e}")

    print(f"Matching and ingesting {len(raw_tickets)} tickets...")
    
    # Pre-train vectorizers for each category
    category_vectorizers = {}
    for cat, seeds in seeds_by_cat.items():
        seed_descs = [s["description"] for s in seeds]
        vec = TfidfVectorizer(stop_words='english')
        tfidf_seeds = vec.fit_transform(seed_descs)
        category_vectorizers[cat] = (vec, tfidf_seeds, seeds)

    # Insert seeds first
    for desc, cat, res, script in SEED_TICKETS:
        cursor.execute(
            "INSERT INTO tickets (description, category, resolution, script) VALUES (?, ?, ?, ?)",
            (desc, cat, res, script)
        )

    # Match and insert raw tickets
    inserted_count = len(SEED_TICKETS)
    for item in raw_tickets:
        desc = item["description"]
        cat = item["category"]
        if cat not in category_vectorizers:
            cat = "Software & OS"
        
        vec, tfidf_seeds, seeds = category_vectorizers[cat]
        query_vector = vec.transform([desc])
        similarities = cosine_similarity(query_vector, tfidf_seeds).flatten()
        best_idx = similarities.argmax()
        matched_seed = seeds[best_idx]
        
        cursor.execute(
            "INSERT INTO tickets (description, category, resolution, script) VALUES (?, ?, ?, ?)",
            (desc, cat, matched_seed["resolution"], matched_seed["script"])
        )
        inserted_count += 1

    conn.commit()
    conn.close()
    print(f"Successfully populated database with {inserted_count} records!")

if __name__ == "__main__":
    main()
