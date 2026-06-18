import os
import sqlite3
import pandas as pd

DB_NAME = "tickets_knowledge_base.db"

# Rich set of seed data to populate database if no external CSV is present
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

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            category TEXT,
            resolution TEXT,
            script TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_issue TEXT NOT NULL,
            predicted_category TEXT,
            priority TEXT,
            diagnostic_report TEXT,
            recovery_script TEXT
        )
    """)
    conn.commit()
    conn.close()

def import_from_csv(csv_path):
    print(f"Importing tickets from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Try to find standard columns
    desc_col = None
    cat_col = None
    res_col = None
    script_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if "desc" in col_lower or "body" in col_lower or "text" in col_lower or "issue" in col_lower:
            desc_col = col
        elif "cat" in col_lower or "class" in col_lower or "type" in col_lower:
            cat_col = col
        elif "res" in col_lower or "sol" in col_lower or "fix" in col_lower:
            res_col = col
        elif "script" in col_lower or "cmd" in col_lower or "command" in col_lower:
            script_col = col
            
    if not desc_col:
        print("Could not find a valid description column. Skipping file.")
        return False
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    count = 0
    for _, row in df.iterrows():
        description = str(row[desc_col])
        category = str(row[cat_col]) if cat_col else "General"
        resolution = str(row[res_col]) if res_col else "Standard IT Support diagnosis required."
        script = str(row[script_col]) if script_col else "# Execute standard diagnostics"
        
        cursor.execute(
            "INSERT INTO tickets (description, category, resolution, script) VALUES (?, ?, ?, ?)",
            (description, category, resolution, script)
        )
        count += 1
        
    conn.commit()
    conn.close()
    print(f"Successfully imported {count} tickets into knowledge base.")
    return True

def populate_seed_data():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM tickets")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("Populating database with seed IT Support tickets...")
        for desc, cat, res, script in SEED_TICKETS:
            cursor.execute(
                "INSERT INTO tickets (description, category, resolution, script) VALUES (?, ?, ?, ?)",
                (desc, cat, res, script)
            )
        conn.commit()
        print(f"Database seeded with {len(SEED_TICKETS)} tickets.")
    else:
        print(f"Database already contains {count} records. Skipping seeding.")
        
    conn.close()

def main():
    init_db()
    
    # Check if there are any CSV files in the current folder to import
    csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
    
    imported = False
    for csv_file in csv_files:
        if csv_file != "system_logs.csv":  # Skip logs from Project 12
            imported = import_from_csv(csv_file)
            
    # If no CSV was imported, seed default data
    if not imported:
        populate_seed_data()
        
if __name__ == "__main__":
    main()
