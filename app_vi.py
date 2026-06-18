import os
import sqlite3
import urllib.request
import json
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Cấu hình trang
st.set_page_config(
    page_title="Trợ lý Hỗ trợ IT AI Cục bộ (Ollama & RAG)",
    page_icon="🦅",
    layout="wide"
)

# Custom CSS cho giao diện kính mờ (Glassmorphic) cao cấp
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background and glassmorphism styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Bảng thông báo tiêu đề */
    .header-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .header-banner h1 {
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 10px;
        color: white;
    }
    .header-banner p {
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Thẻ kính mờ */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 35px 0 rgba(31, 38, 135, 0.08);
    }
    
    /* Nhãn phân loại */
    .badge {
        padding: 6px 14px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-network { background: linear-gradient(135deg, #0d6efd, #0a58ca); }
    .badge-hardware { background: linear-gradient(135deg, #fd7e14, #d95f02); }
    .badge-software { background: linear-gradient(135deg, #198754, #146c43); }
    .badge-security { background: linear-gradient(135deg, #dc3545, #b02a37); }
    .badge-general { background: linear-gradient(135deg, #6c757d, #495057); }
    
    /* Đèn trạng thái */
    .status-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 15px;
    }
    .status-dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-online { background-color: #198754; box-shadow: 0 0 8px #198754; }
    .dot-offline { background-color: #dc3545; box-shadow: 0 0 8px #dc3545; }
    
    .status-online { color: #198754; font-weight: bold; }
    .status-offline { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

DB_NAME = "tickets_knowledge_base.db"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Khởi tạo CSDL
if not os.path.exists(DB_NAME):
    from ingest_dataset import init_db, populate_seed_data
    init_db()
    populate_seed_data()
else:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
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

# Huấn luyện mô hình dự phòng Naive Bayes
@st.cache_resource
def train_fallback_model():
    conn = sqlite3.connect(DB_NAME)
    df_db = pd.read_sql_query("SELECT description, category FROM tickets", conn)
    conn.close()
    
    if len(df_db) == 0:
        return None
        
    model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words='english'), MultinomialNB())
    model.fit(df_db["description"], df_db["category"])
    return model

fallback_classifier = train_fallback_model()

# Các giải pháp dự phòng có sẵn
fallback_solutions = {
    "Network & Internet": {
        "badge": "badge-network",
        "priority": "Trung bình",
        "checklist": ["Kiểm tra kết nối cáp mạng vật lý", "Xóa bộ nhớ đệm DNS local (Flush DNS)", "Cấp phát lại địa chỉ IP từ DHCP"],
        "script": "Clear-DnsClientCache\nipconfig /release\nipconfig /renew"
    },
    "Hardware & Peripherals": {
        "badge": "badge-hardware",
        "priority": "Thấp",
        "checklist": ["Kiểm tra kết nối cáp USB/HDMI vật lý", "Kiểm tra nguồn điện thiết bị", "Khởi động lại tiến trình Spooler quản lý thiết bị"],
        "script": "Restart-Service -Name Spooler -Force\npnputil /scan-devices"
    },
    "Software & OS": {
        "badge": "badge-software",
        "priority": "Trung bình",
        "checklist": ["Kiểm tra hiệu suất CPU/RAM trong Task Manager", "Khởi động lại hệ điều hành", "Quét tính toàn vẹn hệ thống tệp tin"],
        "script": "sfc /scannow\nDISM /Online /Cleanup-Image /RestoreHealth"
    },
    "Access & Security": {
        "badge": "badge-security",
        "priority": "Cao",
        "checklist": ["Kiểm tra phím Caps Lock", "Xác nhận kết nối mạng miền doanh nghiệp", "Kiểm tra trạng thái khóa tài khoản Active Directory"],
        "script": "net user $env:USERNAME"
    },
    "General": {
        "badge": "badge-general",
        "priority": "Thấp",
        "checklist": ["Kiểm tra cấu hình cơ bản", "Thực hiện khởi động lại hệ thống vật lý"],
        "script": "# Chạy chẩn đoán cơ bản\nWrite-Host 'Running basic system diagnostics...'"
    }
}

# Từ điển Tiếng Việt
t = {
    "title": "Trợ lý IT AI Cục bộ (Ollama & RAG)",
    "desc": "Gửi sự cố của người dùng để truy vấn Cơ sở Tri thức SQLite cục bộ (RAG) và tạo các giải pháp tự động bằng mô hình AI chạy offline.",
    "sidebar_status": "📡 Trạng thái AI Local",
    "sidebar_db": "📁 Thông tin Cơ sở Tri thức",
    "stored_records": "Số lượng vé trong CSDL: **{}** bản ghi",
    "status_online": "ĐANG HOẠT ĐỘNG",
    "status_offline": "NGOẠI TUYẾN",
    "select_model": "Chọn mô hình AI cục bộ:",
    "fallback_info": "⚠️ Ollama không chạy trên localhost:11434. Ứng dụng sẽ chạy ở **Chế độ phân loại dự phòng**.",
    "issue_label": "Mô tả sự cố IT cần khắc phục:",
    "issue_placeholder": "Ví dụ: Máy khách VPN bị ngắt kết nối và không thể kết nối lại với thư mục của bộ phận.",
    "btn_submit": "Gửi & Giải quyết Sự cố",
    "err_empty": "Vui lòng nhập mô tả chi tiết của sự cố.",
    "rag_header": "📚 Các trường hợp tương tự trong CSDL (Ngữ cảnh RAG)",
    "no_rag_matches": "Không tìm thấy trường hợp tương tự trong CSDL tri thức. LLM sẽ giải quyết sự cố dựa trên tri thức cơ bản.",
    "report_header": "🤖 Báo cáo Chẩn đoán & Giải quyết từ AI",
    "success_llm": "Đã tạo giải pháp thành công bằng AI Local!",
    "err_llm": "Lỗi truy vấn AI Local: {}. Đang chuyển sang mô hình dự phòng.",
    "fallback_exec": "Đang chạy Mô hình dự phòng Naive Bayes cục bộ...",
    "category_label": "Danh mục",
    "priority_label": "Mức độ ưu tiên",
    "checklist_header": "🔍 Danh sách kiểm tra khắc phục sự cố",
    "script_header": "⚡ Kịch bản phục hồi tự động",
    "triage_success": "Hoàn tất chẩn đoán sơ bộ bằng mô hình sao lưu cục bộ.",
    "tab_diagnostics": "🩺 Bảng điều khiển Chẩn đoán",
    "tab_admin": "🗃️ Quản trị Cơ sở Tri thức",
    "tab_history": "⏳ Lịch sử & Yêu cầu Cũ",
    "log_upload_label": "🔌 Tải lên nhật ký hệ thống log (Tùy chọn):",
    "log_upload_help": "Tải lên file log từ Dự án 12 để tự động phát hiện lỗi.",
    "log_detect_success": "Đã tìm thấy {} log nghiêm trọng. Đã điền sẵn vào hộp mô tả!",
    "db_manager_title": "🗃️ Trình quản lý bản ghi Cơ sở Tri thức",
    "search_kb": "Tìm kiếm trong CSDL:",
    "add_kb_header": "➕ Thêm vé Cơ sở Tri thức Mới",
    "add_kb_desc": "Mô tả sự cố:",
    "add_kb_cat": "Danh mục:",
    "add_kb_res": "Giải pháp khắc phục:",
    "add_kb_script": "Kịch bản lệnh PowerShell/Bash:",
    "add_kb_btn": "Thêm Vé vào Cơ sở Dữ liệu",
    "add_kb_success": "Đã thêm vé thành công!",
    "delete_kb_btn": "Xóa ID vé",
    "delete_kb_success": "Đã xóa vé thành công!",
    "export_report": "📥 Tải Báo cáo (Markdown)",
    "export_script": "💾 Tải Kịch bản (.ps1)",
    "history_empty": "Chưa có lịch sử yêu cầu nào được lưu.",
    "history_title": "⏳ Lịch sử chẩn đoán đã lưu",
    "ollama_models_found": "Mô hình Local: {}",
}

# Kiểm tra kết nối Ollama
def check_ollama():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []

# Tìm kiếm vé tương đồng (RAG)
def retrieve_similar_tickets(query, num_results=3):
    conn = sqlite3.connect(DB_NAME)
    tickets = pd.read_sql_query("SELECT id, description, category, resolution, script FROM tickets", conn)
    conn.close()
    
    if len(tickets) == 0:
        return []
        
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(tickets["description"])
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-num_results:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            results.append(tickets.iloc[idx].to_dict())
    return results

# Gửi yêu cầu tới API Ollama
def query_local_llm(model_name, prompt):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        OLLAMA_URL, 
        data=data, 
        headers={'Content-Type': 'application/json'},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        res_data = json.loads(response.read().decode())
        return res_data.get("response", "")

# Thiết kế Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/System-Ollama_RAG_Helpdesk-blue?style=for-the-badge&logo=windows", use_container_width=True)
    st.markdown(f"### {t['sidebar_status']}")
    
    ollama_online, installed_models = check_ollama()
    if ollama_online:
        st.markdown(
            f'<div class="status-container"><span class="status-dot dot-online"></span>Ollama: <span class="status-online">{t["status_online"]}</span></div>', 
            unsafe_allow_html=True
        )
        model_selection = st.selectbox(t["select_model"], installed_models if installed_models else ["llama3.1"])
        st.caption(t["ollama_models_found"].format(", ".join(installed_models)))
    else:
        st.markdown(
            f'<div class="status-container"><span class="status-dot dot-offline"></span>Ollama: <span class="status-offline">{t["status_offline"]}</span></div>', 
            unsafe_allow_html=True
        )
        st.info(t["fallback_info"])
        model_selection = None

    st.markdown("---")
    st.markdown(f"### {t['sidebar_db']}")
    conn = sqlite3.connect(DB_NAME)
    db_count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    conn.close()
    st.write(t["stored_records"].format(db_count))

# Phân tách Tab
tab1, tab2, tab3 = st.tabs([t["tab_diagnostics"], t["tab_admin"], t["tab_history"]])

# --- Tab 1: Bảng chẩn đoán chính ---
with tab1:
    st.markdown(f"""
    <div class="header-banner">
        <h1>{t["title"]}</h1>
        <p>{t["desc"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload File Log
    st.markdown(f"##### {t['log_upload_label']}")
    uploaded_file = st.file_uploader(t["log_upload_label"], type=["csv", "txt", "log"], label_visibility="collapsed", help=t["log_upload_help"])
    
    log_context = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_log = pd.read_csv(uploaded_file)
                anomalies = []
                if "is_anomaly" in df_log.columns:
                    anomalies = df_log[df_log["is_anomaly"] == 1]
                elif "level" in df_log.columns:
                    anomalies = df_log[df_log["level"].isin(["ERROR", "CRITICAL", "ANOMALY"])]
                
                if len(anomalies) > 0:
                    log_context = f"Phát hiện nhật ký lỗi hệ thống:\n"
                    for _, row in anomalies.head(5).iterrows():
                        log_context += f"- [{row.get('level', 'ERROR')}] Dịch vụ: {row.get('service', 'N/A')}, thông báo: {row.get('message', 'N/A')}, mã trạng thái: {row.get('status_code', 'N/A')}\n"
                    st.success(t["log_detect_success"].format(len(anomalies)))
                else:
                    log_context = f"Nội dung log: " + ", ".join(df_log["message"].head(3).tolist())
            else:
                lines = uploaded_file.read().decode("utf-8").splitlines()
                errs = [l for l in lines if any(w in l.upper() for w in ["ERROR", "FAILED", "CRITICAL", "TIMEOUT", "EXCEPTION"])]
                if errs:
                    log_context = "Nhật ký dòng lỗi:\n" + "\n".join(errs[:5])
                    st.success(t["log_detect_success"].format(len(errs[:5])))
                else:
                    log_context = "Không phát hiện lỗi trực tiếp trong tệp tin log dạng text."
        except Exception as e:
            st.error(f"Lỗi phân tích file log: {e}")
            
    default_text = ""
    if log_context:
        default_text = f"Phân tích lỗi log hệ thống:\n{log_context}"
        
    user_issue = st.text_area(
        t["issue_label"],
        value=default_text,
        placeholder=t["issue_placeholder"],
        height=150
    )
    
    if st.button(t["btn_submit"], type="primary"):
        if not user_issue.strip():
            st.warning(t["err_empty"])
        else:
            # Ngữ cảnh RAG
            st.markdown(f"### {t['rag_header']}")
            similar_tickets = retrieve_similar_tickets(user_issue, 3)
            
            if similar_tickets:
                for i, ticket in enumerate(similar_tickets):
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>Vé tương tự #{i+1}: <span class="badge badge-{ticket['category'].lower().replace(' & ', '-').replace(' ', '-')}">{ticket['category']}</span></h4>
                        <p><b>Mô tả:</b> {ticket['description']}</p>
                        <p><b>Giải pháp:</b> {ticket['resolution']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander(f"Xem kịch bản lệnh"):
                        st.code(ticket["script"], language="powershell")
            else:
                st.info(t["no_rag_matches"])
                
            # Tạo báo cáo từ AI
            st.markdown(f"### {t['report_header']}")
            
            generated_report = ""
            generated_script = ""
            predicted_category = "General"
            suggested_priority = "Medium"
            
            if ollama_online and model_selection:
                context_str = ""
                for i, ticket in enumerate(similar_tickets):
                    context_str += f"Ticket {i+1}:\n- Description: {ticket['description']}\n- Category: {ticket['category']}\n- Resolution: {ticket['resolution']}\n- Script:\n{ticket['script']}\n\n"
                
                prompt = (
                    "You are an expert IT Systems Engineer and Level 2 Helpdesk support specialist.\n"
                    "Review the following user issue and similar past tickets to formulate a solution.\n\n"
                    f"### Similar Resolved Tickets (Context):\n{context_str}\n"
                    f"### User Issue to Resolve:\n{user_issue}\n\n"
                    "### Output requirements:\n"
                    "1. Predicted Category (Network & Internet, Hardware & Peripherals, Software & OS, or Access & Security)\n"
                    "2. Suggested Priority Level (Low, Medium, High)\n"
                    "3. Step-by-step troubleshooting checklist\n"
                    "4. A clean, executable PowerShell or Bash script block to diagnostic/fix the issue.\n\n"
                    "Respond in clear, professional Vietnamese. The PowerShell or Bash scripts must remain in English but comments can be in Vietnamese."
                )
                
                with st.spinner(f"Đang xử lý bằng {model_selection}..."):
                    try:
                        response_text = query_local_llm(model_selection, prompt)
                        st.markdown(response_text)
                        st.success(t["success_llm"])
                        
                        generated_report = response_text
                        if "```powershell" in response_text:
                            generated_script = response_text.split("```powershell")[1].split("```")[0].strip()
                        elif "```bash" in response_text:
                            generated_script = response_text.split("```bash")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            generated_script = response_text.split("```")[1].split("```")[0].strip()
                        else:
                            generated_script = "# Diagnostics"
                            
                        for cat in ["Network", "Hardware", "OS", "Software", "Security"]:
                            if cat.lower() in response_text.lower():
                                if cat == "Network": predicted_category = "Network & Internet"
                                elif cat == "Hardware": predicted_category = "Hardware & Peripherals"
                                elif cat == "Security": predicted_category = "Access & Security"
                                else: predicted_category = "Software & OS"
                                break
                    except Exception as e:
                        st.error(t["err_llm"].format(e))
                        ollama_online = False
                        
            if not ollama_online or not model_selection:
                st.info(t["fallback_exec"])
                if fallback_classifier:
                    prediction = fallback_classifier.predict([user_issue])[0]
                else:
                    prediction = "General"
                    
                sol = fallback_solutions.get(prediction, fallback_solutions["General"])
                predicted_category = prediction
                suggested_priority = sol["priority"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{t['category_label']}:** <span class='badge {sol['badge']}'>{prediction}</span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{t['priority_label']}:** **{sol['priority']}**")
                    
                st.markdown(f"### {t['checklist_header']}")
                for step in sol["checklist"]:
                    st.checkbox(step)
                    
                st.markdown(f"### {t['script_header']}")
                st.code(sol["script"], language="powershell")
                
                generated_report = f"Danh mục: {prediction}\nMức độ ưu tiên: {sol['priority']}\n\nCác bước khắc phục sự cố:\n" + "\n".join([f"- {step}" for step in sol["checklist"]])
                generated_script = sol["script"]
                st.success(t["triage_success"])
                
            # Ghi lịch sử vào CSDL
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO ticket_history (user_issue, predicted_category, priority, diagnostic_report, recovery_script) VALUES (?, ?, ?, ?, ?)",
                    (user_issue, predicted_category, suggested_priority, generated_report, generated_script)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                st.warning(f"Không thể ghi lịch sử chẩn đoán: {e}")
                
            # Các nút Tải file giải pháp
            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label=t["export_report"],
                    data=generated_report,
                    file_name="Bao_Cao_Chan_Doan_IT.md",
                    mime="text/markdown"
                )
            with col_d2:
                st.download_button(
                    label=t["export_script"],
                    data=generated_script,
                    file_name="Kich_Ban_Phuc_Hoi.ps1",
                    mime="text/plain"
                )

# --- Tab 2: Quản trị Cơ sở Dữ liệu ---
with tab2:
    st.markdown(f"### {t['db_manager_title']}")
    
    search_query = st.text_input(t["search_kb"], placeholder="Tìm kiếm theo mô tả hoặc danh mục...")
    
    conn = sqlite3.connect(DB_NAME)
    if search_query:
        df_kb = pd.read_sql_query(
            "SELECT id, category, description, resolution FROM tickets WHERE description LIKE ? OR category LIKE ?",
            conn,
            params=(f"%{search_query}%", f"%{search_query}%")
        )
    else:
        df_kb = pd.read_sql_query("SELECT id, category, description, resolution FROM tickets", conn)
    conn.close()
    
    st.dataframe(df_kb, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"#### {t['add_kb_header']}")
    with st.form("add_ticket_form_vi", clear_on_submit=True):
        new_cat = st.selectbox(t["add_kb_cat"], ["Network & Internet", "Hardware & Peripherals", "Software & OS", "Access & Security", "General"])
        new_desc = st.text_area(t["add_kb_desc"])
        new_res = st.text_area(t["add_kb_res"])
        new_script = st.text_area(t["add_kb_script"], value="# Script logic")
        
        submit_new = st.form_submit_button(t["add_kb_btn"])
        if submit_new:
            if not new_desc.strip():
                st.error("Mô tả sự cố là bắt buộc.")
            else:
                conn = sqlite3.connect(DB_NAME)
                conn.execute(
                    "INSERT INTO tickets (description, category, resolution, script) VALUES (?, ?, ?, ?)",
                    (new_desc, new_cat, new_res, new_script)
                )
                conn.commit()
                conn.close()
                st.success(t["add_kb_success"])
                st.rerun()
                
    st.markdown("---")
    col_del1, col_del2 = st.columns([1, 3])
    with col_del1:
        del_id = st.number_input(t["delete_kb_btn"], min_value=1, step=1)
    with col_del2:
        st.write("")
        st.write("")
        if st.button(t["delete_kb_btn"], type="secondary"):
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tickets WHERE id = ?", (del_id,))
            exists = cursor.fetchone()[0]
            if exists > 0:
                cursor.execute("DELETE FROM tickets WHERE id = ?", (del_id,))
                conn.commit()
                st.success(t["delete_kb_success"])
                conn.close()
                st.rerun()
            else:
                st.error("ID vé không tồn tại.")
                conn.close()

# --- Tab 3: Lịch sử yêu cầu ---
with tab3:
    st.markdown(f"### {t['history_title']}")
    
    conn = sqlite3.connect(DB_NAME)
    df_hist = pd.read_sql_query("SELECT id, timestamp, user_issue, predicted_category, priority FROM ticket_history ORDER BY id DESC", conn)
    conn.close()
    
    if len(df_hist) == 0:
        st.info(t["history_empty"])
    else:
        for idx, row in df_hist.iterrows():
            with st.expander(f"[{row['timestamp']}] {row['predicted_category']} - Mức độ ưu tiên: {row['priority']}"):
                st.write(f"**Mô tả sự cố:** {row['user_issue']}")
                
                conn = sqlite3.connect(DB_NAME)
                full_row = conn.execute("SELECT diagnostic_report, recovery_script FROM ticket_history WHERE id = ?", (int(row['id']),)).fetchone()
                conn.close()
                
                if full_row:
                    st.markdown("#### Báo cáo chẩn đoán:")
                    st.markdown(full_row[0])
                    st.markdown("#### Kịch bản tự động:")
                    st.code(full_row[1], language="powershell")
                    
                    st.download_button(
                        label=f"{t['export_report']} ({row['id']})",
                        data=full_row[0],
                        file_name=f"IT_Report_{row['id']}.md",
                        mime="text/markdown",
                        key=f"dl_rep_vi_{row['id']}"
                    )
