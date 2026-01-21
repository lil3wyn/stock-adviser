import streamlit as st
import time

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="TA Alex 2026", page_icon="💎")

# --- 2. KHO KEY VÔ HẠN (5 KEYS) ---
API_KEY_POOL = [
    "AIzaSyAcIDpmFgBVzIlb41m1cz4BPlTCjKM9Hl0",
    "AIzaSyBC_V9ACvGCElaWQL5BILKQCv_ikBGcsHs", 
    "AIzaSyCFgTf678MHOoaOMmfV6y0uXLVrT2VwPV8",
    "AIzaSyBJhszyVcCesLBHlL2mfEP3Tx-ykMyA4_w",
    "AIzaSyA9S1V66bDs9UrnnVJKy_zDbxWQh6MMxtM"
]

# --- 3. BẢO VỆ THƯ VIỆN ---
try:
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
except Exception as e:
    st.error(f"❌ Lỗi thư viện: {e}")
    st.stop()

# --- 4. HÀM AI XOAY TUA (BẤT TỬ) ---
def call_ai_rotation(prompt):
    models = ["gemini-3-flash-preview", "gemini-2.0-flash-exp"]
    for i, key in enumerate(API_KEY_POOL):
        for model_name in models:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)
                safety = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                response = model.generate_content(prompt, safety_settings=safety)
                if response.text:
                    return response.text, f"{model_name} (Key {i+1})"
            except: continue
    return "❌ Mạng nghẽn, AI chưa trả lời kịp.", "Error"

# --- 5. HÀM LẤY DATA (CHỐNG SẬP) ---
@st.cache_data(ttl=300)
def get_data_safe(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        if df is None or df.empty:
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            # Chỉ báo
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            return df
        return None
    except: return None

# --- 6. GIAO DIỆN CHÍNH ---
st.sidebar.title("💎 TA Alex 2026")
st.sidebar.success(f"✅ Đã nạp {len(API_KEY_POOL)} Key Vô Hạn")

tab1, tab2 = st.tabs(["📊 Phân Tích", "🚀 Scanner (Full)"])

# === TAB 1: PHÂN TÍCH ===
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="MBB").upper()
    
    if st.button("🔍 Phân Tích Ngay", type="primary"):
        status = st.status("🚀 Đang xử lý...", expanded=True)
        try:
            df = get_data_safe(symbol)
            if df is not None:
                last = df.iloc[-1]
                status.write("✅ Dữ liệu OK.")
                
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Giá", f"{last['close']:,.0f}")
                c2.metric("RSI", f"{last['RSI']:.1f}")
                c3.metric("MACD", f"{last['MACD']:.2f}")
                c4.metric("Vol", f"{last['volume']:,.0f}")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Giá"))
                fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], line=dict(color='orange'), name="MA20"))
                st.plotly_chart(fig, use_container_width=True)
                
                # AI
                status.write("🤖 Đang gọi Alex...")
                data_text = df.tail(30).to_string()
                prompt = f"Giá {symbol}: {last['close']}. Dữ liệu:\n{data_text}\n. Phân tích xu hướng ngắn gọn."
                ai_text, model_used = call_ai_rotation(prompt)
                
                st.info(f"💡 Nhận định ({model_used}):")
                st.write(ai_text)
                status.update(label="Hoàn tất!", state="complete", expanded=False)
            else:
                status.update(label="Lỗi mã!", state="error")
                st.error("Không tìm thấy mã này.")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# === TAB 2: SCANNER (ĐÃ SỬA LẠI FULL TÍNH NĂNG) ===
with tab2:
    st.header("🕵️ Quét Cổ Phiếu")
    
    # 1. Cho nhập list dài thoả thích
    default_list = "ACB, FPT, HPG, MBB, MSN, MWG, SSI, STB, TCB, VHM, VIC, VNM, VPB, DIG, CEO, DXG"
    scan_input = st.text_area("Nhập danh sách mã (cách nhau dấu phẩy):", value=default_list, height=100)
    
    if st.button("🚀 Quét Toàn Bộ"):
        # Tách chuỗi thành list
        symbols = [s.strip().upper() for s in scan_input.split(",") if s.strip()]
        results = []
        
        # Thanh tiến trình
        bar = st.progress(0)
        status_text = st.empty()
        
        for i, sym in enumerate(symbols):
            status_text.text(f"Đang soi: {sym}...")
            try:
                d = get_data_safe(sym)
                if d is not None:
                    r = d.iloc[-1]
                    
                    # --- Logic Chấm Điểm ---
                    score = 0
                    reasons = []
                    
                    # 1. Xu hướng
                    if r['close'] > r['MA20']: score += 1
                    if r['MA20'] > r['MA50']: score += 1; reasons.append("Uptrend")
                    
                    # 2. Động lượng
                    if r['MACD'] > r['Signal_Line']: score += 1.5; reasons.append("MACD cắt lên")
                    if 40 < r['RSI'] < 60: score += 0.5
                    
                    # Xếp loại
                    rank = "Yếu"
                    if score >= 3.5: rank = "🔥 Khỏe"
                    elif score >= 2: rank = "😐 Trung"
                    
                    results.append({
                        "Mã": sym,
                        "Giá": r['close'],
                        "Điểm": score,
                        "Xếp loại": rank,
                        "Lý do": ", ".join(reasons)
                    })
            except: pass
            
            # Cập nhật thanh tiến trình
            bar.progress((i + 1) / len(symbols))
            
        status_text.empty()
        bar.empty()
        
        # HIỂN THỊ KẾT QUẢ
        if results:
            df_res = pd.DataFrame(results).sort_values(by="Điểm", ascending=False)
            
            # Tô màu cho đẹp
            def highlight(val):
                if "Khỏe" in str(val): return 'background-color: #d4edda; color: black'
                return ''
            
            st.dataframe(df_res.style.applymap(highlight, subset=['Xếp loại']), use_container_width=True)
            
            # --- AI KHUYẾN NGHỊ (ĐÃ CÓ LẠI) ---
            top_stock = df_res.iloc[0]
            st.markdown("---")
            st.subheader(f"🏆 Alex chọn ngôi sao sáng nhất: {top_stock['Mã']}")
            
            with st.spinner("Alex đang viết bài phân tích chi tiết..."):
                prompt = f"""
                Dựa trên kết quả quét: {top_stock['Mã']} có điểm kỹ thuật cao nhất ({top_stock['Điểm']} điểm).
                Lý do kỹ thuật: {top_stock['Lý do']}.
                Giá hiện tại: {top_stock['Giá']}.
                Hãy viết một khuyến nghị MUA ngắn gọn, bao gồm điểm cắt lỗ và chốt lời dự kiến.
                """
                ai_reply, model_used = call_ai_rotation(prompt)
                st.write(ai_reply)
                st.caption(f"(Phân tích bởi {model_used})")
        else:
            st.warning("Không quét được mã nào. Kiểm tra lại danh sách nhé.")
