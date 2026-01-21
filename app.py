import streamlit as st
import time
import re # Thư viện xử lý văn bản để bắt mã cổ phiếu

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="TA Alex 2026 Pro", page_icon="💎")

# --- 2. KHO KEY VÔ HẠN (5 KEYS) ---
API_KEY_POOL = [
    "AIzaSyAcIDpmFgBVzIlb41m1cz4BPlTCjKM9Hl0",
    "AIzaSyBC_V9ACvGCElaWQL5BILKQCv_ikBGcsHs", 
    "AIzaSyCFgTf678MHOoaOMmfV6y0uXLVrT2VwPV8",
    "AIzaSyBJhszyVcCesLBHlL2mfEP3Tx-ykMyA4_w",
    "AIzaSyA9S1V66bDs9UrnnVJKy_zDbxWQh6MMxtM"
]

# --- 3. BẢO VỆ THƯ VIỆN (CHỐNG SẬP) ---
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
                # Tắt bộ lọc an toàn để tránh lỗi trả về rỗng
                safety = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                response = model.generate_content(prompt, safety_settings=safety)
                if response.text:
                    return response.text, model_name
            except: continue
    return "❌ Mạng nghẽn, AI chưa trả lời kịp.", "Error"

# --- 5. HÀM LẤY DATA & CHỈ SỐ ---
@st.cache_data(ttl=300)
def get_data_safe(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        # Thử DNSE
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        # Nếu lỗi thử TCBS
        if df is None or df.empty:
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
        
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            # Chỉ báo kỹ thuật
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

def get_market_index():
    """Lấy chỉ số VNINDEX thực tế"""
    try:
        df = get_data_safe("VNINDEX")
        if df is not None:
            last = df.iloc[-1]
            change = last['close'] - df.iloc[-2]['close']
            pct = (change / df.iloc[-2]['close']) * 100
            return f"VN-Index: {last['close']:,.0f} điểm ({change:+.2f}đ, {pct:+.2f}%). Xu hướng: {'Tăng' if change>0 else 'Giảm'}."
    except: pass
    return "Không lấy được VNINDEX."

# --- 6. GIAO DIỆN CHÍNH ---
st.sidebar.title("💎 TA Alex 2026")
st.sidebar.success(f"✅ Đã nạp {len(API_KEY_POOL)} Key Vô Hạn")

# TẠO 3 TAB
tab1, tab2, tab3 = st.tabs(["📊 Phân Tích", "🚀 Scanner VN30", "💬 Chat AI"])

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
                status.write("✅ Data OK.")
                
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
                status.write("🤖 Gọi Alex...")
                data_text = df.tail(30).to_string()
                vnindex_ctx = get_market_index()
                prompt = f"""
                Bối cảnh thị trường: {vnindex_ctx}
                Mã: {symbol}. Giá: {last['close']}.
                Dữ liệu kỹ thuật 30 phiên:
                {data_text}
                Hãy phân tích xu hướng Mua/Bán ngắn gọn.
                """
                ai_text, model_used = call_ai_rotation(prompt)
                
                st.info(f"💡 Nhận định ({model_used}):")
                st.write(ai_text)
                status.update(label="Hoàn tất!", state="complete", expanded=False)
            else:
                status.update(label="Lỗi mã!", state="error")
                st.error("Không tìm thấy mã này.")
        except Exception as e: st.error(str(e))

# === TAB 2: SCANNER VN30 ===
with tab2:
    st.header("🕵️ Quét VN30 - Tìm Mã MUA")
    vn30_list = "ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, MBB, MSN, MWG, PLX, PNJ, POW, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE"
    scan_input = st.text_area("Danh sách:", value=vn30_list, height=100)
    
    if st.button("🚀 Tìm Mã MUA Ngay"):
        symbols = [s.strip().upper() for s in scan_input.split(",") if s.strip()]
        results = []
        bar = st.progress(0)
        
        for i, sym in enumerate(symbols):
            try:
                d = get_data_safe(sym)
                if d is not None:
                    r = d.iloc[-1]
                    s = 0
                    if r['close'] > r['MA20']: s += 1
                    if r['MA20'] > r['MA50']: s += 1
                    if r['MACD'] > r['Signal_Line']: s += 1.5
                    if 40 < r['RSI'] < 65: s += 0.5
                    
                    act = "⚪ Quan sát"
                    if s >= 4: act = "🟢 MUA MẠNH"
                    elif s >= 3: act = "🟢 MUA GOM"
                    elif s <= 1: act = "🔴 BÁN"
                    
                    results.append({"Mã": sym, "Giá": r['close'], "Điểm": s, "Hành động": act})
            except: pass
            bar.progress((i + 1) / len(symbols))
            
        if results:
            df_res = pd.DataFrame(results).sort_values(by="Điểm", ascending=False)
            def color_act(val):
                if "MUA" in str(val): return 'background-color: #28a745; color: white'
                return ''
            st.dataframe(df_res.style.applymap(color_act, subset=['Hành động']), use_container_width=True)
            
            # AI Top 3
            top3 = df_res.head(3)
            st.markdown("---")
            st.subheader(f"🏆 Top 3 Mã Ngon Nhất")
            with st.spinner("Đang soi kỹ thuật Top 3..."):
                prompt = f"Top 3 mã kỹ thuật đẹp hôm nay: {top3.to_string()}. Khuyến nghị điểm mua/bán cho từng mã."
                ai_reply, _ = call_ai_rotation(prompt)
                st.write(ai_reply)

# === TAB 3: CHAT AI THÔNG MINH (REAL-TIME STOCK) ===
with tab3:
    st.header("💬 Trò chuyện với Alex (Live Data)")
    st.caption("Mẹo: Hãy viết hoa mã cổ phiếu (VD: MBB, FPT) để Alex tự lấy dữ liệu.")

    # Hàm tìm mã chứng khoán trong câu chat
    def extract_symbol(text):
        matches = re.findall(r'\b[A-Z]{3}\b', text)
        valid = [m for m in matches if m not in ["MUA", "BAN", "GIA", "RSI", "MACD", "VNI", "TANG", "GIAM"]]
        return valid[0] if valid else None

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Hỏi về MBB, FPT, hay thị trường..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Đang soi bảng điện..."):
                # 1. Lấy VNINDEX
                market_info = get_market_index()
                ctx_data = f"- Thị trường chung: {market_info}\n"
                
                # 2. Tìm & Lấy dữ liệu Cổ phiếu riêng (Nếu có)
                detected_symbol = extract_symbol(prompt)
                if detected_symbol:
                    df_s = get_data_safe(detected_symbol)
                    if df_s is not None:
                        l = df_s.iloc[-1]
                        ctx_data += f"- {detected_symbol}: Giá {l['close']:,.0f}, RSI {l['RSI']:.1f}, MACD {l['MACD']:.2f}, Xu hướng {'Tăng' if l['close']>l['MA20'] else 'Giảm'}.\n"
                        st.toast(f"Đã lấy dữ liệu {detected_symbol}", icon="✅")
                
                # 3. Ghép Prompt
                full_prompt = f"Dữ liệu LIVE 2026:\n{ctx_data}\nCâu hỏi: {prompt}\nHãy trả lời dựa trên dữ liệu trên."
                
                response, _ = call_ai_rotation(full_prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
