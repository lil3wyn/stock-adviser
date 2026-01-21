import streamlit as st
import time

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="TA Alex Pro", page_icon="🤑")

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

# --- 4. HÀM AI XOAY TUA ---
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
                    return response.text, model_name
            except: continue
    return "❌ Mạng nghẽn, AI chưa trả lời kịp.", "Error"

# --- 5. HÀM LẤY DATA ---
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
st.sidebar.title("🤑 TA Alex Advisor")
st.sidebar.success(f"✅ Đã nạp {len(API_KEY_POOL)} Key")

tab1, tab2, tab3 = st.tabs(["📊 Phân Tích", "🚀 Scanner VN30", "💬 Chat AI"])

# === TAB 1: PHÂN TÍCH ===
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="MBB").upper()
    
    if st.button("🔍 Phân Tích Ngay", type="primary"):
        status = st.status("🚀 Đang soi...", expanded=True)
        try:
            df = get_data_safe(symbol)
            if df is not None:
                last = df.iloc[-1]
                status.write("✅ Dữ liệu OK.")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Giá", f"{last['close']:,.0f}")
                c2.metric("RSI", f"{last['RSI']:.1f}")
                c3.metric("MACD", f"{last['MACD']:.2f}")
                c4.metric("Vol", f"{last['volume']:,.0f}")
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Giá"))
                fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], line=dict(color='orange'), name="MA20"))
                st.plotly_chart(fig, use_container_width=True)
                
                status.write("🤖 Alex đang viết nhận định...")
                data_text = df.tail(30).to_string()
                prompt = f"Giá {symbol}: {last['close']}. Dữ liệu:\n{data_text}\n. Hãy cho tôi khuyến nghị MUA hay BÁN ngắn gọn."
                ai_text, model_used = call_ai_rotation(prompt)
                
                st.info(f"💡 Nhận định ({model_used}):")
                st.write(ai_text)
                status.update(label="Xong!", state="complete", expanded=False)
            else:
                status.update(label="Lỗi mã!", state="error")
                st.error("Không tìm thấy mã này.")
        except Exception as e: st.error(f"Lỗi: {e}")

# === TAB 2: SCANNER VN30 (PHÍM HÀNG RÕ RÀNG) ===
with tab2:
    st.header("🕵️ Quét VN30 - Tìm Mã MUA")
    
    vn30_list = "ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, MBB, MSN, MWG, PLX, PNJ, POW, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE"
    scan_input = st.text_area("Danh sách:", value=vn30_list, height=100)
    
    if st.button("🚀 Tìm Mã MUA Ngay"):
        symbols = [s.strip().upper() for s in scan_input.split(",") if s.strip()]
        results = []
        bar = st.progress(0)
        status_text = st.empty()
        
        for i, sym in enumerate(symbols):
            status_text.text(f"Đang soi: {sym}...")
            try:
                d = get_data_safe(sym)
                if d is not None:
                    r = d.iloc[-1]
                    
                    # --- CHẤM ĐIỂM ---
                    score = 0
                    if r['close'] > r['MA20']: score += 1
                    if r['MA20'] > r['MA50']: score += 1
                    if r['MACD'] > r['Signal_Line']: score += 1.5
                    if 40 < r['RSI'] < 65: score += 0.5
                    
                    # --- RA QUYẾT ĐỊNH ---
                    action = "⚪ Quan sát"
                    if score >= 4:
                        action = "🟢 MUA MẠNH"
                    elif score >= 3:
                        action = "🟢 MUA GOM"
                    elif score >= 2:
                        action = "🟡 CÂN NHẮC"
                    else:
                        action = "🔴 YẾU / BÁN"

                    results.append({
                        "Mã": sym,
                        "Giá": r['close'],
                        "Điểm": score,
                        "Hành động": action,
                        "RSI": round(r['RSI'], 1)
                    })
            except: pass
            bar.progress((i + 1) / len(symbols))
            
        status_text.empty()
        bar.empty()
        
        if results:
            df_res = pd.DataFrame(results).sort_values(by="Điểm", ascending=False)
            
            # Tô màu chữ MUA
            def highlight_buy(val):
                if "MUA" in str(val): return 'background-color: #28a745; color: white; font-weight: bold'
                if "YẾU" in str(val): return 'background-color: #dc3545; color: white'
                return ''
            
            st.dataframe(df_res.style.applymap(highlight_buy, subset=['Hành động']), use_container_width=True)
            
            # AI KHUYẾN NGHỊ TOP 3
            top_stocks = df_res.head(3)
            st.markdown("---")
            st.subheader(f"🏆 Top 3 Siêu Phẩm Hôm Nay")
            
            with st.spinner("Alex đang phân tích kỹ Top 3..."):
                prompt = f"""
                Đây là Top 3 cổ phiếu có kỹ thuật đẹp nhất hôm nay:
                {top_stocks.to_string()}
                
                Hãy viết khuyến nghị đầu tư ngắn gọn cho từng mã.
                Nói rõ: Điểm mua, Điểm cắt lỗ, Điểm chốt lời cho từng mã.
                """
                ai_reply, model_used = call_ai_rotation(prompt)
                st.write(ai_reply)

# === TAB 3: CHAT AI ===
with tab3:
    st.header("💬 Chat với Chuyên gia")
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi gì đi (VD: Mai thị trường sao?)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("..."):
                response, _ = call_ai_rotation(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
