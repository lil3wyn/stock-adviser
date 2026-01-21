import streamlit as st
import time
import random

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="TA Alex 2026", page_icon="💎")

# --- 2. KHO VŨ KHÍ: 5 KEY CỦA BẠN (Đã nạp sẵn) ---
API_KEY_POOL = [
    "AIzaSyAcIDpmFgBVzIlb41m1cz4BPlTCjKM9Hl0",
    "AIzaSyBC_V9ACvGCElaWQL5BILKQCv_ikBGcsHs", 
    "AIzaSyCFgTf678MHOoaOMmfV6y0uXLVrT2VwPV8",
    "AIzaSyBJhszyVcCesLBHlL2mfEP3Tx-ykMyA4_w",
    "AIzaSyA9S1V66bDs9UrnnVJKy_zDbxWQh6MMxtM"
]

# --- 3. NẠP THƯ VIỆN AN TOÀN ---
try:
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
except Exception as e:
    st.error(f"❌ Lỗi thư viện: {e}")
    st.stop()

# --- 4. HÀM GỌI AI "BẤT TỬ" (Auto-Rotation) ---
def call_ai_rotation(prompt):
    # Danh sách model ưu tiên (2026)
    # Ưu tiên dùng bản 3.0, nếu chết thì lùi về 2.0
    models_to_try = ["gemini-3-flash-preview", "gemini-2.0-flash-exp"]
    
    msg = st.empty()
    
    # Chiến thuật: Thử từng Key
    for i, key in enumerate(API_KEY_POOL):
        # Với mỗi Key, thử từng Model
        for model_name in models_to_try:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)
                
                # Tắt bộ lọc an toàn để tránh lỗi "finish_reason 1" (trả về rỗng)
                safety = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                
                # Gọi AI
                response = model.generate_content(prompt, safety_settings=safety)
                
                if response.text:
                    msg.empty()
                    return response.text, model_name # Trả về kết quả và tên model đã dùng
                    
            except Exception as e:
                err = str(e)
                # Nếu là lỗi Quota (429) -> Đổi Key khác
                if "429" in err or "Quota" in err:
                    msg.warning(f"⚠️ Key {i+1} quá tải. Đang đổi sang Key {i+2}...", icon="🔄")
                    break # Thoát vòng lặp model để đổi Key mới
                
                # Nếu là lỗi Model không tìm thấy (404) hoặc lỗi khác -> Thử model tiếp theo
                continue 

    return "❌ Tất cả 5 Key đều tạch! Mai quay lại nhé.", "None"

# --- 5. HÀM DỮ LIỆU ---
@st.cache_data(ttl=300)
def get_data_safe(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        if df is None or df.empty:
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            std = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['MA20'] + (std * 2)
            df['BB_Lower'] = df['MA20'] - (std * 2)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
            df['Vol_Ratio'] = df['volume'] / df['Vol_MA20']
            return df
        return None
    except: return None

def get_live_price(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1m', type='stock', source='DNSE')
        if df is not None and not df.empty: return float(df.iloc[-1]['close'])
        return None
    except: return None

# --- 6. GIAO DIỆN ---
st.sidebar.title("💎 TA Alex 2026")
st.sidebar.success(f"✅ Đã kích hoạt 5 Key Vô Hạn!")
st.sidebar.info("Mặc định: gemini-3-flash-preview")

tab1, tab2 = st.tabs(["📊 Phân Tích", "🚀 Scanner"])

# TAB 1
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="SSI").upper()
    if st.button("🔍 Phân Tích Ngay", type="primary"):
        with st.spinner("Đang tải dữ liệu..."):
            df = get_data_safe(symbol)
            live = get_live_price(symbol)
            if df is not None:
                last = df.iloc[-1]
                price = live if live else last['close']
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Giá", f"{price:,.0f}")
                c2.metric("RSI", f"{last['RSI']:.1f}")
                c3.metric("MACD", "Tăng" if last['MACD']>last['Signal_Line'] else "Giảm")
                c4.metric("Vol", f"{last['Vol_Ratio']*100:.0f}% TB20" if pd.notna(last['Vol_Ratio']) else "-")
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.tail(60)['time'], open=df.tail(60)['open'], high=df.tail(60)['high'], low=df.tail(60)['low'], close=df.tail(60)['close'], name="Giá"))
                fig.add_trace(go.Scatter(x=df.tail(60)['time'], y=df.tail(60)['MA20'], line=dict(color='orange'), name="MA20"))
                st.plotly_chart(fig, use_container_width=True)
                
                # GỌI AI XOAY TUA
                data_ctx = df.tail(30)[['time', 'close', 'RSI', 'MACD']].to_string(index=False)
                prompt = f"Giá {symbol}: {price}. Dữ liệu:\n{data_ctx}\n. Phân tích ngắn gọn Mua/Bán."
                
                ai_reply, used_model = call_ai_rotation(prompt)
                
                st.success(f"🤖 Alex ({used_model}) nhận định:")
                st.write(ai_reply)
            else: st.error("Lỗi mã.")

# TAB 2
with tab2:
    st.header("🕵️ Quét Cổ Phiếu")
    scan_list = st.text_area("Danh sách:", value="ACB, FPT, HPG, MBB, MSN, SSI, STB, TCB, VHM, VIC, VNM, VPB")
    if st.button("🚀 Quét"):
        symbols = [s.strip().upper() for s in scan_list.split(",") if s.strip()]
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
                    rk = "Yếu"
                    if s >= 3.5: rk = "🔥 Khỏe"
                    elif s >= 2: rk = "😐 Trung"
                    results.append({"Mã": sym, "Giá": r['close'], "Điểm": s, "Xếp loại": rk})
            except: pass
            bar.progress((i+1)/len(symbols))
        
        if results:
            df_res = pd.DataFrame(results).sort_values(by="Điểm", ascending=False)
            st.dataframe(df_res, use_container_width=True)
            top = df_res.iloc[0]
            st.subheader(f"🏆 Top 1: {top['Mã']}")
            
            # GỌI AI XOAY TUA
            explain, used_model = call_ai_rotation(f"Tại sao {top['Mã']} kỹ thuật tốt? Ngắn gọn.")
            st.write(f"*(Phân tích bởi {used_model})*")
            st.write(explain)
