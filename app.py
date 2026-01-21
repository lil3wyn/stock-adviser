import streamlit as st

# --- 1. CẤU HÌNH ĐẦU TIÊN (BẮT BUỘC) ---
st.set_page_config(layout="wide", page_title="TA Alex Pro Advisor", page_icon="📈")

# --- 2. KIỂM TRA TRẠNG THÁI (Để tránh màn hình trắng) ---
status_text = st.empty()
status_text.info("🔄 Đang khởi động hệ thống... (Vui lòng đợi 10s)")

# --- 3. NẠP THƯ VIỆN (Trong vòng bảo vệ) ---
try:
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    # Nạp vnstock an toàn
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
    import time
    
    # Xóa thông báo khởi động khi nạp xong
    status_text.empty()
    
except Exception as e:
    st.error(f"❌ LỖI KHỞI ĐỘNG: {str(e)}")
    st.warning("Gợi ý: Hãy kiểm tra file requirements.txt xem đã có đủ thư viện chưa.")
    st.stop()

# --- 4. HÀM XỬ LÝ (GIỮ NGUYÊN) ---
def calculate_indicators(df):
    if df is None or df.empty: return None
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

def get_stock_data(symbol, days=365):
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df = calculate_indicators(df)
            return df
        return None
    except: return None

def get_live_price(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1m', type='stock', source='DNSE')
        if df is not None and not df.empty:
            return float(df.iloc[-1]['close'])
        return None
    except: return None

# --- 5. SIDEBAR ---
st.sidebar.title("⚙️ Cấu hình Pro")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ VIP Member: Active")
else:
    api_key = st.sidebar.text_input("Nhập Gemini API Key", type="password")

available_models = []
if api_key:
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                if "1.0" not in name and "1.5" not in name: available_models.append(name)
    except: pass

if available_models:
    available_models.sort(key=lambda x: ('3' not in x, 'flash' not in x))
    model_name = st.sidebar.selectbox("Brain:", available_models, index=0)
else:
    model_name = st.sidebar.selectbox("Brain:", ["gemini-2.0-flash-exp"], index=0)

# --- 6. GIAO DIỆN CHÍNH ---
st.title("📈 TA Alex Pro System")
tab1, tab2 = st.tabs(["📊 Phân Tích Sâu", "🚀 Siêu Bộ Lọc (Scanner)"])

# === TAB 1: PHÂN TÍCH ===
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="FPT").upper()
    
    if symbol and api_key:
        if st.button("Phân tích", type="primary"):
            with st.spinner('Đang tính toán...'):
                df = get_stock_data(symbol)
                live = get_live_price(symbol)
                if df is not None:
                    last = df.iloc[-1]
                    price = live if live else last['close']
                    
                    m1, m2, m3, m4 = st.columns(4)
                    change = price - df.iloc[-2]['close']
                    m1.metric("Giá", f"{price:,.0f}", f"{change:,.0f}")
                    m2.metric("RSI", f"{last['RSI']:.1f}")
                    m3.metric("MACD", "Tăng" if last['MACD']>last['Signal_Line'] else "Giảm")
                    m4.metric("Dòng tiền", f"{last['Vol_Ratio']*100:.0f}% TB20")

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.tail(100)['time'], open=df.tail(100)['open'], high=df.tail(100)['high'], low=df.tail(100)['low'], close=df.tail(100)['close'], name="Giá"))
                    fig.add_trace(go.Scatter(x=df.tail(100)['time'], y=df.tail(100)['MA20'], line=dict(color='orange'), name="MA20"))
                    fig.add_trace(go.Scatter(x=df.tail(100)['time'], y=df.tail(100)['MA50'], line=dict(color='blue'), name="MA50"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    data_ctx = df.tail(60)[['time', 'close', 'RSI', 'MACD', 'Signal_Line', 'Vol_Ratio']].to_string(index=False)
                    sys_prompt = f"Bạn là TA Alex. Model: {model_name}. Giá {symbol}: {price}. Dữ liệu:\n{data_ctx}\n. Hãy phân tích kỹ thuật chuyên sâu (MACD, RSI, Volume) và đưa ra hành động Mua/Bán."
                    
                    safety_settings = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                    try:
                        model = genai.GenerativeModel(model_name)
                        st.success(model.generate_content(sys_prompt, safety_settings=safety_settings).text)
                    except Exception as e: st.error(str(e))

# === TAB 2: SCANNER ===
with tab2:
    st.header("🕵️ Máy Quét Cơ Hội")
    scan_list = st.text_area("Danh sách mã (cách nhau dấu phẩy):", value="ACB, FPT, HPG, MBB, MSN, MWG, SSI, STB, TCB, VHM, VIC, VNM, VPB")
    
    if st.button("🔍 Quét Ngay"):
        symbols = [s.strip().upper() for s in scan_list.split(",") if s.strip()]
        results = []
        bar = st.progress(0)
        
        for i, sym in enumerate(symbols):
            df = get_stock_data(sym, days=150)
            if df is not None:
                row = df.iloc[-1]
                score = 0
                if row['close'] > row['MA20']: score += 1
                if row['MA20'] > row['MA50']: score += 1
                if row['MACD'] > row['Signal_Line']: score += 1.5
                if row['Vol_Ratio'] > 1.2: score += 1.5
                
                rank = "Yếu"
                if score >= 4: rank = "🔥 Khỏe"
                
                results.append({"Mã": sym, "Giá": row['close'], "Điểm": score, "Xếp loại": rank})
            bar.progress((i+1)/len(symbols))
            
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Điểm", ascending=False)
            st.dataframe(res_df, use_container_width=True)
            
            top = res_df.head(1)
            if not top.empty:
                st.subheader(f"🏆 Alex chọn: {top.iloc[0]['Mã']}")
