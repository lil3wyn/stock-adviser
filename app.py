import streamlit as st
import time

# --- 1. KHỞI ĐỘNG AN TOÀN ---
st.set_page_config(layout="wide", page_title="TA Alex Final", page_icon="💎")
status_placeholder = st.empty()
status_placeholder.info("⏳ Đang khởi động hệ thống... (0%)")

# --- 2. NẠP THƯ VIỆN (CÓ BÁO CÁO) ---
try:
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
    status_placeholder.info("⏳ Đang nạp thư viện dữ liệu... (50%)")
    time.sleep(0.5) # Nghỉ xíu cho hệ thống thở
except Exception as e:
    st.error(f"❌ Lỗi nạp thư viện: {e}")
    st.stop()

status_placeholder.success("✅ Hệ thống đã sẵn sàng!")
time.sleep(1)
status_placeholder.empty() # Xóa thông báo loading

# --- 3. HÀM XỬ LÝ (CHỐNG LỖI) ---
def get_data_safe(symbol, days=365):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Thử DNSE
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        
        # Fallback TCBS
        if df is None or df.empty:
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
            
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            
            # Chỉ báo
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
        if df is not None and not df.empty:
            return float(df.iloc[-1]['close'])
        return None
    except: return None

# --- 4. GIAO DIỆN ---
st.sidebar.title("💎 TA Alex Pro")

# Auto Key
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ Đã kích hoạt bản quyền")
else:
    api_key = st.sidebar.text_input("Nhập Gemini API Key", type="password")

# Model Selector (FIX LỖI 404 CỰC MẠNH)
model_options = ["gemini-2.0-flash-exp"] # Mặc định an toàn
if api_key:
    try:
        genai.configure(api_key=api_key)
        # Lấy danh sách model thực tế đang sống
        models = genai.list_models()
        found_models = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                # Lọc bỏ model 1.5 đã chết
                if "1.5" not in name and "1.0" not in name:
                    found_models.append(name)
        if found_models:
            # Ưu tiên bản mới nhất lên đầu
            found_models.sort(key=lambda x: ('3' not in x, 'flash' not in x))
            model_options = found_models
    except: pass

model_name = st.sidebar.selectbox("Model:", model_options, index=0)

# --- 5. TABS ---
tab1, tab2 = st.tabs(["📊 Phân Tích", "🚀 Scanner"])

# TAB 1
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="FPT").upper()
        
    if st.button("🔍 Phân Tích Ngay", type="primary"):
        if not api_key: st.warning("Cần nhập API Key!")
        else:
            with st.spinner("Đang tải dữ liệu..."):
                df = get_data_safe(symbol)
                live = get_live_price(symbol)
                
                if df is not None:
                    last = df.iloc[-1]
                    price = live if live else last['close']
                    
                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    change = price - df.iloc[-2]['close']
                    pct = (change/df.iloc[-2]['close'])*100
                    c1.metric("Giá", f"{price:,.0f}", f"{change:,.0f} ({pct:.1f}%)")
                    c2.metric("RSI", f"{last['RSI']:.1f}")
                    c3.metric("MACD", "Tăng" if last['MACD']>last['Signal_Line'] else "Giảm")
                    c4.metric("Vol/TB20", f"{last['Vol_Ratio']*100:.0f}%" if pd.notna(last['Vol_Ratio']) else "-")
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.tail(60)['time'], open=df.tail(60)['open'], high=df.tail(60)['high'], low=df.tail(60)['low'], close=df.tail(60)['close'], name="Giá"))
                    fig.add_trace(go.Scatter(x=df.tail(60)['time'], y=df.tail(60)['MA20'], line=dict(color='orange'), name="MA20"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI Analysis (Đã fix lỗi model)
                    data_ctx = df.tail(60)[['time', 'close', 'RSI', 'MACD', 'Signal_Line']].to_string(index=False)
                    prompt = f"Giá {symbol}: {price}. Dữ liệu:\n{data_ctx}\n. Phân tích kỹ thuật ngắn gọn, khuyến nghị Mua/Bán."
                    
                    try:
                        model = genai.GenerativeModel(model_name)
                        # Tắt bộ lọc an toàn để tránh lỗi
                        safety = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                        st.info("🤖 Alex đang viết nhận định...")
                        resp = model.generate_content(prompt, safety_settings=safety)
                        if resp.text: st.success(resp.text)
                    except Exception as e: st.error(f"Lỗi AI: {e}")
                else: st.error("Không tìm thấy mã này.")

# TAB 2 (Scanner)
with tab2:
    st.header("🕵️ Máy Quét")
    scan_list = st.text_area("Danh sách:", value="ACB, FPT, HPG, MBB, MSN, SSI, STB, TCB, VHM, VIC, VNM, VPB")
    
    if st.button("🚀 Quét"):
        symbols = [s.strip().upper() for s in scan_list.split(",") if s.strip()]
        results = []
        bar = st.progress(0)
        
        for i, sym in enumerate(symbols):
            try:
                df = get_data_safe(sym, days=150)
                if df is not None:
                    row = df.iloc[-1]
                    score = 0
                    if row['close'] > row['MA20']: score += 1
                    if row['MA20'] > row['MA50']: score += 1
                    if row['MACD'] > row['Signal_Line']: score += 1.5
                    
                    rank = "Yếu"
                    if score >= 3.5: rank = "🔥 Khỏe"
                    elif score >= 2: rank = "😐 Trung"
                    
                    results.append({"Mã": sym, "Giá": row['close'], "Điểm": score, "Xếp loại": rank})
            except: pass
            bar.progress((i+1)/len(symbols))
            
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Điểm", ascending=False)
            st.dataframe(res_df, use_container_width=True)
            
            # AI Comment Top 1
            top = res_df.iloc[0]
            st.subheader(f"🏆 Top 1: {top['Mã']}")
            try:
                model = genai.GenerativeModel(model_name)
                st.write(model.generate_content(f"Tại sao {top['Mã']} kỹ thuật tốt? Ngắn gọn.").text)
            except: pass
