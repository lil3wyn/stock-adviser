import streamlit as st

# --- 1. CẤU HÌNH TRANG (BẮT BUỘC Ở DÒNG ĐẦU) ---
st.set_page_config(layout="wide", page_title="TA Alex Pro", page_icon="💎")

# --- 2. NẠP THƯ VIỆN AN TOÀN ---
try:
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
    import time
except Exception as e:
    st.error(f"❌ Lỗi thư viện: {e}")
    st.stop()

# --- 3. CÁC HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_data(ttl=300) 
def get_data_safe(symbol, days=365):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Thử lấy dữ liệu từ DNSE
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        
        # Nếu lỗi, thử TCBS
        if df is None or df.empty:
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
            
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            
            # Chỉ báo xu hướng
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            
            # Bollinger Bands
            std = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['MA20'] + (std * 2)
            df['BB_Lower'] = df['MA20'] - (std * 2)
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Volume
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

# --- 4. SIDEBAR & CẤU HÌNH AI ---
st.sidebar.title("💎 TA Alex Pro")

# Tự động nhận Key
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ Đã kích hoạt bản quyền")
else:
    api_key = st.sidebar.text_input("Nhập Gemini API Key", type="password")

# Tự động chọn Model sống (QUAN TRỌNG: Lọc bỏ model 1.5 đã chết)
available_models = []
if api_key:
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                # Chỉ lấy những model đời mới (tránh lỗi 404 của bản 1.5)
                if
