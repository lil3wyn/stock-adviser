import streamlit as st
# --- 1. QUAN TRỌNG: CẤU HÌNH TRANG PHẢI NẰM ĐẦU TIÊN ---
# (Chuyển dòng này lên đây để tránh lỗi màn hình trắng)
st.set_page_config(layout="wide", page_title="TA Alex Pro Advisor", page_icon="📈")

import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from vnstock import stock_historical_data
from datetime import datetime, timedelta
import time

# --- 2. HÀM TÍNH TOÁN KỸ THUẬT NÂNG CAO ---
def calculate_indicators(df):
    if df is None or df.empty: return None
    
    # 1. Basic Trend
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # 2. Bollinger Bands
    std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    # 3. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD (Chỉ báo quan trọng)
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. Volume Analysis (Dòng tiền)
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['volume'] / df['Vol_MA20'] # >1.5 là tiền vào mạnh
    
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

# --- 3. SIDEBAR ---
st.sidebar.title("⚙️ Cấu hình Pro")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ VIP Member: Active")
else:
    api_key = st.sidebar.text_input("Nhập Gemini API Key", type="password")

# Model Selection logic
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

# --- 4. GIAO DIỆN CHÍNH ---
st.title("📈 TA Alex Pro System")
tab1, tab2 = st.tabs(["📊 Phân Tích Sâu", "🚀 Siêu Bộ Lọc (Pro Scanner)"])

# === TAB 1: PHÂN TÍCH SÂU ===
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="FPT").upper()
    
    if symbol and api_key:
        if st.button("Phân tích", type="primary"):
            with st.spinner('Đang tính toán chỉ số nâng cao...'):
                df = get_stock_data(symbol)
                live = get_live_price(symbol)
                
                if df is not None:
                    last = df.iloc[-1]
                    price = live if live else last['close']
                    
                    # Metrics Display
                    m1, m2, m3, m4 = st.columns(4)
                    change = price - df.iloc[-2]['close']
                    m1.metric("Giá", f"{price:,.0f}", f"{change:,.0f}")
                    m2.metric("RSI (Sức mạnh)", f"{last['RSI']:.1f}")
                    
                    # MACD Signal
                    macd_status = "Tăng" if last['MACD'] > last['Signal_Line'] else "Giảm"
                    m3.metric("MACD Trend", macd_status, f"{last['MACD']:.2f}")
                    
                    # Volume Analysis
                    vol_status = "Đột biến" if last['Vol_Ratio'] > 1.2 else "Bình thường"
                    m4.metric("Dòng tiền", vol_status, f"{last['Vol_Ratio']*100:.0f}% TB20")

                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.tail(100)['time'],
                        open=df.tail(100)['open'], high=df.tail(100)['high'],
                        low=df.tail(100)['low'], close=df.tail(100)['close'], name="Giá"))
                    fig.add_trace(go.Scatter(x=df.tail(100)['time'], y=df.tail(100)['MA20'], line=dict(color='orange', width=1), name="MA20"))
                    fig.add_trace(go.Scatter(x=df.tail(100)['time'], y=df.tail(100)['MA50'], line=dict(color='blue', width=1), name="MA50 (Trung hạn)"))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI PRO PROMPT
                    data_ctx = df.tail(60)[['time', 'close', 'RSI', 'MACD', 'Signal_Line', 'Vol_Ratio', 'MA20', 'MA50']].to_string(index=False)
                    sys_prompt = f"""
                    Bạn là TA Alex (Pro Trader). Model: {model_name}.
                    DỮ LIỆU KỸ THUẬT CHUYÊN SÂU CỦA {symbol}:
                    - Giá hiện tại: {price}
                    - Chỉ báo xu hướng: MACD, MA20, MA50.
                    - Chỉ báo động lượng: RSI.
                    - Chỉ báo dòng tiền: Vol_Ratio (Lớn hơn 1.0 là tiền vào).
                    
                    Dữ liệu 60 phiên gần nhất:
                    {data_ctx}
                    
                    YÊU CẦU PHÂN TÍCH:
                    1. Xu hướng chính (Uptrend/Downtrend) dựa trên MA và MACD.
                    2. Có tín hiệu "Dòng tiền thông minh" (Smart Money) vào không? (Dựa trên Vol_Ratio).
                    3. Kết luận: MUA GOM / MUA ĐUỔI / CHỐT LỜI / CẮT LỖ.
                    """
                    
                    # Safety Settings (Quan trọng)
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]

                    try:
                        model = genai.GenerativeModel(model_name)
                        with st.spinner("Đang kích hoạt não bộ AI..."):
                            resp = model.generate_content(sys_prompt, safety_settings=safety_settings)
                            if resp.text:
                                st.success(resp.text)
                            else:
                                st.warning("AI không phản hồi.")
                    except Exception as e: st.error(str(e))

# === TAB 2: PRO SCANNER ===
with tab2:
    st.header("🕵️ Máy Quét Cơ Hội (Scoring System)")
    st.caption("Chấm điểm sức mạnh kỹ thuật (Technical Rating) trên thang 10.")
    
    default_list = "ACB, FPT, HPG, MBB, MSN, MWG, PNJ, SSI, STB, TCB, VHM, VIC, VNM, VPB, DIG, CEO, DXG, VND, SHS"
    scan_list = st.text_area("Danh sách mã:", value=default_list)
    
    if st.button("🔍 Quét & Chấm điểm", type="primary"):
        symbols = [s.strip().upper() for s in scan_list.split(",") if s.strip()]
        results = []
        bar = st.progress(0)
        
        with st.spinner("Đang tính toán MACD, RSI, Volume Flow..."):
            for i, sym in enumerate(symbols):
                df = get_stock_data(sym, days=150)
                if df is not None:
                    row = df.iloc[-1]
                    
                    # --- HỆ THỐNG CHẤM ĐIỂM (ALGO) ---
                    score = 0
                    reasons = []
                    
                    # 1. Điểm Xu hướng (MA) - Max 3 điểm
                    if row['close'] > row['MA20']: score += 1; reasons.append("Trên MA20")
                    if row['MA20'] > row['MA50']: score += 1; reasons.append("Trend Tăng Trung Hạn")
                    if 'MA200' in row and row['close'] > row['MA200']: score += 1
                    
                    # 2. Điểm Động lượng (RSI & MACD) - Max 3 điểm
                    if 45 < row['RSI'] < 70: score += 1
                    if row['MACD'] > row['Signal_Line']: score += 1.5; reasons.append("MACD Báo Mua")
                    elif row['MACD'] > 0: score += 0.5
                    
                    # 3. Điểm Dòng tiền (Volume) - Max 2 điểm
                    if row['Vol_Ratio'] > 1.2: score += 1.5; reasons.append(f"Tiền vào mạnh (x{row['Vol_Ratio']:.1f})")
                    elif row['Vol_Ratio'] > 1.0: score += 0.5
                    
                    # 4. Điểm Đột phá (Breakout) - Max 2 điểm
                    if row['close'] >= df['close'].tail(20).max(): score += 2; reasons.append("Breakout Đỉnh 20 ngày")

                    # Xếp loại
                    rank = "Yếu"
                    if score >= 7: rank = "💎 SIÊU CỔ"
                    elif score >= 5: rank = "🔥 Khỏe"
                    elif score >= 3: rank = "😐 Trung tính"
                    
                    results.append({
                        "Mã": sym,
                        "Giá": row['close'],
                        "Điểm (10)": round(score, 1),
                        "Xếp loại": rank,
                        "RSI": round(row['RSI'], 1),
                        "Lý do chính": ", ".join(reasons[:2])
                    })
                bar.progress((i+1)/len(symbols))
                
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Điểm (10)", ascending=False)
            
            # Tô màu bảng kết quả
            def color_rank(val):
                if "SIÊU" in val: return 'background-color: #28a745; color: white' 
                if "Khỏe" in val: return 'background-color: #90ee90; color: black'
                if "Yếu" in val: return 'background-color: #ffcccc; color: black'
                return ''

            # Dùng applymap (phiên bản cũ cho chắc ăn)
            st.dataframe(res_df.style.applymap(color_rank, subset=['Xếp loại']), use_container_width=True)
            
            # AI Nhận xét
            top_stocks = res_df.head(3)
            if not top_stocks.empty:
                st.markdown("---")
                st.subheader(f"🏆 Alex chọn: {top_stocks.iloc[0]['Mã']}")
                prompt = f"""
                Dựa trên bảng điểm kỹ thuật này:
                {top_stocks.to_string()}
                
                Hãy phân tích kỹ thuật nhanh cho mã đứng đầu (Top 1).
                Tại sao nó lại có điểm số cao như vậy?
                """
                
                # Safety Settings cho phần Scanner
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                try:
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(prompt, safety_settings=safety_settings)
                    st.write(resp.text)
                except: pass
