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
                if "1.5" not in name and "1.0" not in name: 
                    available_models.append(name)
    except: pass

if available_models:
    # Ưu tiên bản 3.0 hoặc flash
    available_models.sort(key=lambda x: ('3' not in x, 'flash' not in x))
    model_name = st.sidebar.selectbox("Model:", available_models, index=0)
else:
    # Fallback nếu không tìm thấy gì (Dùng bản 2.0 experimental)
    model_name = st.sidebar.selectbox("Model:", ["gemini-2.0-flash-exp"], index=0)

# --- 5. GIAO DIỆN CHÍNH ---
tab1, tab2 = st.tabs(["📊 Phân Tích Chi Tiết", "🚀 Siêu Bộ Lọc (Scanner)"])

# === TAB 1: PHÂN TÍCH ===
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="MBB").upper()
    
    if st.button("🔍 Phân Tích Ngay", type="primary"):
        if not api_key:
            st.warning("Vui lòng nhập API Key trước.")
        else:
            with st.spinner(f'Đang tải dữ liệu {symbol}...'):
                try:
                    df = get_data_safe(symbol)
                    live = get_live_price(symbol)
                    
                    if df is not None:
                        last = df.iloc[-1]
                        price = live if live else last['close']
                        
                        # Hiển thị chỉ số
                        m1, m2, m3, m4 = st.columns(4)
                        change = price - df.iloc[-2]['close']
                        pct = (change / df.iloc[-2]['close']) * 100
                        
                        m1.metric("Giá", f"{price:,.0f}", f"{change:,.0f} ({pct:.1f}%)")
                        m2.metric("RSI", f"{last['RSI']:.1f}")
                        m3.metric("MACD", "Tăng" if last['MACD'] > last['Signal_Line'] else "Giảm")
                        
                        vol_str = f"{last['Vol_Ratio']*100:.0f}%" if pd.notna(last['Vol_Ratio']) else "N/A"
                        m4.metric("Vol/TB20", vol_str)
                        
                        # Vẽ biểu đồ
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.tail(80)['time'], open=df.tail(80)['open'], high=df.tail(80)['high'], low=df.tail(80)['low'], close=df.tail(80)['close'], name="Giá"))
                        fig.add_trace(go.Scatter(x=df.tail(80)['time'], y=df.tail(80)['MA20'], line=dict(color='orange'), name="MA20"))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # AI Phân tích (Dùng model_name động, không fix cứng)
                        data_ctx = df.tail(60)[['time', 'close', 'RSI', 'MACD', 'Signal_Line']].to_string(index=False)
                        sys_prompt = f"Bạn là TA Alex. Giá {symbol}: {price}. Dữ liệu:\n{data_ctx}\n. Phân tích kỹ thuật ngắn gọn và đưa ra hành động Mua/Bán."
                        
                        # Cấu hình safety để tránh lỗi Empty Response của bản 3.0
                        safety_settings = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                        
                        try:
                            model = genai.GenerativeModel(model_name)
                            resp = model.generate_content(sys_prompt, safety_settings=safety_settings)
                            if resp.text: st.success(resp.text)
                            else: st.warning("AI không trả lời.")
                        except Exception as e:
                            st.error(f"Lỗi AI: {e}")
                            
                    else:
                        st.error(f"Không tìm thấy dữ liệu {symbol}.")
                except Exception as e:
                    st.error(f"Lỗi phân tích: {e}")

# === TAB 2: SCANNER ===
with tab2:
    st.header("🕵️ Máy Quét Cơ Hội")
    scan_list = st.text_area("Danh sách mã (cách nhau dấu phẩy):", value="ACB, FPT, HPG, MBB, MSN, MWG, SSI, STB, TCB, VHM, VIC, VNM, VPB")
    
    if st.button("🚀 Quét Thị Trường"):
        symbols = [s.strip().upper() for s in scan_list.split(",") if s.strip()]
        results = []
        progress_bar = st.progress(0)
        
        for i, sym in enumerate(symbols):
            try:
                # Dùng try-except để 1 mã lỗi không làm chết cả App
                df = get_data_safe(sym, days=150)
                if df is not None:
                    row = df.iloc[-1]
                    score = 0
                    if row['close'] > row['MA20']: score += 1
                    if row['MA20'] > row['MA50']: score += 1
                    if row['MACD'] > row['Signal_Line']: score += 1.5
                    if pd.notna(row['Vol_Ratio']) and row['Vol_Ratio'] > 1.2: score += 1.5
                    
                    rank = "Yếu"
                    if score >= 4: rank = "🔥 Khỏe"
                    elif score >= 2.5: rank = "😐 Trung tính"
                    
                    results.append({"Mã": sym, "Giá": row['close'], "Điểm": score, "Xếp loại": rank})
            except: pass
            progress_bar.progress((i + 1) / len(symbols))
            
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Điểm", ascending=False)
            st.dataframe(res_df, use_container_width=True)
            
            # AI chọn mã ngon nhất
            top = res_df.iloc[0]
            st.subheader(f"🏆 Alex chọn: {top['Mã']}")
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"Tại sao {top['Mã']} lại có điểm kỹ thuật cao nhất trong danh sách này? Giải thích ngắn."
                st.write(model.generate_content(prompt).text)
            except: pass
