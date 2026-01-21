import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from vnstock import stock_historical_data
from datetime import datetime, timedelta
import time

# --- 1. CẤU HÌNH ---
st.set_page_config(layout="wide", page_title="TA Alex Stock Advisor", page_icon="📈")

# --- 2. HÀM DỮ LIỆU ---
def get_stock_data(symbol, days=365):
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df['MA20'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['MA20'] + (std * 2)
            df['BB_Lower'] = df['MA20'] - (std * 2)
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            return df
        return None
    except: return None

def get_live_price(symbol):
    try:
        # Lấy giá realtime bằng nến phút
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1m', type='stock', source='DNSE')
        if df is not None and not df.empty:
            return float(df.iloc[-1]['close'])
        return None
    except: return None

# --- 3. SIDEBAR ---
st.sidebar.title("⚙️ Cấu hình")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ Bản quyền: Đã kích hoạt")
else:
    api_key = st.sidebar.text_input("Nhập Gemini API Key", type="password")

# Chọn Model
available_models = []
if api_key:
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                if "1.0" not in name and "1.5" not in name: 
                    available_models.append(name)
    except: pass

if available_models:
    available_models.sort(key=lambda x: ('3' not in x, 'flash' not in x))
    model_name = st.sidebar.selectbox("Model:", available_models, index=0)
else:
    model_name = st.sidebar.selectbox("Model:", ["gemini-2.0-flash-exp"], index=0)

# --- 4. GIAO DIỆN CHÍNH (TABS) ---
st.title("📈 TA Alex Stock Advisor")
tab1, tab2 = st.tabs(["🔍 Phân Tích Chi Tiết", "🚀 Tìm Cổ Phiếu (Scanner)"])

# === TAB 1: PHÂN TÍCH 1 MÃ (CŨ) ===
with tab1:
    symbol = st.text_input("Nhập mã cổ phiếu (VD: FPT)", value="FPT").upper()
    show_ma20 = st.checkbox("MA20", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=True)

    if symbol and api_key:
        if st.button("Phân tích ngay", type="primary"):
            with st.spinner('Đang tải dữ liệu...'):
                df_daily = get_stock_data(symbol)
                live_price = get_live_price(symbol)
                
                display_price = 0.0
                if df_daily is not None:
                    if live_price: display_price = live_price
                    else: display_price = df_daily.iloc[-1]['close']
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df_daily.tail(60)['time'],
                        open=df_daily.tail(60)['open'], high=df_daily.tail(60)['high'],
                        low=df_daily.tail(60)['low'], close=df_daily.tail(60)['close'], name="Giá"))
                    if show_ma20: fig.add_trace(go.Scatter(x=df_daily.tail(60)['time'], y=df_daily.tail(60)['MA20'], line=dict(color='orange'), name="MA20"))
                    if show_bb:
                        fig.add_trace(go.Scatter(x=df_daily.tail(60)['time'], y=df_daily.tail(60)['BB_Upper'], line=dict(color='gray', dash='dot'), name="Upper"))
                        fig.add_trace(go.Scatter(x=df_daily.tail(60)['time'], y=df_daily.tail(60)['BB_Lower'], line=dict(color='gray', dash='dot'), name="Lower", fill='tonexty', fillcolor='rgba(200,200,200,0.1)'))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI Analysis
                    data_ctx = df_daily.tail(60)[['time', 'close', 'RSI', 'MA20', 'BB_Upper', 'BB_Lower']].to_string(index=False)
                    sys_prompt = f"Bạn là TA Alex. Model: {model_name}. Giá {symbol} hiện tại: {display_price}. Dữ liệu quá khứ:\n{data_ctx}\n. Hãy phân tích xu hướng, RSI, Bollinger Bands và đưa ra khuyến nghị MUA/BÁN ngắn gọn."
                    
                    try:
                        model = genai.GenerativeModel(model_name)
                        with st.spinner("Alex đang suy nghĩ..."):
                            resp = model.generate_content(sys_prompt)
                            st.info(resp.text)
                    except Exception as e: st.error(f"Lỗi AI: {e}")

# === TAB 2: QUÉT THỊ TRƯỜNG (MỚI) ===
with tab2:
    st.header("🚀 Sàng Lọc Cơ Hội Đầu Tư")
    st.caption("Nhập danh sách mã bạn muốn quét (cách nhau dấu phẩy). Mẹo: Nhập khoảng 10-20 mã VN30.")
    
    # Danh sách mặc định là một vài mã Hot
    default_list = "ACB, FPT, HPG, MBB, MSN, MWG, PNJ, SSI, STB, TCB, TPB, VHM, VIC, VNM, VPB, VRE"
    scan_list_text = st.text_area("Danh sách mã:", value=default_list)
    
    if st.button("🔍 Quét ngay (Tìm mã MUA)", key="scan_btn"):
        symbols = [s.strip().upper() for s in scan_list_text.split(",") if s.strip()]
        
        scan_results = []
        progress_bar = st.progress(0)
        
        with st.spinner(f"Đang quét {len(symbols)} cổ phiếu... (Vui lòng đợi)"):
            for i, sym in enumerate(symbols):
                df = get_stock_data(sym, days=100) # Lấy 100 ngày để tính chỉ báo
                if df is not None:
                    last_row = df.iloc[-1]
                    # Logic chấm điểm đơn giản của Code (Sơ loại)
                    trend = "Tăng" if last_row['close'] > last_row['MA20'] else "Giảm"
                    rsi = last_row['RSI']
                    
                    # Tìm điểm mua tiềm năng (RSI thấp hoặc vừa cắt lên MA20)
                    signal = "Theo dõi"
                    if rsi < 35: signal = "Bắt đáy (RSI thấp)"
                    elif trend == "Tăng" and 40 < rsi < 60: signal = "Mua Trend (An toàn)"
                    elif rsi > 75: signal = "Quá mua (Cẩn thận)"
                    
                    scan_results.append({
                        "Mã": sym,
                        "Giá": last_row['close'],
                        "RSI": round(rsi, 1),
                        "Xu hướng (MA20)": trend,
                        "Tín hiệu thô": signal
                    })
                # Cập nhật thanh tiến trình
                progress_bar.progress((i + 1) / len(symbols))
                
        # Hiển thị bảng kết quả
        if scan_results:
            results_df = pd.DataFrame(scan_results)
            st.dataframe(results_df.style.apply(lambda x: ['background-color: #d4edda' if 'Mua' in v or 'Bắt đáy' in v else '' for v in x], subset=['Tín hiệu thô']), use_container_width=True)
            
            # --- AI CHỌN LỌC ---
            st.markdown("---")
            st.subheader("🤖 Alex Chọn Mã Nào?")
            
            # Chỉ gửi Top 5 mã tiềm năng nhất cho AI để tiết kiệm Token
            potential_stocks = results_df[results_df['Tín hiệu thô'].str.contains("Mua|Bắt đáy")].head(5)
            
            if not potential_stocks.empty:
                data_for_ai = potential_stocks.to_string(index=False)
                ai_prompt = f"""
                Tôi có danh sách các cổ phiếu tiềm năng sau đây (đã lọc thô):
                {data_for_ai}
                
                Với tư cách là chuyên gia TA Alex, hãy:
                1. Chọn ra ĐÚNG 1 MÃ bạn thấy đẹp nhất để MUA ngay lúc này.
                2. Giải thích ngắn gọn tại sao (dựa trên RSI và Xu hướng).
                3. Đưa ra giá cắt lỗ dự kiến.
                """
                
                try:
                    model = genai.GenerativeModel(model_name)
                    with st.spinner("Alex đang so sánh để tìm 'Hoa Hậu'..."):
                        resp = model.generate_content(ai_prompt)
                        st.success(resp.text)
                except Exception as e: st.error(f"Lỗi AI: {e}")
            else:
                st.warning("Không tìm thấy mã nào có điểm mua đẹp theo bộ lọc thô. Hãy thử danh sách khác!")
        else:
            st.error("Không tải được dữ liệu. Vui lòng kiểm tra lại danh sách mã.")
