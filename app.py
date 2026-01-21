import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from vnstock import stock_historical_data
from datetime import datetime, timedelta

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(layout="wide", page_title="TA Alex Stock Advisor", page_icon="📈")

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---
def get_stock_data(symbol, days=365):
    # Lấy dữ liệu D1 (Ngày) để vẽ biểu đồ tổng quan
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        # Dùng source='DNSE' để ổn định
        df = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date, resolution='1D', type='stock', source='DNSE')
        
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            
            # Chỉ báo MA20 & Bollinger Bands
            df['MA20'] = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['MA20'] + (std_dev * 2)
            df['BB_Lower'] = df['MA20'] - (std_dev * 2)
            
            # Chỉ báo RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df
        return None
    except: return None

def get_live_price_1m(symbol):
    # KỸ THUẬT: Lấy nến 1 phút để có giá Realtime ngay tức thì
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        # resolution='1m' -> Lấy chi tiết từng phút
        df_minute = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1m', type='stock', source='DNSE')
        
        if df_minute is not None and not df_minute.empty:
            latest = df_minute.iloc[-1]
            return float(latest['close'])
        return None
    except: return None

# --- 3. SIDEBAR THÔNG MINH ---
st.sidebar.title("⚙️ Cấu hình")

# LOGIC MỚI: Ưu tiên lấy Key từ Secrets (để bạn bè dùng luôn)
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ Đã kích hoạt bản quyền của Alex")
else:
    # Nếu không có Secrets thì hiện ô nhập như cũ
    api_key = st.sidebar.text_input("Nhập Gemini API Key", type="password")

symbol = st.sidebar.text_input("Mã cổ phiếu (VD: FPT)", value="FPT").upper()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Chọn Model")

# Auto-detect Model
available_models = []
if api_key:
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                available_models.append(name)
    except: pass

if available_models:
    # Ưu tiên đưa model 3.0 hoặc flash lên đầu danh sách
    available_models.sort(key=lambda x: ('3' not in x, 'flash' not in x))
    model_name = st.sidebar.selectbox("Model:", available_models, index=0)
    st.sidebar.success(f"✅ Đang chọn: {model_name}")
else:
    model_name = st.sidebar.selectbox("Model:", ["Đang chờ kết nối..."], disabled=True)

st.sidebar.markdown("---")
show_ma20 = st.sidebar.checkbox("Đường MA20", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

# --- 4. GIAO DIỆN CHÍNH ---
st.title(f"📈 Phân Tích: {symbol}")

if symbol and api_key:
    with st.spinner('Đang kết nối dữ liệu trực tiếp...'):
        # 1. Lấy lịch sử ngày (D1)
        df_daily = get_stock_data(symbol)
        
        # 2. Lấy giá Live (1 phút)
        live_price = get_live_price_1m(symbol)
        
        # Xử lý hiển thị giá
        display_price = 0.0
        change_val = 0.0
        change_pct = 0.0
        
        if df_daily is not None:
            # Giá tham chiếu = Giá đóng cửa phiên trước
            # (Logic: Lấy cây áp chót nếu cây cuối là hôm nay)
            ref_price = df_daily.iloc[-2]['close'] if len(df_daily) > 1 else df_daily.iloc[-1]['close']
            
            if live_price:
                display_price = live_price
                st.success(f"⚡ Đã lấy được giá Realtime: {display_price:,.0f}")
            else:
                display_price = df_daily.iloc[-1]['close']
                st.warning("⚠️ Không lấy được tick phút, dùng giá đóng cửa phiên gần nhất.")
                
            change_val = display_price - ref_price
            change_pct = (change_val / ref_price) * 100

    if df_daily is not None:
        # Dashboard Chỉ số
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giá Hiện Tại", f"{display_price:,.0f}", f"{change_val:,.0f} ({change_pct:.2f}%)")
        c2.metric("RSI (14)", f"{df_daily.iloc[-1]['RSI']:.1f}")
        c3.metric("Vol (TB 20p)", f"{df_daily.iloc[-1]['volume'].mean():,.0f}")
        c4.metric("MA20", f"{df_daily.iloc[-1]['MA20']:.0f}")

        # Vẽ Biểu đồ (Plotly)
        fig = go.Figure()
        
        # 1. Nến Nhật
        fig.add_trace(go.Candlestick(
            x=df_daily.tail(60)['time'],
            open=df_daily.tail(60)['open'], high=df_daily.tail(60)['high'],
            low=df_daily.tail(60)['low'], close=df_daily.tail(60)['close'],
            name="Giá"
        ))
        
        # 2. Đường MA20
        if show_ma20: 
            fig.add_trace(go.Scatter(
                x=df_daily.tail(60)['time'], y=df_daily.tail(60)['MA20'], 
                line=dict(color='orange'), name="MA20"
            ))
            
        # 3. Bollinger Bands (Chia dòng để tránh lỗi copy)
        if show_bb:
             fig.add_trace(go.Scatter(
                 x=df_daily.tail(60)['time'], y=df_daily.tail(60)['BB_Upper'], 
                 line=dict(color='gray', dash='dot'), name="Upper"
             ))
             fig.add_trace(go.Scatter(
                 x=df_daily.tail(60)['time'], y=df_daily.tail(60)['BB_Lower'], 
                 line=dict(color='gray', dash='dot'), name="Lower", 
                 fill='tonexty', fillcolor='rgba(200,200,200,0.1)'
             ))
             
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        # Chatbot AI
        st.markdown("---")
        st.subheader(f"💬 Chat với {model_name}")
        
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Hỏi TA Alex..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # --- GỬI DỮ LIỆU ---
            # 1. Gửi giá Realtime chính xác
            # 2. Gửi 60 phiên (3 tháng) để AI nhìn mô hình
            data_ctx = df_daily.tail(60)[['time', 'close', 'RSI', 'MA20', 'BB_Upper', 'BB_Lower']].to_string(index=False)
            
            sys_prompt = f"""
            Bạn là "TA Alex" - Chuyên gia Swing Trading.
            Model: {model_name}.
            
            DỮ LIỆU THỊ TRƯỜNG:
            - Giá Realtime lúc này: {display_price} (Hãy dùng giá này để khuyến nghị).
            - Xu hướng 60 phiên gần nhất (để soi mô hình giá):
            {data_ctx}
            
            YÊU CẦU:
            1. Phân tích xu hướng (Trend) và Mô hình giá (Pattern).
            2. Đánh giá rủi ro dựa trên RSI và Bollinger Bands.
            3. Đưa ra KẾT LUẬN hành động: MUA / BÁN / GIỮ.
            
            Câu hỏi của user: {prompt}
            """
            
            # Cấu hình phá bộ lọc an toàn (để không bị chặn lời khuyên tài chính)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            try:
                if model_name and model_name != "Đang chờ kết nối...":
                    model = genai.GenerativeModel(model_name)
                    with st.spinner(f"Alex đang soi chart giá {display_price:,.0f}..."):
                        resp = model.generate_content(sys_prompt, safety_settings=safety_settings)
                        if resp.text:
                            st.chat_message("assistant").write(resp.text)
                            st.session_state.messages.append({"role": "assistant", "content": resp.text})
                        else: st.error("AI không phản hồi. Hãy thử lại.")
            except Exception as e: st.error(f"Lỗi: {e}")
