import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from vnstock import stock_historical_data
from datetime import datetime, timedelta

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

def get_live_price_1m(symbol):
    try:
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

symbol = st.sidebar.text_input("Mã cổ phiếu", value="FPT").upper()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Chọn Model")

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
    model_name = st.sidebar.selectbox("Model khả dụng:", available_models, index=0)
    st.sidebar.success(f"🚀 Đang dùng: {model_name}")
else:
    model_name = st.sidebar.selectbox("Model:", ["gemini-2.0-flash-exp"], index=0)

st.sidebar.markdown("---")
show_ma20 = st.sidebar.checkbox("Đường MA20", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)

# --- 4. GIAO DIỆN CHÍNH ---
st.title(f"📈 Phân Tích: {symbol}")

if symbol and api_key:
    with st.spinner('Đang tải dữ liệu...'):
        df_daily = get_stock_data(symbol)
        live_price = get_live_price_1m(symbol)
        
        display_price = 0.0
        change_val = 0.0
        change_pct = 0.0
        
        if df_daily is not None:
            ref_price = df_daily.iloc[-2]['close'] if len(df_daily) > 1 else df_daily.iloc[-1]['close']
            if live_price:
                display_price = live_price
                st.success(f"⚡ Giá Realtime: {display_price:,.0f}")
            else:
                display_price = df_daily.iloc[-1]['close']
                st.warning("⚠️ Dùng giá đóng cửa gần nhất.")
            
            change_val = display_price - ref_price
            change_pct = (change_val / ref_price) * 100

    if df_daily is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giá", f"{display_price:,.0f}", f"{change_val:,.0f} ({change_pct:.2f}%)")
        c2.metric("RSI", f"{df_daily.iloc[-1]['RSI']:.1f}")
        c3.metric("Vol", f"{df_daily.iloc[-1]['volume'].mean():,.0f}")
        c4.metric("MA20", f"{df_daily.iloc[-1]['MA20']:.0f}")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_daily.tail(60)['time'],
            open=df_daily.tail(60)['open'], high=df_daily.tail(60)['high'],
            low=df_daily.tail(60)['low'], close=df_daily.tail(60)['close'], name="Giá"))
        
        if show_ma20: 
            fig.add_trace(go.Scatter(x=df_daily.tail(60)['time'], y=df_daily.tail(60)['MA20'], line=dict(color='orange'), name="MA20"))
        
        if show_bb:
             fig.add_trace(go.Scatter(x=df_daily.tail(60)['time'], y=df_daily.tail(60)['BB_Upper'], line=dict(color='gray', dash='dot'), name="Upper"))
             fig.add_trace(go.Scatter(x=df_daily.tail(60)['time'], y=df_daily.tail(60)['BB_Lower'], line=dict(color='gray', dash='dot'), name="Lower", fill='tonexty', fillcolor='rgba(200,200,200,0.1)'))
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, b=0, l=0, r=0))
        
        # --- ĐÂY LÀ DÒNG BỊ LỖI LÚC NÃY (Đã sửa) ---
        st.plotly_chart(fig, use_container_width=True)

        # Chatbot
        st.markdown("---")
        st.subheader(f"💬 Chat với {model_name}")
        
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Hỏi Alex..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            data_ctx = df_daily.tail(60)[['time', 'close', 'RSI', 'MA20', 'BB_Upper', 'BB_Lower']].to_string(index=False)
            
            sys_prompt = f"""
            Bạn là TA Alex. Model: {model_name}.
            Dữ liệu {symbol} (60 phiên):
            {data_ctx}
            Giá Realtime: {display_price}.
            User hỏi: {prompt}.
            Hãy phân tích xu hướng và đưa ra khuyến nghị Mua/Bán.
            """
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            try:
                model = genai.GenerativeModel(model_name)
                with st.spinner("Đang phân tích..."):
                    resp = model.generate_content(sys_prompt, safety_settings=safety_settings)
                    if resp.text:
                        st.chat_message("assistant").write(resp.text)
                        st.session_state.messages.append({"role": "assistant", "content": resp.text})
                    else:
                        st.error("AI không trả lời. Hãy thử model khác.")
            except Exception as e:
                st.error(f"Lỗi: {e}")
