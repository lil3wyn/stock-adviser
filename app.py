import streamlit as st
import time

# --- 1. KHỞI ĐỘNG (LUÔN Ở DÒNG ĐẦU) ---
st.set_page_config(layout="wide", page_title="TA Alex Safe", page_icon="🛡️")

# Vùng thông báo trạng thái
status = st.container()

# --- 2. NẠP THƯ VIỆN (CÓ BẮT LỖI) ---
try:
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
except Exception as e:
    st.error(f"❌ Lỗi nạp thư viện: {e}")
    st.stop()

# --- 3. HÀM XỬ LÝ (TẮT CACHE ĐỂ TRÁNH LỖI) ---
def get_data_debug(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Thử DNSE
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        
        # Nếu lỗi thử TCBS
        if df is None or df.empty:
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
            
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            
            # Tính chỉ báo cơ bản (Tránh tính toán phức tạp gây lỗi)
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
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None

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
st.sidebar.title("🛡️ TA Alex Safe Mode")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ Đã có Key")
else:
    api_key = st.sidebar.text_input("API Key", type="password")

# Ép dùng Model mới nhất để tránh lỗi 404
model_name = "gemini-2.0-flash-exp"
st.sidebar.info(f"🤖 Đang dùng: {model_name}")

# --- 5. TABS ---
tab1, tab2 = st.tabs(["🔍 Phân Tích", "🚀 Scanner"])

# === TAB 1: PHÂN TÍCH ===
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã CP", value="MBB").upper()
    
    if st.button("Kiểm Tra Ngay", type="primary"):
        log_box = st.expander("📝 Nhật ký chạy (Xem nếu lỗi)", expanded=True)
        
        if not api_key:
            st.warning("⚠️ Chưa nhập Key!")
        else:
            try:
                log_box.write("1️⃣ Đang tải dữ liệu...")
                df = get_data_debug(symbol)
                live = get_live_price(symbol)
                
                if df is not None:
                    log_box.write(f"✅ Đã tải {len(df)} dòng dữ liệu.")
                    last = df.iloc[-1]
                    price = live if live else last['close']
                    
                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Giá", f"{price:,.0f}")
                    c2.metric("RSI", f"{last['RSI']:.1f}")
                    c3.metric("MACD", f"{last['MACD']:.2f}")
                    c4.metric("Vol", f"{last['volume']:,.0f}")
                    
                    log_box.write("2️⃣ Đang vẽ biểu đồ...")
                    # Vẽ biểu đồ ĐƠN GIẢN NHẤT để tránh sập
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.tail(60)['time'], open=df.tail(60)['open'], high=df.tail(60)['high'], low=df.tail(60)['low'], close=df.tail(60)['close'], name="Giá"))
                    st.plotly_chart(fig, use_container_width=True)
                    log_box.write("✅ Vẽ xong.")
                    
                    log_box.write("3️⃣ Đang gọi AI...")
                    # AI Call (Try-Except chặt chẽ)
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_name)
                        
                        data_ctx = df.tail(30)[['time', 'close', 'RSI', 'MACD']].to_string(index=False)
                        prompt = f"Giá {symbol}: {price}. Dữ liệu:\n{data_ctx}\n. Phân tích Mua/Bán ngắn gọn."
                        
                        resp = model.generate_content(prompt)
                        if resp.text:
                            st.success("🤖 AI Nhận định:")
                            st.write(resp.text)
                            log_box.write("✅ AI Xong.")
                    except Exception as e_ai:
                        st.error(f"⚠️ Lỗi AI: {e_ai}")
                        log_box.write(f"❌ AI Chết: {e_ai}")

                else:
                    st.error("❌ Không lấy được dữ liệu. Kiểm tra lại mã.")
            except Exception as e_main:
                st.error(f"❌ LỖI SẬP NGUỒN: {e_main}")
                st.exception(e_main) # Hiện chi tiết lỗi code

# === TAB 2: SCANNER ===
with tab2:
    st.header("🕵️ Scanner (Chế độ An toàn)")
    if st.button("🚀 Quét Thử 5 Mã HOT"):
        list_stocks = ["FPT", "HPG", "SSI", "STB", "MBB"]
        res = []
        bar = st.progress(0)
        
        for i, s in enumerate(list_stocks):
            try:
                d = get_data_debug(s)
                if d is not None:
                    r = d.iloc[-1]
                    sc = 0
                    if r['close'] > r['MA20']: sc += 1
                    res.append({"Mã": s, "Giá": r['close'], "Điểm": sc})
            except: pass
            bar.progress((i+1)/5)
            
        if res:
            st.dataframe(pd.DataFrame(res))
        else:
            st.warning("Không quét được.")
