import streamlit as st
import time

# --- 1. CẤU HÌNH TRANG (BẮT BUỘC DÒNG 1) ---
st.set_page_config(layout="wide", page_title="TA Alex 2026", page_icon="💎")

# --- 2. BỘ KHUNG BẢO VỆ (Safety Wrapper) ---
# Mọi lỗi nhập thư viện sẽ bị bắt ở đây thay vì làm trắng màn hình
try:
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
except Exception as e:
    st.error(f"❌ Lỗi nghiêm trọng khi khởi động: {e}")
    st.stop()

# --- 3. CẤU HÌNH API (5 KEYS CỦA BẠN) ---
API_KEY_POOL = [
    "AIzaSyAcIDpmFgBVzIlb41m1cz4BPlTCjKM9Hl0",
    "AIzaSyBC_V9ACvGCElaWQL5BILKQCv_ikBGcsHs", 
    "AIzaSyCFgTf678MHOoaOMmfV6y0uXLVrT2VwPV8",
    "AIzaSyBJhszyVcCesLBHlL2mfEP3Tx-ykMyA4_w",
    "AIzaSyA9S1V66bDs9UrnnVJKy_zDbxWQh6MMxtM"
]

# --- 4. CÁC HÀM XỬ LÝ (ĐƯỢC BỌC KỸ) ---

def call_ai_rotation(prompt):
    """Hàm gọi AI xoay tua Key + Model 2026"""
    models = ["gemini-3-flash-preview", "gemini-2.0-flash-exp"]
    
    for i, key in enumerate(API_KEY_POOL):
        for model_name in models:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)
                # Tắt bộ lọc an toàn để tránh lỗi trả về rỗng
                safety = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                response = model.generate_content(prompt, safety_settings=safety)
                
                if response.text:
                    return response.text, f"{model_name} (Key {i+1})"
            except Exception:
                continue # Lỗi thì thử cái tiếp theo, không báo lỗi để tránh rác màn hình
                
    return "❌ Hệ thống quá tải, không lấy được nhận định AI lúc này.", "Error"

def get_data_safe(symbol):
    """Hàm lấy dữ liệu chứng khoán an toàn"""
    try:
        # Lấy ngắn ngày thôi cho nhẹ (200 ngày)
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
        
        # Thử DNSE
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        
        # Nếu lỗi thử TCBS
        if df is None or df.empty:
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
            
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            # Tính toán chỉ báo (đơn giản hóa để tránh lỗi tính toán)
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # BB
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
            
            return df
        return None
    except Exception as e:
        return None

# --- 5. GIAO DIỆN CHÍNH ---
st.sidebar.title("💎 TA Alex 2026")
st.sidebar.success(f"✅ Đã nạp {len(API_KEY_POOL)} Key")

tab1, tab2 = st.tabs(["📊 Phân Tích", "🚀 Scanner"])

# === TAB 1: PHÂN TÍCH ===
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Mã cổ phiếu", value="MBB").upper()
    
    # Nút bấm được bọc trong try-except LỚN NHẤT
    if st.button("🔍 Phân Tích Ngay", type="primary"):
        status = st.status("🚀 Đang khởi động...", expanded=True)
        try:
            # BƯỚC 1: TẢI DATA
            status.write("1️⃣ Đang kết nối dữ liệu máy chủ...")
            df = get_data_safe(symbol)
            
            if df is None or df.empty:
                status.update(label="❌ Lỗi dữ liệu!", state="error")
                st.error(f"Không tìm thấy dữ liệu cho mã **{symbol}**. Có thể do sàn chưa mở cửa hoặc mã sai.")
            
            else:
                status.write("✅ Tải xong dữ liệu.")
                last = df.iloc[-1]
                
                # BƯỚC 2: HIỂN THỊ
                status.write("2️⃣ Đang vẽ biểu đồ...")
                
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Giá", f"{last['close']:,.0f}")
                c2.metric("RSI", f"{last['RSI']:.1f}")
                c3.metric("MACD", f"{last['MACD']:.2f}")
                c4.metric("Vol", f"{last['volume']:,.0f}")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Giá"))
                fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], line=dict(color='orange'), name="MA20"))
                st.plotly_chart(fig, use_container_width=True)
                
                # BƯỚC 3: GỌI AI
                status.write("3️⃣ Đang gọi chuyên gia Alex (AI)...")
                data_text = df.tail(30).to_string()
                prompt = f"Giá {symbol}: {last['close']}. Dữ liệu:\n{data_text}\n. Phân tích xu hướng ngắn gọn."
                
                ai_text, model_used = call_ai_rotation(prompt)
                
                st.info(f"🤖 Nhận định từ **{model_used}**:")
                st.write(ai_text)
                
                status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

        except Exception as e:
            # BẮT MỌI LỖI SẬP NGUỒN TẠI ĐÂY
            status.update(label="❌ HỆ THỐNG GẶP LỖI", state="error")
            st.error(f"⚠️ Phát hiện lỗi lạ: {str(e)}")
            st.code("Gợi ý: Hãy thử tải lại trang (F5) hoặc đổi mã cổ phiếu khác.")

# === TAB 2: SCANNER ===
with tab2:
    st.header("🕵️ Quét Cổ Phiếu")
    if st.button("🚀 Quét Nhanh (Demo 5 mã)"):
        stocks = ["HPG", "SSI", "STB", "FPT", "MWG"]
        res = []
        bar = st.progress(0)
        
        for i, s in enumerate(stocks):
            try:
                d = get_data_safe(s)
                if d is not None:
                    r = d.iloc[-1]
                    score = 0
                    if r['close'] > r['MA20']: score += 1
                    if r['RSI'] > 50: score += 1
                    res.append({"Mã": s, "Giá": r['close'], "Điểm": score})
            except: pass
            bar.progress((i+1)/5)
            
        if res:
            st.dataframe(pd.DataFrame(res))
