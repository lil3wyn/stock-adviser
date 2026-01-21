import streamlit as st

# --- 1. CẤU HÌNH TRANG (LUÔN ĐỂ DÒNG 1) ---
st.set_page_config(layout="wide", page_title="TA Alex Debug", page_icon="🛠️")

st.title("🛠️ Chế độ Sửa Lỗi (Debug Mode)")
st.caption("Nếu gặp lỗi, nó sẽ hiện ra chi tiết ở dưới thay vì trắng xóa màn hình.")

# --- 2. NẠP THƯ VIỆN AN TOÀN ---
status = st.empty()
try:
    status.info("⏳ Đang nạp thư viện...")
    import pandas as pd
    import plotly.graph_objects as go
    import google.generativeai as genai
    from vnstock import stock_historical_data
    from datetime import datetime, timedelta
    status.success("✅ Nạp thư viện thành công!")
    import time
    time.sleep(1)
    status.empty()
except Exception as e:
    st.error(f"❌ Lỗi nạp thư viện: {e}")
    st.stop()

# --- 3. HÀM XỬ LÝ (CÓ BẮT LỖI) ---
def get_data_safe(symbol):
    try:
        # Lấy thử 100 ngày (ngắn thôi cho nhẹ)
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        
        # Thử nguồn DNSE
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='DNSE')
        
        if df is None or df.empty:
            # Nếu DNSE lỗi, thử nguồn TCBS dự phòng
            st.warning("⚠️ Nguồn DNSE không trả về dữ liệu, đang thử TCBS...")
            df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock', source='TCBS')
            
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            # Tính toán nhẹ
            df['MA20'] = df['close'].rolling(window=20).mean()
            return df
        return None
    except Exception as e:
        st.error(f"Lỗi hàm get_data: {e}")
        return None

# --- 4. GIAO DIỆN CHÍNH ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

symbol = st.text_input("Nhập mã cổ phiếu:", value="FPT").upper()

if st.button("🚀 BẮT ĐẦU PHÂN TÍCH"):
    debug_box = st.expander("Xem nhật ký chạy (Logs)", expanded=True)
    
    # --- BƯỚC 1 ---
    debug_box.write("1️⃣ Bắt đầu tải dữ liệu...")
    try:
        df = get_data_safe(symbol)
        if df is None:
            st.error("❌ Không tải được dữ liệu. Kiểm tra lại mã cổ phiếu hoặc nguồn dữ liệu.")
            st.stop()
        debug_box.write(f"✅ Đã tải được {len(df)} dòng dữ liệu.")
    except Exception as e:
        st.error(f"❌ Chết ở Bước 1: {e}")
        st.stop()
        
    # --- BƯỚC 2 ---
    debug_box.write("2️⃣ Đang vẽ biểu đồ...")
    try:
        price = df.iloc[-1]['close']
        st.metric("Giá hiện tại", f"{price:,.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Giá"))
        st.plotly_chart(fig, use_container_width=True)
        debug_box.write("✅ Vẽ biểu đồ xong.")
    except Exception as e:
        st.error(f"❌ Chết ở Bước 2 (Vẽ hình): {e}")
        st.stop()

    # --- BƯỚC 3 ---
    debug_box.write("3️⃣ Đang gọi AI (Gemini)...")
    if not api_key:
        st.warning("⚠️ Chưa có API Key nên bỏ qua bước AI.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash") # Dùng bản ổn định nhất để test
            
            # Kiểm tra xem model có sống không
            try:
                debug_box.write("...Đang thử kết nối Google...")
                models = list(genai.list_models())
                debug_box.write("✅ Kết nối Google OK.")
            except:
                st.warning("⚠️ Key sai hoặc Google chặn kết nối.")
            
            # Gửi Prompt
            prompt = f"Phân tích ngắn gọn xu hướng giá cổ phiếu {symbol} giá {price}."
            resp = model.generate_content(prompt)
            
            if resp.text:
                st.success("🤖 AI Trả lời:")
                st.write(resp.text)
                debug_box.write("✅ AI chạy xong.")
            else:
                st.error("AI trả về rỗng.")
                
        except Exception as e:
            # Quan trọng: Bắt lỗi API mà không làm sập App
            st.error(f"❌ Chết ở Bước 3 (AI): {e}")
            debug_box.write("Gợi ý: Nếu lỗi 404/400 thì đổi Model khác.")

    st.balloons()
