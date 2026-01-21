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
