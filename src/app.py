# app.py
"""
Ứng dụng Streamlit:
- Nhận câu tiếng Việt từ người dùng
- Gọi NLP để phân loại cảm xúc
- Lưu lịch sử vào SQLite
- Hiển thị lịch sử phân loại
"""

import streamlit as st

from utils.db import init_db, save_result, get_history
from utils.nlp import classify

# Khởi tạo DB ngay khi run app
init_db()

# State cho lịch sử
if "history_limit" not in st.session_state:
    st.session_state["history_limit"] = 10  # mặc định 10 bản ghi gần nhất

if "history_filter" not in st.session_state:
    st.session_state["history_filter"] = "ALL"  # ALL / POSITIVE / NEGATIVE / NEUTRAL
    
# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Trợ lý phân loại cảm xúc Tiếng Việt",
    page_icon="💬",
    layout="centered",
)

# Tiêu đề & mô tả
st.title("Trợ lý phân loại cảm xúc Tiếng Việt")
st.write(
    "Nhập một câu tiếng Việt bất kỳ. Ứng dụng sẽ phân loại cảm xúc thành "
    "`POSITIVE`, `NEUTRAL` hoặc `NEGATIVE`."
)

# Phần nhập liệu & phân loại
st.markdown("---")
st.subheader("Input")
# Nhập liệu
user_text = st.text_input(
    "Nhập câu tiếng Việt:",
    placeholder="Ví dụ: Hôm nay tôi rất vui...",
)

col1, col2 = st.columns([1, 1])

with col1:
    classify_btn = st.button("Phân loại cảm xúc")

color_map = {
    "POSITIVE": ("TÍCH CỰC", "green"),
    "NEGATIVE": ("TIÊU CỰC", "red"),
    "NEUTRAL": ("TRUNG TÍNH", "gold"),
}

# Kết quả phân loại
if classify_btn:
    if not user_text or len(user_text.strip()) == 0:
        st.error("Không được để trống. Vui lòng nhập câu tiếng Việt.")
    elif len(user_text.strip()) < 5:
        st.warning("Vui lòng nhập câu rõ nghĩa hơn (>= 5 ký tự).")
    else:
        with st.spinner("Đang phân tích cảm xúc..."):
            try:
                result = classify(user_text)

                original_text = result["original_text"]
                normalized_text = result["normalized_text"]
                sentiment = result["sentiment"]
                score = result["score"]

                # Lưu vào DB (lưu theo câu gốc)
                save_result(original_text, sentiment)

                # ===== Hiển thị câu gốc & câu chuẩn hoá =====
                st.subheader("Kết quả phân loại cảm xúc")
                st.write("* **Câu gốc (Input):** ", original_text)
                st.write("* **Câu đã được chuẩn hóa:** ", normalized_text)

                # ===== Hiển thị cảm xúc theo màu =====
                st.markdown("* **Cảm xúc (sentiment):**")
                label_vi, color = color_map.get(
                    sentiment, ("KHÔNG RÕ","gray")
                )
                st.markdown(
                    f"""
                    <div style='padding:8px 12px;border-radius:8px;border:1px solid {color};'>
                        <h3 style='color:{color};margin:0;'><strong>{label_vi}</strong></h3>
                        <p style='margin:4px 0;'>Độ tin cậy: <b>{score:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Hiển thị đúng kiểu "dictionary 2 trường" như yêu cầu
                st.markdown("* **Kết quả dạng dictionary (lưu trong lịch sử):**")
                st.json({
                    "text": normalized_text,
                    "sentiment": sentiment
                })

            except ValueError as e:
                # Lỗi do mình chủ động raise (câu vô nghĩa / không phải tiếng Việt)
                st.error(f"❗ {e}")
            except Exception as e:
                # Lỗi kỹ thuật khác
                st.error(f"Đã xảy ra lỗi kỹ thuật khi phân loại: {e}")

st.markdown("---")
st.subheader("Lịch sử phân loại cảm xúc")

# --- Bộ lọc + nút tải thêm ---
col_filter, col_info, col_more = st.columns([2, 1, 1])

with col_filter:
    filter_label = st.selectbox(
        "Lọc theo nhãn:",
        options=["Tất cả", "Positive", "Neutral", "Negative"],
        index=0,
    )

filter_map = {
    "Tất cả": "ALL",
    "Positive": "POSITIVE",
    "Neutral": "NEUTRAL",
    "Negative": "NEGATIVE",
}
st.session_state["history_filter"] = filter_map[filter_label]


# Xác định sentiment filter thật gửi xuống DB
sentiment_filter = (
    None if st.session_state["history_filter"] == "ALL"
    else st.session_state["history_filter"]
)

# Lấy lịch sử từ DB
history = get_history(
    limit=st.session_state["history_limit"],
    sentiment=sentiment_filter,
)

def load_more():
    st.session_state["history_limit"] += st.session_state.get("history_increment", 10)

if not history:
    st.info("Chưa có lịch sử nào khớp với bộ lọc hiện tại.")
else:
    for item in history:
        text = item["text"]
        sentiment = item["sentiment"]
        timestamp = item["timestamp"]

        label_vi, color = color_map.get(sentiment, ("KHÔNG RÕ", "gray"))

        st.markdown(
            f"""
            <div style='padding:8px 12px;border-radius:8px;border:1px solid {color};margin: 8px 0;'>
                    <h3 style='color:{color};margin:0;'><strong>{label_vi}</strong></h3>
                    <p style='margin:4px 0;'>Text: {text}</p>
                    <p style='margin:4px 0;'>Sentiment: {sentiment}</p>
                    <p style='margin:4px 0;font-size:0.9em;color:gray;'>Thời gian: {timestamp}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if len(history) >= st.session_state["history_limit"]:
        if "history_increment" not in st.session_state:
            st.session_state["history_increment"] = 10  
        if st.button("Tải thêm", on_click=load_more):
            st.session_state["history_limit"] += st.session_state["history_increment"]
