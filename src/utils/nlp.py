# nlp.py
"""
Module xử lý NLP:
- Chuẩn hoá câu tiếng Việt
- Gọi mô hình Transformer để phân loại cảm xúc
"""
import streamlit as st
from underthesea import word_tokenize
from transformers import pipeline

# CẤU HÌNH MODEL
SELECTED_MODEL_TYPE = "PHOBERT"

# Dictionary ánh xạ từ Yêu cầu đồ án -> Model thực tế trên Hugging Face
MODEL_MAPPING = {
    # Lựa chọn 1: phobert-base-v2 
    "PHOBERT": {
        "name_in_report": "phobert-base-v2", 
        "hf_id": "duchienmtp/PhoBERT-sentiment-analysis"
    },
    # Lựa chọn 2: distilbert-base-multilingual-cased 
    "DISTILBERT": {
        "name_in_report": "distilbert-base-multilingual-cased",
        "hf_id": "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    }
}

# Ánh xạ nhãn cảm xúc từ mô hình sang nhãn chuẩn
LABEL_MAPPING = {
    # (Chữ đầy đủ)
    "NEGATIVE": "NEGATIVE",
    "POSITIVE": "POSITIVE",
    "NEUTRAL": "NEUTRAL",
    
    # 2. Nhóm viết thường 
    "negative": "NEGATIVE",
    "positive": "POSITIVE",
    "neutral": "NEUTRAL",

    # 3. Nhóm viết tắt 
    "NEG": "NEGATIVE",
    "POS": "POSITIVE",
    "NEU": "NEUTRAL",
    
    # 4. Nhóm Label số 
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}

@st.cache_resource
def get_sentiment_pipeline():
    """
    Chỉ load model 1 lần duy nhất và chia sẻ tài nguyên cho toàn bộ app.
    Giúp ứng dụng không bị chậm khi F5 hoặc tương tác.
    """
    # Lấy ID model từ cấu hình
    model_info = MODEL_MAPPING[SELECTED_MODEL_TYPE]
    MODEL_NAME = model_info['hf_id']
    
    print(f"--- ĐANG TẢI MODEL: {model_info['name_in_report']} ---")
    
    # Khởi tạo pipeline
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )
    return sentiment_pipeline

# Tập kí tự nguyên âm tiếng Việt
VIET_VOWELS = set(
    "aeiouyAEIOUY"
    "ăâêôơưĂÂÊÔƠƯ"
    "áàảãạÁÀẢÃẠ"
    "ắằẳẵặẮẰẲẴẶ"
    "ấầẩẫậẤẦẨẪẬ"
    "éèẻẽẹÉÈẺẼẸ"
    "óòỏõọÓÒỎÕỌ"
    "ốồổỗộỐỒỔỖỘ"
    "ớờởỡợỚỜỞỠỢ"
    "íìỉĩịÍÌỈĨỊ"
    "úùủũụÚÙỦŨỤ"
    "ýỳỷỹỵÝỲỶỸỴ"
    "đĐ"
)

# Tập stopwords tiếng Việt phổ biến
VIET_STOPWORDS = {
    "là", "và", "của", "không", "rất", "này", "kia", "đó", "này",
    "tôi", "ban", "bạn", "mình", "anh", "em", "chỉ", "thì",
    "nhưng", "nếu", "vì", "nên", "cho", "khi", "đã", "đang", "sẽ",
    "ở", "trong", "trên", "với", "hay", "cũng", "rồi", "luôn",
}

# Từ điển chuẩn hoá viết tắt / không dấu thường gặp trong tiếng Việt
VIET_NORMALIZATION_DICT = {
    # Viết tắt
        "khong": "không",
        "k": "không",
        "ko": "không",
        "k0": "không",
        "hok": "không",
        "khg": "không",
        "hk": "không",
        "kh": "không",
        
        # Câu hỏi
        "gi": "gì",
        "j": "gì",
        "sao": "sao",
        "dc": "được",
    
    # Tiếng Việt không dấu
    # Đại từ – Ngôi xưng
        "toi": "tôi",
        "ban": "bạn",
        "minh": "mình",
        "anh": "anh",
        "chi": "chị",
        "em": "em",
        "co": "cô",
        "chu": "chú",
        "ba": "bà",
        "ong": "ông",
        "nguoi": "người",
        "ho": "họ",
    
    # Động từ – tính từ phổ biến liên quan đến cảm xúc
        "yeu": "yêu",
        "thuong": "thương",
        "ghet": "ghét",
        "thich": "thích",
        "biet": "biết",
        "hieu": "hiểu",
        "thay": "thấy",
        "khoe": "khỏe",
        "om": "ốm",
        "dau": "đau",
        "met": "mệt",
        "vui": "vui",
        "buon": "buồn",
        "gian": "giận",
        "nong": "nóng",
        "lanh": "lạnh",
        "dep": "đẹp",
        "xau": "xấu",
        
            # Các trạng từ
        "rat": "rất",
        "hon": "hơn",
        "lam": "lắm",
        "qua": "quá",
        "nhieu": "nhiều",
        "it": "ít",

        # Địa điểm – thời gian
        "nay": "nay",
        "mai": "mai",
        "hom": "hôm",
        "truoc": "trước",
        "sau": "sau",
        "o": "ở",

        # Từ liên kết
        "va": "và",
        "voi": "với",
        "vi": "vì",
        "nen": "nên",
        "nhung": "nhưng",

        # Các từ cơ bản
        "co": "có",
        "duoc": "được",
        "du": "đủ",
        "thoi": "thôi",
        "roi": "rồi",
        "cung": "cũng",
        "luon": "luôn",
        "neu": "nếu",
        "dang": "đang",
        "se": "sẽ",
        "da": "đã",

        # Danh từ
        "con": "con",
        "nguoi": "người",
        "ban be": "bạn bè",
        "gia dinh": "gia đình",
        "cong viec": "công việc",
        "truong": "trường",
        "lop": "lớp",
        "mon": "món",
        "an": "ăn",
        "quan": "quán",
        "nha": "nhà",
        "cua": "của",

        # Cảm xúc – đánh giá
        "te": "tệ",
        "tot": "tốt",
        "hay": "hay",
        "do": "dở",
        "chap nhan": "chấp nhận",
        "tuyet": "tuyệt",
        "kha": "khá",
        "de": "dễ",
        "kho": "khó",
        "to": "to",
        "nho": "nhỏ",
        "lon": "lớn",
        "nhe": "nhẹ",
        "man": "mặn",
        "ngot": "ngọt",  
        "lam": "lắm" 
}

def is_valid_vietnamese(text: str) -> bool:
    """
    Heuristic đơn giản:
    - Có ít nhất 2 từ
    - Phần lớn kí tự là chữ cái, có đủ nguyên âm tiếng Việt
    - Có ít nhất 1 stopword Việt phổ biến
    - Không toàn là kí tự ngẫu nhiên / số / ký hiệu
    """
    if not isinstance(text, str):
        return False

    text = text.strip()
    # Kiểm tra độ dài tối thiểu, nếu dưới 5 kí tự thường không đủ nghĩa
    if len(text) <= 5:
        return False

    # Tách từ bằng Underthesea
    words = word_tokenize(text, format="list")  # Tách từ, trả về danh sách các từ
    print("Từ sau khi tách:", words)

    # Phải có ít nhất 2 từ
    if len(words) < 2:
        return False

    # Ghép tất cả chữ cái lại
    letters = "".join(words)

    # Phải có ít nhất 1 chữ cái
    if not letters:
        return False

    # Tỉ lệ nguyên âm
    vowel_count = sum(1 for ch in letters if ch in VIET_VOWELS)
    ratio_vowel = vowel_count / len(letters)

    # Nếu quá ít nguyên âm → thường là random / không phải tiếng Việt
    if ratio_vowel < 0.25:
        return False

    # Nếu có ít nhất 1 stopword Việt → ưu tiên coi là hợp lệ
    lower_words = [w.lower() for w in words]
    if any(w in VIET_STOPWORDS for w in lower_words):
        return True

    # Trung bình độ dài từ nếu quá dài → dễ là chuỗi vô nghĩa
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 10:
        return False

    # Mặc định: tạm coi là hợp lệ
    return True

def normalize_text(text: str) -> str:
    """
    Chuẩn hoá câu để HIỂN THỊ cho người dùng.
    Ví dụ: 'Ban khoe ko?' -> 'Bạn khỏe không?'
    (Chỉ làm 1 số luật đơn giản cho demo, không phải phục hồi dấu 100%)
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip().lower()

    # Tách token (giữ cả dấu ? ! , .)
    tokens = word_tokenize(text, format="list")  # Tách từ, trả về danh sách các từ

    new_tokens = []
    for tok in tokens:
        if tok.isalpha():  # chỉ map với chữ cái
            mapped = VIET_NORMALIZATION_DICT.get(tok, tok)
            new_tokens.append(mapped)
        else:
            new_tokens.append(tok)

    # Ghép lại, xử lý khoảng trắng trước dấu câu
    sentence = ""
    for tok in new_tokens:
        if tok in [".", ",", ":", ";", "?", "!", "…"]:
            sentence = sentence.rstrip() + tok + " "
        else:
            sentence += tok + " "
    sentence = sentence.strip()

    # Viết hoa chữ cái đầu
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    return sentence

def preprocess(text: str) -> str:
    """
    Chuẩn hoá câu để đưa vào mô hình Transformer:
    - đưa về chữ thường
    - thay viết tắt cơ bản
    - tách từ bằng underthesea
    """
    
    # CÉp kiểu chuỗi str
    if not isinstance(text, str):
        text = str(text)

    # Chữ thường và loại khoảng trắng thừa
    text = text.strip().lower()
    
    # Thay thế viết tắt / không dấu
    for k, v in VIET_NORMALIZATION_DICT.items():
        text = text.replace(f" {k} ", f" {v} ")

    # Tách từ bằng underthesea
    tokens = word_tokenize(text, format="text")
    return tokens

def classify(text: str) -> dict:
    """
    Phân loại cảm xúc cho 1 câu tiếng Việt.
    Trả về dict:
    {
        "original_text": ...,
        "normalized_text": ...,
        "sentiment": ...,
        "score": ...
    }
    """
    if not text or not text.strip():
        raise ValueError("Câu nhập vào rỗng.")
    if not is_valid_vietnamese(text):
        raise ValueError("Câu nhập vào không giống tiếng Việt hoặc không có nghĩa rõ ràng.")

    # Câu chuẩn hoá để hiển thị cho người dùng
    normalized = normalize_text(text)

    # Câu chuẩn hoá cho mô hình
    cleaned = preprocess(text)
    
    # Gọi mô hình phân loại cảm xúc
    sentiment_pipeline = get_sentiment_pipeline()
    
    all_scores = sentiment_pipeline(cleaned, truncation=True, max_length=256, top_k=None)
    print(all_scores)
    
    # Lấy nhãn có điểm số cao nhất
    result = max(all_scores, key=lambda x: x['score'])
    
    # Chuẩn hoá nhãn
    raw_label = result.get("label", "").upper()

    # Ánh xạ nhãn sang chuẩn
    sentiment = LABEL_MAPPING.get(raw_label, "NEUTRAL")
    
    # Độ tin cậy (làm tròn 2 chữ số thập phân)
    score = float(result.get("score", 0.0))
    
    if score < 0.5:
        sentiment = "NEUTRAL"

    return {
        "original_text": text,
        "normalized_text": normalized,
        "sentiment": sentiment,
        "score": score,
    }
