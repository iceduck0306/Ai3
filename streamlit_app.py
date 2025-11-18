# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1o3zlwmIIlLyc8AJJWcacRpc_XThxL3CT")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {

     labels[0]: {
       "texts": ["네이마르는브라질 축구선수입니다", "여친 다수 보유", "에버랜드 방문"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://youtu.be/XXXXXXXXXXX"]
     },

     labels[1]: {
       "texts": ["네이마르는브라질 축구선수입니다", "여친 다수 보유", "에버랜드 방문"],
       "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
       "videos": ["https://youtu.be/XXXXXXXXXXX"]
     },

     labels[2]: {
       "texts": ["네이마르는브라질 축구선수입니다", "여친 다수 보유", "에버랜드 방문"],
       "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
       "videos": ["https://youtu.be/XXXXXXXXXXX"]
     },
    
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
