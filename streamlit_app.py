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
       "texts": ["현존 최고의 축구선수", "최고의 드리블러", "fc온라인 최고 인기매물"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://youtu.be/XXXXXXXXXXX"]
     },

     labels[2]: {
       "texts": ["미친 골결정력", "강한 피지컬과 몸싸움", "상대 수비를 흔드는 지능적인 움직임"],
       "images": ["data:image/jpeg;base64,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"],
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
