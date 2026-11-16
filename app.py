
import os
import time

import streamlit as st
from dotenv import load_dotenv

# Page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Fieldnote · Research Pipeline",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Support both local .env and Streamlit Cloud secrets.
load_dotenv()
for secret_name in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
    try:
        secret_value = st.secrets.get(secret_name)
        if secret_value:
            os.environ[secret_name] = str(secret_value)
    except Exception:
        pass

# Import after loading credentials into the environment.
from pipeline import run_research_pipeline


st.markdown(
    """
    
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #252525 !important;
}

.stApp {
    background: #ffffff !important;
}

.block-container {
    max-width: 1180px;
    padding: 2rem 2rem 4rem;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #202020 !important;
}

.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}

.eyebrow {
    font-family: 'DM Mono', monospace;
    color: #d18b00;
    letter-spacing: .25em;
    font-size: .72rem;
}

.hero h1 {
    font-size: clamp(3rem, 7vw, 5rem);
    font-weight: 800;
    margin: .5rem 0;
    color: #202020 !important;
}

.hero h1 span {
    color: #e5a000;
}

.sub {
    color: #666666;
    font-size: 1.05rem;
    max-width: 650px;
    margin: auto;
    line-height: 1.7;
}

hr {
    border-color: #e5c77d;
}

/* Research topic input */
.stTextInput input,
.stTextInput input:focus {
    background-color: #ffffff !important;
    color: #202020 !important;
    -webkit-text-fill-color: #202020 !important;
    border: 1px solid #d6a63b !important;
    caret-color: #d18b00 !important;
    border-radius: 6px !important;
}

.stTextInput input::placeholder {
    color: #888888 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #888888 !important;
}

.stTextInput label {
    color: #252525 !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background: #e5a000 !important;
    color: #ffffff !important;
    border: 0 !important;
    font-weight: 700;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #c98b00 !important;
    color: #ffffff !important;
}

/* Pipeline panel */
.panel {
    border: 1px solid #e5c77d;
    border-radius: 6px;
    padding: 1rem 1.3rem;
    background: #fffaf0;
    color: #252525;
}

.panel b {
    color: #222222;
}

.muted {
    color: #666666;
    font-size: .9rem;
}

.notice {
    color: #777777;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    text-align: center;
    margin-top: 3rem;
}

/* General text visibility */
.stMarkdown, .stCaption, .stText, p, label {
    color: #252525;
}

/* Status and progress */
[data-testid="stStatusWidget"] {
    background: #fffaf0;
    color: #252525;
    border: 1px solid #e5c77d;
}

</style>

    """,
    unsafe_allow_html=True,
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Start your research")

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Future of data science in India",
        key="topic_input",
    )

    run_clicked = st.button(
        "Run research pipeline",
        use_container_width=True
    )

    st.caption(
        "Examples: AI in healthcare · India's semiconductor ecosystem · Green hydrogen"
    )

with right:
    st.subheader("Pipeline stages")

    st.markdown(
        """
        <div class="panel">
          <p><b>01 · Search</b><br>
          <span class="muted">Tavily retrieves relevant web results.</span></p>

          <p><b>02 · Reader</b><br>
          <span class="muted">Python extracts content from selected webpages.</span></p>

          <p><b>03 · Writer</b><br>
          <span class="muted">Gemini generates a structured research report.</span></p>

          <p><b>04 · Critic</b><br>
          <span class="muted">Gemini reviews report quality and evidence.</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        progress = st.progress(0)
        status = st.status("Starting research pipeline...", expanded=True)

        def update_progress(step, message):
            status.write(f"{step}: {message}")

            try:
                progress.progress(min(int(step) / 4, 1.0))
            except (ValueError, TypeError):
                pass

        try:
            result = run_research_pipeline(
                topic,
                progress_callback=update_progress
            )

            st.session_state.last_result = result

            progress.progress(1.0)
            status.update(
                label="Research pipeline completed!",
                state="complete",
                expanded=False
            )

        except Exception as e:
            status.update(
                label="Research pipeline failed",
                state="error",
                expanded=True
            )
            st.error(f"Error: {e}")

# Display research results
if st.session_state.last_result:
    result = st.session_state.last_result

    st.divider()
    st.subheader("Research Results")

    mode = result.get("execution_mode", "UNKNOWN")

    if mode == "LIVE_AI":
        st.success("Report generated using Gemini AI.")
    elif mode == "FALLBACK":
        st.warning("Fallback report generated from collected sources.")
    elif mode == "PARTIAL_FALLBACK":
        st.warning("Partial fallback report generated.")
    
    st.markdown("### Research Report")
    st.markdown(result.get("report", "No report available."))

    st.markdown("### Critic Feedback")
    st.markdown(result.get("feedback", "No critic feedback available."))

    st.markdown("### Sources")
    for source in result.get("sources", []):
        st.write(source)
