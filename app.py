import asyncio
import io
import math
import os
import shutil
import tempfile
import pandas as pd
from pydub import AudioSegment, effects
from shazamio import Shazam
import streamlit as st

# Set page layout
st.set_page_config(
    page_title="Disco2Disco - Tempo Shift Shazam", page_icon="🪩", layout="wide"
)

# Title with "TRACK IDENTIFIER" vertically baseline-aligned with Disco2Disco
st.markdown(
    """
    <h1 style='display: inline-flex; align-items: baseline; gap: 0.3em; margin-bottom: 0.2em;'>
        <span>🪩 Disco2Disco</span>
        <sub style='font-size: 0.45em; text-transform: uppercase; opacity: 0.8; vertical-align: baseline;'>TRACK IDENTIFIER</sub>
    </h1>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "Upload an audio file to apply tempo-shifted variations and search Shazam"
    " for matches."
)

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
enable_eq = False  # Hardcoded toggle while UI controls are hidden

st.sidebar.header("⚙️ Search Configuration")

step_percent = st.sidebar.number_input(
    "Step Increment (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.5
)
range_percent = st.sidebar.number_input(
    "Total Range (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0
)
direction = st.sidebar.selectbox(
    "Direction",
    options=["symmetrical", "up", "down"],
    format_func=lambda x: x.capitalize(),
)

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Network & Retry Settings")
rate_limit_delay = st.sidebar.slider(
    "Delay Between Requests (s)", 0.0, 5.0, 1.0, 0.5
)
request_timeout = st.sidebar.slider(
    "Request Timeout (s)", 3.0, 30.0, 10.0, 1.0
)
max_retries = st.sidebar.number_input(
    "Max Retries", min_value=1, max_value=5, value=3
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def apply_basic_eq(
    audio: AudioSegment, low_pass_hz: int, high_pass_hz: int, do_normalize: bool
) -> AudioSegment:
    """Applies high-pass, low-pass, and optional volume normalization via Pydub."""
    processed = audio

    if high_pass_hz > 0:
        processed = processed.high_pass_filter(high_pass_hz)

    if low_pass_hz < 20000:
        processed = processed.low_pass_filter(low_pass_hz)

    if do_normalize:
        processed = effects.normalize(processed)

    return processed


def calculate_percentages(
    step: float, max_range: float, direction_str: str
) -> list[float]:
    num_steps = math.floor(max_range / step)
    percents = set()

    if direction_str in ("symmetrical", "up"):
        for i in range(1, num_steps + 1):
            percents.add(round(i * step, 2))

    if direction_str in ("symmetrical", "down"):
        for i in range(1, num_steps + 1):
            percents.add(round(-i * step, 2))

    percents.add(0.0)
    return sorted(list(percents))


def create_tempo_shifted_copy(
    audio: AudioSegment, percent_shift: float, output_path: str
) -> str:
    """Exports shifted audio as WAV for lossless processing and faster writes."""
    if percent_shift == 0.0:
        audio.export(output_path, format="wav")
        return output_path

    multiplier = 1.0 + (percent_shift / 100.0)
    new_frame_rate = int(audio.frame_rate * multiplier)

    shifted_audio = audio._spawn(
        audio.raw_data, overrides={"frame_rate": new_frame_rate}
    )
    shifted_audio = shifted_audio.set_frame_rate(audio.frame_rate)
    shifted_audio.export(output_path, format="wav")
    return output_path


def render_fixed_status_card(step_text: str, match_text: str, sub_msg: str) -> str:
    """Renders a single HTML string on a single line to prevent Streamlit code-block escaping."""
    if match_text:
        match_html = f"<div style='background-color: rgba(46, 125, 50, 0.15); border: 1px solid #2e7d32; padding: 6px 12px; border-radius: 6px; color: #81c784; font-weight: 500; height: 36px; line-height: 24px; box-sizing: border-box; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%;'>{match_text}</div>"
    else:
        match_html = "<div style='background-color: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.1); padding: 6px 12px; border-radius: 6px; color: rgba(255, 255, 255, 0.5); font-weight: 400; height: 36px; line-height: 24px; box-sizing: border-box; width: 100%;'>🔍 No matches yet...</div>"

    return f"<div style='background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; height: 155px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;'><div style='font-size: 1.05rem; font-weight: 600; height: 28px; line-height: 28px; width: 100%;'>{step_text}</div><div style='width: 100%; height: 36px;'>{match_html}</div><div style='background: rgba(33, 150, 243, 0.12); border-left: 4px solid #2196f3; padding: 6px 12px; border-radius: 4px; font-size: 0.9rem; height: 36px; line-height: 24px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; box-sizing: border-box; width: 100%;'>{sub_msg}</div></div>"


async def recognize_with_retry(
    shazam: Shazam,
    audio_bytes: bytes,
    timeout: float,
    retries: int,
    status_slot,
    step_str: str,
    hit_str: str,
):
    for attempt in range(1, retries + 1):
        try:
            if attempt > 1:
                status_slot.html(
                    render_fixed_status_card(
                        step_str,
                        hit_str,
                        f"⚠️ Timeout on attempt {attempt - 1}. Retrying attempt {attempt}/{retries} in 10s...",
                    )
                )
            out = await asyncio.wait_for(
                shazam.recognize(audio_bytes), timeout=timeout
            )
            return "SUCCESS", out
        except asyncio.TimeoutError:
            if attempt < retries:
                status_slot.html(
                    render_fixed_status_card(
                        step_str,
                        hit_str,
                        f"⏱️ Attempt {attempt} timed out after {timeout}s. Retrying in 10s...",
                    )
                )
                await asyncio.sleep(10.0)
            else:
                return "TIMEOUT", None
        except Exception:
            if attempt < retries:
                status_slot.html(
                    render_fixed_status_card(
                        step_str,
                        hit_str,
                        f"⚠️ Network error on attempt {attempt}. Retrying in 10s...",
                    )
                )
                await asyncio.sleep(10.0)
            else:
                return "ERROR", None


async def process_all_shifts(
    shazam, percents, base_audio, temp_dir, request_timeout, max_retries, rate_limit_delay,
    status_card_slot, progress_bar, table_placeholder, audio_container
):
    """Wraps the entire processing loop in a single async task to prevent event loop crashes."""
    results = []
    latest_hit_html = ""

    for idx, percent in enumerate(percents, start=1):
        label = f"{percent:+}%" if percent != 0.0 else "Original (0%)"
        filename = f"track_{idx}.wav"
        temp_filepath = os.path.join(temp_dir, filename)

        step_display = (
            f"Current Step: <b>{idx}/{len(percents)}</b>"
            f" (<b>{label}</b>)"
        )
        sub_msg = f"⏳ Generating tempo-shifted audio slice ({label})..."

        status_card_slot.html(
            render_fixed_status_card(step_display, latest_hit_html, sub_msg)
        )

        create_tempo_shifted_copy(base_audio, percent, temp_filepath)

        with open(temp_filepath, "rb") as f:
            audio_bytes = f.read()

        # Mount Audio Player into Collapsible Expander
        with audio_container:
            st.caption(f"**Slice {idx} ({label})**")
            st.audio(audio_bytes, format="audio/wav")

        sub_msg = f"📡 Querying Shazam API ({label})... (Timeout: {request_timeout}s)"
        status_card_slot.html(
            render_fixed_status_card(step_display, latest_hit_html, sub_msg)
        )

        status_code, out = await recognize_with_retry(
            shazam,
            audio_bytes,
            request_timeout,
            max_retries,
            status_card_slot,
            step_display,
            latest_hit_html,
        )

        if status_code == "SUCCESS" and "track" in out:
            track_info = out["track"]
            title = track_info.get("title", "Unknown Title")
            artist = track_info.get("subtitle", "Unknown Artist")
            results.append(
                {
                    "Shift": label,
                    "Status": "🪩 HIT",
                    "Track Title": title,
                    "Artist": artist,
                }
            )
            latest_hit_html = f"🎉 <b>Latest Match ({label}):</b> {title} — <i>{artist}</i>"
        elif status_code == "SUCCESS":
            results.append(
                {
                    "Shift": label,
                    "Status": "❌ NO MATCH",
                    "Track Title": "-",
                    "Artist": "-",
                }
            )
        elif status_code == "TIMEOUT":
            results.append(
                {
                    "Shift": label,
                    "Status": "⏱️ TIMEOUT",
                    "Track Title": "(Request timed out)",
                    "Artist": "-",
                }
            )
        else:
            results.append(
                {
                    "Shift": label,
                    "Status": f"⚠️ {status_code}",
                    "Track Title": "(API Error / Rate Limit)",
                    "Artist": "-",
                }
            )

        progress_bar.progress(idx / len(percents))
        df = pd.DataFrame(results)
        table_placeholder.dataframe(df, width="stretch")

        if idx < len(percents) and rate_limit_delay > 0:
            sub_msg = f"😴 Rate-limit delay: Pausing {rate_limit_delay}s before next request..."
            status_card_slot.html(
                render_fixed_status_card(step_display, latest_hit_html, sub_msg)
            )
            await asyncio.sleep(rate_limit_delay)
            
    return results, latest_hit_html


# ==============================================================================
# MAIN UI & PROCESSING
# ==============================================================================
uploaded_file = st.file_uploader(
    "Choose an audio file", type=["mp3", "m4a", "wav", "flac", "ogg"]
)

if uploaded_file is not None:
    st.caption("Original Audio Preview:")
    st.audio(uploaded_file, format=uploaded_file.type)

    percents = calculate_percentages(step_percent, range_percent, direction)
    st.info(
        f"📋 Calculated **{len(percents)}** variation(s) to test across the range."
    )

    if st.button("🚀 Start Shazam Search", type="primary"):
        temp_dir = tempfile.mkdtemp()

        try:
            temp_input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("🎧 Loading source audio file...")
            base_audio = AudioSegment.from_file(temp_input_path)

            if enable_eq:
                st.write("🎛️ Applying Basic EQ & Normalization...")
                base_audio = apply_basic_eq(base_audio, 15000, 80, True)

                preview_buffer = io.BytesIO()
                base_audio.export(preview_buffer, format="mp3")
                preview_buffer.seek(0)

                st.subheader("🔊 Equalized Audio Preview")
                st.audio(preview_buffer, format="audio/mp3")

            shazam = Shazam()

            st.markdown("### 🕺 Search In Progress")

            # Dedicated single-slot UI elements
            status_card_slot = st.empty()
            progress_bar = st.progress(0.0)
            table_placeholder = st.empty()

            # Collapsible Audio Preview Container (Default Collapsed)
            audio_expander = st.expander("🎧 Audio Slices (Tempo Shift Previews)", expanded=False)
            with audio_expander:
                audio_container = st.container()

            # Run the entire processing loop within a single event loop execution
            results, latest_hit_html = asyncio.run(
                process_all_shifts(
                    shazam, percents, base_audio, temp_dir, request_timeout,
                    max_retries, rate_limit_delay, status_card_slot, progress_bar,
                    table_placeholder, audio_container
                )
            )

            # Final complete status display
            status_card_slot.html(
                render_fixed_status_card(
                    "Search Complete!",
                    latest_hit_html,
                    "✅ All tempo variations analyzed successfully.",
                )
            )

            hits = [r for r in results if r["Status"] == "🪩 HIT"]
            timeouts = [
                r for r in results if r["Status"] in ["⏱️️ TIMEOUT", "⚠️ ERROR"]
            ]
            no_matches = len(percents) - len(hits) - len(timeouts)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tested", len(percents))
            col2.metric("Hits Found", len(hits))
            col3.metric("No Matches", no_matches)
            col4.metric("Timeouts / Errors", len(timeouts))

            df_final = pd.DataFrame(results)
            csv_data = df_final.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="shazam_search_results.csv",
                mime="text/csv",
            )

        finally:
            shutil.rmtree(temp_dir)