"""
Vendored, patched copy of streamlit-mic-recorder (0.0.8).

The upstream package calls getUserMedia({audio:{channelCount:1}}) with no
explicit audio-processing constraints, so the browser applies its default
echoCancellation/noiseSuppression/autoGainControl -- tuned for conversational
speech. A flat, unchanging sustained vowel doesn't look like speech to those
algorithms, so they can gate the signal down to silence partway through the
recording, even though the resulting file has the correct duration and byte
count. This copy explicitly disables all three so the raw signal is kept.

Everything else (the Streamlit component wire protocol, the returned dict
shape) is unchanged from upstream.
"""
import os
import base64
import streamlit as st
import streamlit.components.v1 as components

_component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "build")
_component_func = components.declare_component("mic_recorder_patched", path=_component_dir)


def mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording",
                  just_once=False, use_container_width=False, format="wav", key=None):
    if "_last_mic_recorder_patched_id" not in st.session_state:
        st.session_state._last_mic_recorder_patched_id = 0
    component_value = _component_func(
        start_prompt=start_prompt, stop_prompt=stop_prompt,
        use_container_width=use_container_width, format=format, key=key, default=None,
    )
    if component_value is None:
        return None
    rec_id = component_value["id"]
    is_new = rec_id > st.session_state._last_mic_recorder_patched_id
    st.session_state._last_mic_recorder_patched_id = max(rec_id, st.session_state._last_mic_recorder_patched_id)
    if not is_new:
        return None
    audio_bytes = base64.b64decode(component_value["audio_base64"])
    return {
        "bytes": audio_bytes,
        "sample_rate": component_value["sample_rate"],
        "sample_width": component_value["sample_width"],
        "format": component_value["format"],
        "id": rec_id,
    }
