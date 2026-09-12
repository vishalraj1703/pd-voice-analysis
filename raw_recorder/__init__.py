"""
Hand-written minimal audio recorder component.

Every prior attempt (Streamlit's native st.audio_input, streamlit-mic-recorder,
and by inspection audio-recorder-streamlit) captures audio sample-by-sample on
the page's own JavaScript main thread via the legacy ScriptProcessorNode API,
and/or leaves the browser's default echoCancellation/noiseSuppression/
autoGainControl enabled. Both are known sources of a recording that reports
the correct duration and byte count but contains partial or total silence --
exactly what was reported on both desktop and mobile.

This component instead uses the browser's native MediaRecorder API (encoding
happens outside page JavaScript entirely, immune to main-thread contention)
with audio processing explicitly disabled. No build step, no framework --
just the verified Streamlit component wire protocol handled directly in
frontend/index.html.
"""
import os
import base64
import streamlit as st
import streamlit.components.v1 as components

_component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component_func = components.declare_component("raw_recorder", path=_component_dir)


def raw_recorder(key=None):
    if "_last_raw_recorder_id" not in st.session_state:
        st.session_state._last_raw_recorder_id = 0
    component_value = _component_func(key=key, default=None)
    if component_value is None:
        return None
    rec_id = component_value.get("id", 0)
    is_new = rec_id > st.session_state._last_raw_recorder_id
    st.session_state._last_raw_recorder_id = max(rec_id, st.session_state._last_raw_recorder_id)
    if not is_new:
        return None
    audio_bytes = base64.b64decode(component_value["audio_base64"])
    return {
        "bytes": audio_bytes,
        "mime_type": component_value.get("mime_type", "audio/webm"),
        "size": component_value.get("size", len(audio_bytes)),
    }
