import os
import uuid
import json
import gradio as gr
from datetime import datetime
from llama_cpp import Llama
from typing import Dict
import requests
from bs4 import BeautifulSoup
from googlesearch import search

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

stop_flag = {"stop": False}
llm = None

def search_google(query: str, max_results=3):
    results = []
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        for url in search(query, num_results=max_results):
            try:
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    
                    page_text = soup.get_text(separator=' ', strip=True)
                    page_text_lower = page_text.lower()
                    query_lower = query.lower()

                    pos = page_text_lower.find(query_lower)
                    if pos != -1:
                        start = max(0, pos - 200)
                        end = min(len(page_text), pos + len(query) + 200)
                        snippet = page_text[start:end].strip()

                        results.append(f"🔎 {snippet}\n")
                    else:
                        snippet = page_text[:300].strip()
                        results.append(f"🔎 {snippet}\n")
                else:
                    results.append(f"⚠️ Gagal mengambil halaman dari {url}\n")
            except Exception:
                results.append(f"⚠️ Gagal mengambil halaman dari {url}\n")

    except Exception as e:
        print(f"[ERROR] Google search gagal: {e}")
        results.append("⚠️ Gagal mencari informasi dari Google. Periksa koneksi atau coba lagi nanti.")

    return results

def get_model_list():
    return [f for f in os.listdir("models") if f.endswith(".gguf")]

def list_sessions():
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sid = data.get("session_id", fname.replace(".json", ""))
                timestamp = data.get("timestamp", "")
                name = data.get("name", sid)
                sessions.append({
                    "session_id": sid,
                    "filename": fname,
                    "timestamp": timestamp,
                    "name": name,
                })
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def format_session_label(s):
    ts_str = s.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts_str)
        ts_str = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts_str = ""
    return f"{s.get('name', s['session_id'])} ({ts_str})"

def save_session(state: Dict):
    sid = state["session_id"]
    filename = os.path.join(SESSIONS_DIR, f"{sid}.json")
    data = {
        "session_id": sid,
        "history": state["history"],
        "params": state.get("params", {}),
        "timestamp": datetime.now().isoformat(),
        "name": state.get("name", sid)
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_session(session_id):
    filename = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "history": data.get("history", []),
                "session_id": data.get("session_id", session_id),
                "params": data.get("params", {}),
                "name": data.get("name", session_id)
            }
    return None

def load_model(model_name, ctx_size, n_threads, n_gpu_layers, chat_format):
    global llm
    try:
        model_path = os.path.join("models", model_name)
        llm = Llama(
            model_path=model_path,
            n_ctx=int(ctx_size),
            n_threads=int(n_threads),
            n_gpu_layers=int(n_gpu_layers),
            chat_format=chat_format
        )
        session_id = str(uuid.uuid4())
        state = {
            "history": [],
            "session_id": session_id,
            "params": {
                "model": model_name,
                "ctx": ctx_size,
                "threads": n_threads,
                "gpu_layers": n_gpu_layers,
                "format": chat_format
            },
            "name": f"Session {session_id[:8]}"
        }
        save_session(state)
        return state, gr.update(value=[{"role": "system", "content": f"✅ Model '{model_name}' berhasil dimuat!"}]), gr.update(value="✅ Model loaded")
    except Exception as e:
        return None, gr.update(value=[{"role": "system", "content": f"❌ Gagal memuat model: {str(e)}"}]), gr.update(value="❌ Failed to load model")

def format_chat(history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    return messages

def send_message(user_input, state, allow_search=True):
    global llm

    if llm is None:
        yield [{"role": "system", "content": "❌ Model belum dimuat!"}], state, user_input
        return

    if not state:
        session_id = str(uuid.uuid4())
        history = []
        state = {"session_id": session_id, "history": history, "name": f"Session {session_id[:8]}"}
    else:
        session_id = state.get("session_id", str(uuid.uuid4()))
        history = state.get("history", [])

    messages = format_chat(history)
    messages.append({"role": "user", "content": user_input})

    stop_flag["stop"] = False
    partial = ""

    trigger_words = [
        "tidak tahu",
        "tidak yakin",
        "maaf",
        "informasi tidak tersedia",
        "saya belum tahu",
        "belum ada informasi"
    ]

    for chunk in llm.create_chat_completion(messages, stream=True):
        if stop_flag["stop"]:
            break
        token = chunk["choices"][0]["delta"].get("content", "")
        partial += token

        partial_lower = partial.lower()
        if allow_search and any(trigger in partial_lower for trigger in trigger_words):
            stop_flag["stop"] = True
            break

        yield format_chat(history + [[user_input, partial]]), state, ""

    if allow_search and any(trigger in partial.lower() for trigger in trigger_words):
        partial_clean = partial
        for trigger in trigger_words:
            import re
            partial_clean = re.sub(re.escape(trigger), "", partial_clean, flags=re.IGNORECASE)

        partial_clean = " ".join(partial_clean.split())

        results = search_google(user_input)
        if results:
            extra_info = "\n\n📡 Saya mencari di internet dan menemukan:\n" + "\n".join(results)
            partial_clean += extra_info

        full_response = partial_clean
    else:
        full_response = partial

    history.append([user_input, full_response])
    state["history"] = history

    save_session(state)
    yield format_chat(history), state, ""

def stop_generation():
    stop_flag["stop"] = True

def clear_history(state):
    if state:
        state["history"] = []
        save_session(state)
    return [], state

def create_new_chat():
    new_sid = str(uuid.uuid4())
    new_state = {
        "history": [],
        "session_id": new_sid,
        "params": {},
        "name": f"Session {new_sid[:8]}"
    }
    save_session(new_state)
    return [], new_state, update_sessions_radio()

def delete_session(session_id):
    filename = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filename):
        os.remove(filename)
    return update_sessions_radio()

def update_sessions_radio():
    sessions = list_sessions()
    choices = [(format_session_label(s), s["session_id"]) for s in sessions]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)

def load_chat_from_session(session_id):
    if not session_id:
        return [], None, ""
    state = load_session(session_id)
    if state:
        return format_chat(state["history"]), state, state.get("name", "")
    return [], None, ""

def rename_session(session_id, new_name, state):
    if not session_id or not new_name or not state:
        return gr.update(value=state.get("name", "") if state else ""), None
    state["name"] = new_name
    save_session(state)
    updated_radio = update_sessions_radio()
    return gr.update(value=new_name), updated_radio

custom_css = """
.message.user {
    background-color: #ADD8E6;
    text-align: right;
    align-self: flex-end;
    color: black;
    border-radius: 15px 15px 0 15px;
    padding: 8px 12px;
}
.message.assistant {
    background-color: #E0E0E0;
    text-align: left;
    align-self: flex-start;
    color: black;
    border-radius: 15px 15px 15px 0;
    padding: 8px 12px;
}
.sidebar {
    border-right: 1px solid #ccc;
    padding-right: 8px;
    height: 600px;
    overflow-y: auto;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## 🦙 Chat GGUF ala WhatsApp + Sidebar Sesi dengan Rename")

    state = gr.State()
    session_id_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### Daftar Sesi")
            session_radio = gr.Radio(choices=[], label="Sesi", interactive=True)
            new_chat_btn = gr.Button("➕ New Chat")
            delete_chat_btn = gr.Button("🗑️ Delete")
            gr.Markdown("### Rename")
            rename_input = gr.Textbox(label="Nama Baru")
            rename_btn = gr.Button("Rename")

        with gr.Column(scale=3):
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_model_list(), label="Model GGUF", interactive=True)
                ctx_input = gr.Number(value=2048, label="Context Size")
                thread_input = gr.Number(value=6, label="Threads")
                gpu_layer_input = gr.Number(value=0, label="GPU Layers")
                chat_format = gr.Dropdown(choices=["llama-2", "llama-3", "chatml"], value="llama-3", label="Chat Format")
                load_btn = gr.Button("🚀 Load Model")
            model_status = gr.Textbox(label="Model Status", value="❌ Belum dimuat", interactive=False)

            chatbot = gr.Chatbot(label="Chat", type="messages", height=500)

            with gr.Row():
                user_input = gr.Textbox(placeholder="Tulis pesan...", label="", scale=5)
                send_btn = gr.Button("Kirim", scale=1)

            with gr.Row():
                stop_btn = gr.Button("⛔ Stop")
                clear_btn = gr.Button("🧹 Clear")

    demo.load(fn=update_sessions_radio, outputs=session_radio)
    session_radio.change(fn=load_chat_from_session, inputs=session_radio, outputs=[chatbot, state, rename_input])
    load_btn.click(fn=load_model, inputs=[model_dropdown, ctx_input, thread_input, gpu_layer_input, chat_format], outputs=[state, chatbot, model_status])
    send_btn.click(fn=send_message, inputs=[user_input, state], outputs=[chatbot, state, user_input])
    user_input.submit(fn=send_message, inputs=[user_input, state], outputs=[chatbot, state, user_input])
    send_btn.click(lambda: "", None, user_input)
    user_input.submit(lambda: "", None, user_input)
    stop_btn.click(fn=stop_generation)
    clear_btn.click(fn=clear_history, inputs=[state], outputs=[chatbot, state])
    new_chat_btn.click(fn=create_new_chat, outputs=[chatbot, state, session_radio])
    delete_chat_btn.click(fn=delete_session, inputs=session_radio, outputs=session_radio)
    rename_btn.click(fn=rename_session, inputs=[session_radio, rename_input, state], outputs=[rename_input, session_radio])

demo.launch()
