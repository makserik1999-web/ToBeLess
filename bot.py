# bot.py — надежная обёртка для Telegram API
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "8495984884:AAFNGoLpWK0y0rq0Rtpj_8a4vo6EH4tGWCo")  # поставь через export TG_BOT_TOKEN=...
CHAT_ID = os.environ.get("TG_CHAT_ID", "-1003062260079")      # либо поставь сюда id (пример: 123456789 или -100123456789)

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

def _post(method, data=None, files=None, timeout=8):
    url = f"{BASE_URL}/{method}"
    try:
        resp = requests.post(url, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Telegram API error ({method}): {e}")
        return None

def send_alert(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        logging.error("BOT_TOKEN or CHAT_ID not set. Set environment variables TG_BOT_TOKEN and TG_CHAT_ID.")
        return False
    payload = {"chat_id": CHAT_ID, "text": message}
    res = _post("sendMessage", data=payload)
    return res

def send_photo(image_path_or_bytes, caption: str = ""):
    if not BOT_TOKEN or not CHAT_ID:
        logging.error("BOT_TOKEN or CHAT_ID not set.")
        return False

    files = None
    data = {"chat_id": CHAT_ID, "caption": caption}

    try:
        # If bytes passed
        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            files = {"photo": ("image.jpg", image_path_or_bytes)}
            res = _post("sendPhoto", data=data, files=files)
            return res

        # else assume path
        if not os.path.exists(image_path_or_bytes):
            logging.error(f"File not found: {image_path_or_bytes}")
            return None
        with open(image_path_or_bytes, "rb") as f:
            files = {"photo": f}
            res = _post("sendPhoto", data=data, files=files)
            return res
    except Exception as e:
        logging.error(f"send_photo error: {e}")
        return None

def test_bot():
    """Quick test: sends 'test' message, returns response dict or None"""
    return send_alert("✅ Test message from FightDetector (bot test)")
