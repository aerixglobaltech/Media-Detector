import requests
import logging
import time
import os

log = logging.getLogger("notifier")

class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        raw_ids = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        # Support multiple IDs separated by commas
        self.chat_ids = [i.strip() for i in raw_ids.split(",") if i.strip()]
        self.enabled = os.environ.get("ENABLE_TELEGRAM", "False").lower() == "true"
        
        # Cooldown per track ID to prevent spamming for the same person (60 seconds)
        self.last_notify_time = {} 
        self.cooldown = 60 

    def send_message(self, text):
        if not self.enabled or not self.token or not self.chat_ids:
            return False
            
        success = False
        for cid in self.chat_ids:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": cid,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            try:
                resp = requests.post(url, json=payload, timeout=5)
                if resp.status_code == 200:
                    log.info(f"Telegram notification sent to {cid}.")
                    success = True
                else:
                    log.error(f"Telegram error for {cid}: {resp.text}")
            except Exception as e:
                log.error(f"Failed to send Telegram to {cid}: {e}")
        
        return success

    def notify_person(self, track_id, cam_name, action=""):
        now = time.monotonic()
        last_time = self.last_notify_time.get(track_id, 0)
        
        if (now - last_time) < self.cooldown:
            return # Still in cooldown for this specific person
            
        msg = f"🚨 *MISSION CONTROL ALERT*\n\n"
        msg += f"👤 *New Person Tracked*\n"
        msg += f"📍 *Camera:* {cam_name}\n"
        msg += f"🆔 *Track ID:* {track_id}\n"
        if action:
            msg += f"🏃 *Activity:* {action}\n"
        
        if self.send_message(msg):
            self.last_notify_time[track_id] = now
