import requests
import logging
import time
import os

log = logging.getLogger("notifier")

class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.last_notify_time = {} 
        self.cooldown = 60 

    def _get_active_bots(self):
        """Fetch all active bots from database."""
        from app.db.session import get_db_connection
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT bot_token, chat_ids FROM telegram_bots WHERE is_active = TRUE")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return rows
        except Exception as e:
            log.error(f"Error fetching active bots: {e}")
            return []

    def send_message(self, text):
        active_bots = self._get_active_bots()
        if not active_bots:
            return False
            
        any_success = False
        for bot in active_bots:
            token = bot['bot_token']
            raw_ids = bot['chat_ids']
            chat_ids = [i.strip() for i in raw_ids.split(",") if i.strip()]
            
            for cid in chat_ids:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {
                    "chat_id": cid,
                    "text": text,
                    "parse_mode": "Markdown"
                }
                try:
                    resp = requests.post(url, json=payload, timeout=5)
                    if resp.status_code == 200:
                        log.info(f"Telegram notification sent via bot to {cid}.")
                        any_success = True
                    else:
                        log.error(f"Telegram error for {cid}: {resp.text}")
                except Exception as e:
                    log.error(f"Failed to send Telegram to {cid}: {e}")
        
        return any_success

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
