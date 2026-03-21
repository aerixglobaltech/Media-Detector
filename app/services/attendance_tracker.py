"""
app/services/attendance_tracker.py  –  Smart Attendance Tracker
──────────────────────────────────────────────────────────────────────────────
Tracks per-person entry/exit cycles using camera recognition heartbeats.

Design
──────
• Pipeline calls heartbeat(identity, camera_name) on every frame where a
  recognised staff member is visible.
• A background watcher thread runs every 30 s and declares a person as OUT
  if their last_seen timestamp exceeds the configured exit timeout.
• Each IN/OUT event is written to movement_log (raw event log).
• attendance table holds the daily summary row per person — first_in,
  last_out, movement_count, total_duration_minutes, day_status.
• EOD: after the configured hour, if a person has been OUT for ≥ exit_timeout
  they are considered done for the day (day_status = 'closed').
• Midnight rollover closes any still-open records from the previous day.

Configuration (read from system_settings on every watcher cycle so changes
take effect without restart):
  attendance_exit_timeout_mins  default 5
  attendance_eod_hour           default 19  (7 PM)
  attendance_notify_exit        default true
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, date, timedelta
from typing import Optional

log = logging.getLogger("attendance_tracker")

# How often the watcher thread checks for exits (seconds)
_WATCHER_INTERVAL_SEC = 30


class AttendanceTracker:
    """
    Singleton-safe stateful tracker.  One instance lives inside AIPipeline.
    """

    def __init__(self, notifier=None):
        self._lock = threading.Lock()
        self.notifier = notifier  # TelegramNotifier instance, optional

        # identity -> last seen monotonic timestamp
        self._last_seen: dict[str, float] = {}

        # identity -> current state  'IN' | 'OUT' | None (never seen today)
        self._state: dict[str, str] = {}

        # identity -> camera name where they were last seen
        self._camera: dict[str, str] = {}

        # identity -> monotonic time when they entered (for duration calc)
        self._entry_time: dict[str, float] = {}

        # Start background watcher
        self._stop_evt = threading.Event()
        self._watcher = threading.Thread(
            target=self._watcher_loop, daemon=True, name="AttendanceWatcher"
        )
        self._watcher.start()
        log.info("AttendanceTracker started.")

    # ── Public API ────────────────────────────────────────────────────────────

    def heartbeat(self, identity: str, camera_name: str, roles: list[str] = []) -> None:
        """
        Called by the pipeline on every frame where `identity` is visible.
        Records the sighting and triggers an IN event if they were OUT.
        """
        if not identity or identity == "Unknown":
            return

        now_mono = time.monotonic()
        now_wall = datetime.now()

        with self._lock:
            self._last_seen[identity] = now_mono
            self._camera[identity] = camera_name
            prev_state = self._state.get(identity)

        if prev_state != "IN":
            # ROLE CHECK: Only transition OUT→IN if camera is an Entry camera
            is_entry = (not roles) or ("entry" in [r.lower() for r in roles])
            if is_entry:
                self._record_event(identity, "IN", camera_name, now_wall)
                with self._lock:
                    self._state[identity] = "IN"
                    self._entry_time[identity] = now_mono
            else:
                log.debug(f"AT: Heartbeat for {identity} on non-entry camera {camera_name}. Staying in OUT/None state but updating last_seen.")

    def stop(self) -> None:
        self._stop_evt.set()

    # ── Settings helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_settings() -> dict:
        """Read attendance config from system_settings (live, not cached)."""
        defaults = {
            "attendance_exit_timeout_mins": 5,
            "attendance_eod_hour": 19,
            "attendance_notify_exit": True,
        }
        try:
            from app.db.session import get_db_connection
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT key, value FROM system_settings WHERE key LIKE 'attendance_%'"
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            for row in rows:
                k, v = row["key"], row["value"]
                if k == "attendance_exit_timeout_mins":
                    defaults[k] = int(v)
                elif k == "attendance_eod_hour":
                    defaults[k] = int(v)
                elif k == "attendance_notify_exit":
                    defaults[k] = v.lower() == "true"
        except Exception as exc:
            log.warning("Could not read attendance settings: %s", exc)
        return defaults

    # ── Background watcher ────────────────────────────────────────────────────

    def _watcher_loop(self) -> None:
        while not self._stop_evt.wait(timeout=_WATCHER_INTERVAL_SEC):
            try:
                self._check_exits()
                self._close_previous_day()
            except Exception as exc:
                log.error("AttendanceWatcher error: %s", exc)

    def _check_exits(self) -> None:
        cfg = self._get_settings()
        timeout_sec = cfg["attendance_exit_timeout_mins"] * 60
        eod_hour = cfg["attendance_eod_hour"]
        notify_exit = cfg["attendance_notify_exit"]
        now_mono = time.monotonic()
        now_wall = datetime.now()

        with self._lock:
            identities = list(self._state.keys())

        for identity in identities:
            with self._lock:
                state = self._state.get(identity)
                last = self._last_seen.get(identity, 0)
                camera = self._camera.get(identity, "")
                entry_t = self._entry_time.get(identity)

            if state != "IN":
                continue

            elapsed = now_mono - last
            if elapsed < timeout_sec:
                continue

            # Declare OUT
            log.info(
                "%s declared OUT after %.0fs absence (timeout=%ds)",
                identity, elapsed, timeout_sec,
            )
            
            # ROLE CHECK: Only record OUT event if the last camera seen on was an EXit camera
            # This is a bit tricky if they were last seen on a General camera...
            # But the requirement says "Exit detection should trigger only from cameras assigned as 'Exit'."
            # Let's check roles for the camera where they were last seen.
            # Wait, we don't have roles in self._camera. We should probably store them or fetch them.
            # For now, let's assume if it wasn't an Exit camera, we just mark them as OUT in memory but don't log a formal event?
            # Actually, standard behavior is usually best here unless specified otherwise.
            # I'll fetch roles for the camera.
            
            last_cam_roles = []
            try:
                from app.api.routes.camera import get_all_cameras
                all_cams = get_all_cameras()
                cam_info = next((c for c in all_cams if c["name"] == camera), None)
                if cam_info:
                    last_cam_roles = cam_info.get("roles", [])
            except: pass
            
            is_exit_cam = "exit" in [r.lower() for r in last_cam_roles]
            if is_exit_cam:
                self._record_event(identity, "OUT", camera, now_wall,
                                   entry_mono=entry_t, notify=notify_exit)
            else:
                log.info(f"AT: {identity} disappeared from non-exit camera {camera}. Marking OUT in memory only.")

            with self._lock:
                self._state[identity] = "OUT"
                self._entry_time.pop(identity, None)

            # EOD close — after EOD hour and still OUT
            if now_wall.hour >= eod_hour:
                self._close_day(identity, now_wall.date())

    def _close_previous_day(self) -> None:
        """At midnight, close any attendance records from yesterday."""
        today = date.today()
        try:
            from app.db.session import get_db_connection
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE attendance
                SET day_status = 'closed',
                    last_exit_time   = COALESCE(last_exit_time, timestamp)
                WHERE day_status = 'open'
                  AND attendance_date < %s
                """,
                (today,),
            )
            rows = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()
            if rows:
                log.info("Midnight rollover: closed %d open attendance record(s).", rows)
        except Exception as exc:
            log.error("Midnight rollover failed: %s", exc)

    # ── DB writes ─────────────────────────────────────────────────────────────

    def _record_event(
        self,
        identity: str,
        event_type: str,          # 'IN' | 'OUT'
        camera_name: str,
        wall_time: datetime,
        entry_mono: Optional[float] = None,
        notify: bool = True,
    ) -> None:
        """Write to movement_log and upsert today's attendance summary."""
        try:
            from app.db.session import get_db_connection
            conn = get_db_connection()
            cur = conn.cursor()

            # Resolve staff_id
            cur.execute(
                "SELECT id FROM staff_profiles WHERE name = %s LIMIT 1",
                (identity,),
            )
            row = cur.fetchone()
            if not row:
                log.warning("AttendanceTracker: staff '%s' not in staff_profiles", identity)
                cur.close()
                conn.close()
                return

            staff_id = row["id"]
            today = wall_time.date()

            # 1. Insert movement_log
            from app.services.attendance_service import log_movement
            log_movement(camera_name, "") # Legacy tracker events don't have images here

            # 2. Upsert attendance daily summary
            if event_type == "IN":
                cur.execute(
                    """
                    INSERT INTO attendance
                        (staff_id, status, in_time, timestamp, attendance_date, first_entry_time,
                         movement_count, day_status)
                    VALUES (%s, 'IN', %s, %s, %s, %s, 1, 'open')
                    ON CONFLICT (staff_id, attendance_date) DO UPDATE
                        SET status             = 'IN',
                            in_time            = EXCLUDED.in_time,
                            first_entry_time   = COALESCE(attendance.first_entry_time, EXCLUDED.first_entry_time),
                            movement_count     = attendance.movement_count + 1,
                            day_status         = 'open'
                    """,
                    (staff_id, wall_time, wall_time, today, wall_time),
                )
            else:  # OUT
                # Calculate duration of this visit
                duration_mins = 0
                if entry_mono is not None:
                    duration_mins = int((time.monotonic() - entry_mono) / 60)

                cur.execute(
                    """
                    INSERT INTO attendance
                        (staff_id, status, out_time, timestamp, attendance_date, last_exit_time,
                         total_duration_minutes, day_status)
                    VALUES (%s, 'OUT', %s, %s, %s, %s, %s, 'open')
                    ON CONFLICT (staff_id, attendance_date) DO UPDATE
                        SET status                 = 'OUT',
                            out_time               = EXCLUDED.out_time,
                            last_exit_time         = EXCLUDED.last_exit_time,
                            total_duration_minutes = attendance.total_duration_minutes + %s
                    """,
                    (staff_id, wall_time, wall_time, today,
                     wall_time, duration_mins, duration_mins),
                )

            conn.commit()
            cur.close()
            conn.close()
            log.info("Recorded %s event for %s (staff_id=%d)", event_type, identity, staff_id)

        except Exception as exc:
            log.error("AttendanceTracker._record_event failed: %s", exc)
            return

        # 3. Telegram notification
        if self.notifier and notify:
            try:
                icon = "🟢" if event_type == "IN" else "🔴"
                verb = "entered" if event_type == "IN" else "exited"
                msg = (
                    f"{icon} *{identity}* has {verb} the office.\n"
                    f"📍 Camera: {camera_name}\n"
                    f"🕐 Time: {wall_time.strftime('%H:%M:%S')}"
                )
                if event_type == "OUT" and entry_mono is not None:
                    duration_mins = int((time.monotonic() - entry_mono) / 60)
                    msg += f"\n⏱ Duration: {duration_mins} min"
                self.notifier.send_message(msg)
            except Exception as exc:
                log.warning("Notification failed for %s %s: %s", identity, event_type, exc)

    def _close_day(self, identity: str, today: date) -> None:
        """Mark attendance day_status = closed for this person."""
        try:
            from app.db.session import get_db_connection
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE attendance SET day_status = 'closed'
                WHERE staff_id = (
                    SELECT id FROM staff_profiles WHERE name = %s LIMIT 1
                ) AND attendance_date = %s
                """,
                (identity, today),
            )
            conn.commit()
            cur.close()
            conn.close()
            log.info("EOD: closed attendance for %s on %s", identity, today)
        except Exception as exc:
            log.error("EOD close failed for %s: %s", identity, exc)
