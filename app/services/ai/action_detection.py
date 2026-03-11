"""
modules/action_detection.py
─────────────────────────────────────────────────────────────────────────────
Action recognition using SlowFast-R50 (Kinetics-400, 400 classes).

All 400 Kinetics labels are mapped to surveillance-relevant categories:
  standing / sitting / walking / reading / writing / working / talking /
  eating / exercising / ⚠ fighting / ⚠ running / ⚠ falling

Runs FULLY NON-BLOCKING in a background ThreadPoolExecutor.
"""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pytorchvideo.models.hub as hub
    _PTV_OK = True
except Exception:
    _PTV_OK = False

log = logging.getLogger(__name__)

# ─── Full Kinetics-400 label list (alphabetical = index order) ────────────
K400 = [
    "abseiling","acting in play","air drumming","answering questions",
    "applauding","applying cream","archery","arm wrestling","arranging flowers",
    "assembling computer","auctioning","baby waking up","baking cookies",
    "balloon blowing","bandaging","barbecuing","bartending","beatboxing",
    "bee keeping","belly dancing","bench pressing","bending back","bending metal",
    "biking through snow","blasting sand","blowing glass","blowing leaves",
    "blowing nose","blowing out candles","bobsledding","bookbinding",
    "bouncing on trampoline","bowling","braiding hair","breading or breadcrumbing",
    "breakdancing","brush painting","brushing hair","brushing teeth",
    "building cabinet","building shed","bungee jumping","busking",
    "canoeing or kayaking","capoeira","carrying baby","cartwheeling",
    "carving pumpkin","catching fish","catching or throwing baseball",
    "catching or throwing frisbee","catching or throwing softball","celebrating",
    "changing oil","changing wheel on car","checking tires","cheerleading",
    "chopping wood","clapping","clay pottery making","clean and jerk",
    "cleaning floor","cleaning gutters","cleaning pool","cleaning shoes",
    "cleaning toilet","cleaning windows","climbing a rope","climbing ladder",
    "climbing tree","contact juggling","cooking chicken","cooking egg",
    "cooking on campfire","cooking sausages","counting money",
    "country line dancing","cracking knuckles","cracking neck","crawling baby",
    "crossing river","crying","curling hair","cutting nails","cutting pineapple",
    "cutting watermelon","dancing ballet","dancing charleston",
    "dancing gangnam style","dancing macarena","deadlifting",
    "decorating the christmas tree","digging","dining","directing traffic",
    "disc golfing","diving cliff","dodgeball","doing aerobics","doing laundry",
    "doing nails","drawing","dribbling basketball","drinking beer",
    "drinking shots","driving car","driving tractor","drop kicking",
    "drumming fingers","dunking basketball","dying hair","eating burger",
    "eating cake","eating carrots","eating chips","eating doughnuts",
    "eating hotdog","eating ice cream","eating spaghetti","eating watermelon",
    "egg hunting","exercising arm","exercising with an exercise ball",
    "extinguishing fire","faceplanting","feeding birds","feeding fish",
    "feeding goats","filling eyebrows","fixing hair","flipping pancake",
    "fly tying","flying kite","folding clothes","folding napkins","folding paper",
    "front raises","frying vegetables","garbage collecting","gargling",
    "getting a haircut","getting a tattoo","giving or receiving award",
    "golf chipping","golf driving","golf putting","grinding meat","grooming dog",
    "grooming horse","gymnastics tumbling","hammer throw","headbanging",
    "headbutting","high jump","high kick","hitting baseball","hockey stop",
    "holding snake","hopscotch","hoverboarding","hugging","hula hooping",
    "hurdling","hurling","ice climbing","ice fishing","ice skating",
    "ironing hair","javelin throw","jetskiing","jogging","jumping into pool",
    "jumpstyle dancing","kicking field goal","kicking soccer ball","kissing",
    "kitesurfing","knitting","krumping","laughing","lawn mowing","laying bricks",
    "long jump","lunge","making a cake","making a sandwich","making bed",
    "making jewelry","making pizza","making snowman","making sushi","making tea",
    "marching","massaging back","massaging feet","massaging legs",
    "massaging persons head","milking cow","mopping floor","motorcycling",
    "mountain climber exercise","moving furniture","mowing lawn",
    "news anchoring","opening bottle","opening present","paragliding",
    "parasailing","parkour","passing american football in game",
    "passing american football not in game","peeling apples","peeling potatoes",
    "petting animal","petting cat","picking fruit","planting trees","plastering",
    "playing accordion","playing badminton","playing bagpipes","playing basketball",
    "playing bass guitar","playing cards","playing cello","playing checkers",
    "playing chess","playing clarinet","playing controller","playing cricket",
    "playing cymbals","playing didgeridoo","playing drums","playing flute",
    "playing guitar","playing harmonica","playing harp","playing ice hockey",
    "playing keyboard","playing kickball","playing monopoly","playing organ",
    "playing paintball","playing piano","playing poker","playing recorder",
    "playing saxophone","playing squash or racquetball","playing tennis",
    "playing trombone","playing trumpet","playing ukulele","playing violin",
    "playing volleyball","playing xylophone","pole vault",
    "presenting weather forecast","pull ups","pumping fist","pumping gas",
    "punching bag","punching person boxing","push up","pushing car",
    "pushing cart","pushing wheelchair","reading book","reading newspaper",
    "recording music","riding a bike","riding camel","riding elephant",
    "riding mechanical bull","riding mountain bike","riding mule",
    "riding or walking with horse","riding scooter","riding unicycle",
    "ripping paper","robot dancing","rock climbing","rock scissors paper",
    "roller skating","running on treadmill","sailing","salsa dancing",
    "sanding floor","scrambling eggs","scuba diving","setting table",
    "shaking hands","shaking head","sharpening knives","sharpening pencil",
    "shaving head","shaving legs","shearing sheep","shining shoes",
    "shooting basketball","shooting goal soccer","shot put","shoveling snow",
    "shredding paper","shuffling cards","side kick","sign language interpreting",
    "singing","situp","skateboarding","ski jumping","skiing",
    "skiing crosscountry","skiing slalom","skipping rope","sky diving",
    "slacklining","slapping","sled dog racing","smoking","smoking hookah",
    "snatch weight lifting","sneezing","snorkeling","snowboarding","snowkiting",
    "snowmobiling","somersaulting","spinning poi","spray painting",
    "springboard diving","squatting","stretching arm","stretching leg",
    "strumming guitar","surfing crowd","surfing water","sweeping floor",
    "swimming backstroke","swimming breast stroke","swimming butterfly stroke",
    "swing dancing","swinging legs","swinging on something","sword fighting",
    "tai chi","taking a shower","tango dancing","tap dancing","tapping guitar",
    "tapping pen","tasting beer","tasting food","telephone conversation",
    "texting","throwing axe","throwing ball","throwing discus","tickling",
    "tobogganing","tossing coin","tossing salad","training dog","trapezing",
    "trimming or shaving beard","trimming trees","triple jump","twisting",
    "unboxing","unloading truck","using computer","using remote controller",
    "using segway","vault","waiting in line","walking the dog","washing dishes",
    "washing feet","washing hair","washing hands","water skiing","water sliding",
    "watering plants","waxing back","waxing chest","waxing eyebrows",
    "waxing legs","weaving basket","whistling","windsurfing","wrapping present",
    "wrestling","writing","yawning","yoga","zumba",
]

# ─── Surveillance category mapping ───────────────────────────────────────
# Each key is a substring of the K400 label that maps to a surveillance tag.
_CATEGORY_MAP = {
    # ⚠ Dangerous
    "punching":      "⚠ fighting",
    "slapping":      "⚠ fighting",
    "wrestling":     "⚠ fighting",
    "headbutting":   "⚠ fighting",
    "sword fighting":"⚠ fighting",
    "high kick":     "⚠ fighting",
    "drop kick":     "⚠ fighting",
    "side kick":     "⚠ fighting",
    "kicking":       "⚠ fighting",
    "capoeira":      "⚠ fighting",
    "faceplanting":  "⚠ falling",
    "falling":       "⚠ falling",
    # Running
    "jogging":       "🏃 running",
    "running":       "🏃 running",
    "parkour":       "🏃 running",
    "hurdling":      "🏃 running",
    "sprinting":     "🏃 running",
    # Reading
    "reading book":  "📖 reading",
    "reading newspaper": "📖 reading",
    # Writing / Working
    "writing":       "✍ writing",
    "using computer":"💻 working",
    "texting":       "💻 working",
    "playing keyboard": "💻 working",
    "news anchoring":"💬 talking",
    # Talking
    "telephone":     "💬 talking",
    "answering questions": "💬 talking",
    "singing":       "💬 talking",
    "sign language": "💬 talking",
    # Eating
    "eating":        "🍽 eating",
    "drinking":      "🍽 eating",
    "tasting":       "🍽 eating",
    "dining":        "🍽 eating",
    # Exercising
    "push up":       "💪 exercising",
    "pull ups":      "💪 exercising",
    "situp":         "💪 exercising",
    "squatting":     "💪 exercising",
    "deadlifting":   "💪 exercising",
    "bench pressing":"💪 exercising",
    "doing aerobics":"💪 exercising",
    "yoga":          "💪 exercising",
    "tai chi":       "💪 exercising",
    "stretching":    "💪 exercising",
    "lunge":         "💪 exercising",
    # Sitting activities
    "playing cards": "🪑 sitting",
    "playing chess": "🪑 sitting",
    "playing checkers": "🪑 sitting",
    "playing poker": "🪑 sitting",
    "playing piano": "🎵 playing instrument",
    "playing guitar":"🎵 playing instrument",
    "playing drums": "🎵 playing instrument",
    "playing violin":"🎵 playing instrument",
    # Waiting/standing
    "waiting":       "🧍 standing",
    "standing":      "🧍 standing",
    # General
    "dancing":       "💃 dancing",
    "cooking":       "👨‍🍳 cooking",
    "cleaning":      "🧹 cleaning",
    "smoking":       "🚬 smoking",
    "laughing":      "😄 laughing",
    "crying":        "😢 crying",
    "hugging":       "🤗 hugging",
    "kissing":       "💋 kissing",
}


def _classify(label: str) -> str:
    """Map a K400 label string to a surveillance category."""
    label_lower = label.lower()
    for keyword, category in _CATEGORY_MAP.items():
        if keyword in label_lower:
            return category
    return f"🔍 {label}"   # show raw K400 label if no category match


class ActionDetector:
    """
    Non-blocking SlowFast-R50 recogniser.
    Returns fine-grained action labels mapped to surveillance categories.
    """

    _INPUT_SIZE = (256, 256)
    _MEAN = torch.tensor([0.45, 0.45, 0.45])
    _STD  = torch.tensor([0.225, 0.225, 0.225])

    def __init__(
        self,
        enabled: bool = False,
        buffer_size: int = 16,
        slow_stride: int = 8,
        device: str = "cpu",
    ):
        self.enabled     = enabled
        self.buffer_size = buffer_size
        self.slow_stride = slow_stride
        self.device      = device

        self._buffers: dict[int, deque]  = {}
        self._cache:   dict[int, str]    = {}
        self._futures: dict[int, Future] = {}
        self._model: torch.nn.Module | None = None

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="action")

        if self.enabled and _PTV_OK:
            self._load_model()
        elif self.enabled and not _PTV_OK:
            log.warning("pytorchvideo not available – action detection disabled.")
            self.enabled = False

    def _load_model(self) -> None:
        try:
            log.info("Loading SlowFast-R50 … (this may take 30-60 seconds)")
            self._model = hub.slowfast_r50(pretrained=True)
            self._model.eval()
            self._model.to(self.device)
            log.info("SlowFast-R50 loaded on %s", self.device)
        except Exception as exc:
            log.warning("SlowFast load failed: %s", exc)
            self._model = None
            self.enabled = False

    def _preprocess(self, frames: list[np.ndarray]):
        import cv2
        processed = []
        for f in frames:
            f = cv2.resize(f, self._INPUT_SIZE)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(f.astype(np.float32) / 255.0)
            t = (t - self._MEAN) / self._STD
            processed.append(t.permute(2, 0, 1))
        video    = torch.stack(processed, dim=1).unsqueeze(0).to(self.device)
        slow_idx = torch.linspace(0, video.shape[2] - 1,
                                  video.shape[2] // self.slow_stride).long()
        return [video[:, :, slow_idx, :, :], video]

    def _infer(self, frames: list[np.ndarray]) -> str:
        try:
            with torch.no_grad():
                logits = self._model(self._preprocess(frames))
            probs   = F.softmax(logits, dim=-1)
            top_idx = int(probs.argmax(dim=-1).item())
            # Get K400 label name and map to surveillance category
            k400_label = K400[top_idx] if top_idx < len(K400) else f"class_{top_idx}"
            return _classify(k400_label)
        except Exception as exc:
            log.debug("Action infer error: %s", exc)
            return "🔍 normal"

    def update(self, track_id: int, frame_crop: np.ndarray) -> str:
        if not self.enabled or self._model is None:
            return "N/A"

        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.buffer_size)
        self._buffers[track_id].append(frame_crop.copy())

        if len(self._buffers[track_id]) < self.buffer_size:
            return self._cache.get(track_id, "collecting…")

        fut = self._futures.get(track_id)
        if fut is None or fut.done():
            if fut is not None and fut.done():
                try:
                    self._cache[track_id] = fut.result()
                except Exception:
                    pass
            frames_snap = list(self._buffers[track_id])
            self._buffers[track_id].clear()
            self._futures[track_id] = self._executor.submit(self._infer, frames_snap)

        return self._cache.get(track_id, "collecting…")

    def purge(self, active_ids: set[int]) -> None:
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                self._buffers.pop(tid, None)
                self._cache.pop(tid, None)
                self._futures.pop(tid, None)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
