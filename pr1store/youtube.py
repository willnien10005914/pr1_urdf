"""YouTube URL helpers. Downloads stay in the retarget script."""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        query = parse_qs(parsed.query)
        if query.get("v"):
            video_id = query["v"][0]
        else:
            match = re.search(r"(?:shorts|embed|live)/([^/?&#]+)", parsed.path or "")
            video_id = match.group(1) if match else ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id or ""):
        raise ValueError("That link does not contain a YouTube video id.")
    return video_id


def require_youtube(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if host not in HOSTS:
        raise ValueError("Paste a youtube.com or youtu.be link.")
    return youtube_video_id(url)
