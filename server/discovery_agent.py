"""Background discovery workers for the "find similar artists" flow.

Three independent ``BaseWorker`` peers that the UI worker fans out to via
``start_ui_job_group``. Each worker handles the same ``discover_similar``
job name with its own data source, streams 2-3 ``send_job_update``
messages so the client's progress card has something to render, and
finishes with ``{"artists": [{...}]}``. ``UIWorker`` auto-forwards every
lifecycle envelope to the client as a ``ui-job-group`` message; we don't
have to plumb anything by hand.

Each handler treats ``asyncio.CancelledError`` as the user clicking
Cancel on the progress card (re-raise so the base worker emits a
CANCELLED response) and any other exception as a worker-level failure
(``_respond_error`` mirrors ``CatalogWorker``'s pattern and sends a
FAILED response with the message text).

Why three workers instead of one with three steps? The plural showcases
the fan-out — the user sees three concurrent rows in the progress card,
each finishing on its own clock. Real apps would wire each worker to a
genuinely different data source (a recsys, a graph DB, an editorial
playlist provider, ...); here they're three different angles on the
Deezer/OpenAI tools we already have.
"""

import asyncio
import os
from collections.abc import Iterable

from loguru import logger
from openai import AsyncOpenAI
from pipecat.bus import BusJobRequestMessage
from pipecat.pipeline.job_context import JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.workers.base_worker import BaseWorker

import deezer

# Deezer ``/related`` returns up to ~20 hits; we surface a richer list here
# than the 6-cap used by the quick ``switch_tab("related")`` flow.
RELATED_LIMIT = 20

# Top-tracks pool size for the cross-reference worker. We then pull each
# track's neighbouring artists out of Deezer's search hits.
TOP_TRACKS_POOL = 8
CROSSREF_LIMIT = 12

# OpenAI worker keeps the prompt short and asks for a fixed-shape list.
LLM_SUGGESTIONS = 8


def _minimal_artist(deezer_artist: dict) -> dict:
    """Match ``CatalogWorker._minimal_artist`` so progress cards render uniformly."""
    return {
        "id": str(deezer_artist.get("id", "")),
        "name": deezer_artist.get("name") or "",
        "image_url": (
            deezer_artist.get("picture_xl")
            or deezer_artist.get("picture_big")
            or deezer_artist.get("picture_medium")
            or ""
        ),
    }


def _dedupe(artists: Iterable[dict], *, exclude_id: str | None = None) -> list[dict]:
    """Stable dedupe by id (or normalized name when id is missing)."""
    out: list[dict] = []
    seen: set[str] = set()
    for a in artists:
        aid = str(a.get("id") or "").strip()
        name = a.get("name") or ""
        key = aid or deezer.normalize_name(name)
        if not key or key in seen:
            continue
        if exclude_id and aid == exclude_id:
            continue
        seen.add(key)
        out.append(a)
    return out


class _DiscoveryWorker(BaseWorker):
    """Common error envelope for the three discovery workers."""

    async def _respond_error(self, job_id: str, exc: Exception) -> None:
        logger.exception(f"{self}: discover_similar failed")
        await self.send_job_response(job_id, response={"error": str(exc)}, status=JobStatus.FAILED)


class DeezerRelatedWorker(_DiscoveryWorker):
    """The quick, high-confidence source: Deezer's own ``/related`` endpoint.

    Same data the Related tab uses, but with a larger limit so the card
    has more to show than the abbreviated tab grid.
    """

    @job(name="discover_similar")
    async def discover_similar(self, message: BusJobRequestMessage) -> None:
        job_id = message.job_id
        payload = message.payload or {}
        artist_id = str(payload.get("artist_id") or "")
        artist_name = payload.get("artist_name") or "this artist"
        try:
            await self.send_job_update(
                job_id, {"text": f"Asking Deezer for artists like {artist_name}…"}
            )
            hits = await deezer.get_related_artists(artist_id, limit=RELATED_LIMIT)
            artists = _dedupe((_minimal_artist(h) for h in hits), exclude_id=artist_id)
            await self.send_job_update(job_id, {"text": f"Found {len(artists)} on Deezer."})
            await self.send_job_response(job_id, response={"artists": artists})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._respond_error(job_id, exc)


class TopTracksCrossrefWorker(_DiscoveryWorker):
    """The crossover source: artists who turn up next to the seed in search.

    For each of the artist's top tracks we run a global track search and
    collect the *other* artists Deezer surfaces, weighted by how often
    they appear. This catches collaborators, scene peers, and same-album
    artists that ``/related`` (which is rec-system-based) doesn't always
    include.
    """

    @job(name="discover_similar")
    async def discover_similar(self, message: BusJobRequestMessage) -> None:
        job_id = message.job_id
        payload = message.payload or {}
        artist_id = str(payload.get("artist_id") or "")
        artist_name = payload.get("artist_name") or "this artist"
        try:
            await self.send_job_update(job_id, {"text": f"Pulling top tracks for {artist_name}…"})
            top = await deezer.get_artist_top_tracks(artist_id, limit=TOP_TRACKS_POOL)
            titles = [t.get("title") or "" for t in top if t.get("title")]
            if not titles:
                await self.send_job_response(job_id, response={"artists": []})
                return

            await self.send_job_update(
                job_id,
                {"text": f"Cross-referencing {len(titles)} tracks across Deezer search…"},
            )
            searches = await asyncio.gather(
                *(deezer.search_track(title, limit=8) for title in titles),
                return_exceptions=True,
            )

            # Score by appearance count (more co-occurrence = stronger signal).
            scores: dict[str, int] = {}
            seen: dict[str, dict] = {}
            for hits in searches:
                if isinstance(hits, Exception):
                    continue
                for hit in hits or []:
                    artist = (hit or {}).get("artist") or {}
                    aid = str(artist.get("id") or "")
                    if not aid or aid == artist_id:
                        continue
                    scores[aid] = scores.get(aid, 0) + 1
                    if aid not in seen:
                        seen[aid] = _minimal_artist(artist)

            ranked = sorted(seen.values(), key=lambda a: -scores.get(a["id"], 0))
            artists = ranked[:CROSSREF_LIMIT]
            await self.send_job_update(
                job_id, {"text": f"Ranked {len(artists)} co-occurring artists."}
            )
            await self.send_job_response(job_id, response={"artists": artists})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._respond_error(job_id, exc)


class LLMSuggestionsWorker(_DiscoveryWorker):
    """The model-knowledge source: one OpenAI call for editorial suggestions.

    Uses ``AsyncOpenAI`` directly to mirror ``descriptions.py`` — the
    suggestion list is short, fixed-shape, and doesn't need the full
    pipecat LLMService machinery. Slowest of the three; the client sees
    its row spin a few seconds longer than the others.

    Names are returned without ids or images (the LLM doesn't know
    Deezer ids). The client renders them as plain text rows.
    """

    _PROMPT = (
        "List exactly {n} artists similar in style to {artist_name}"
        "{genre_clause}. One artist name per line. No numbering, no "
        "explanations, no extra punctuation. Use the artists' commonly "
        "recognized stage or band names."
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._client: AsyncOpenAI | None = None
        self._model = os.getenv("OPENAI_MODEL")

    def _openai(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    @job(name="discover_similar")
    async def discover_similar(self, message: BusJobRequestMessage) -> None:
        job_id = message.job_id
        payload = message.payload or {}
        artist_name = (payload.get("artist_name") or "").strip() or "this artist"
        genre = (payload.get("genre") or "").strip()
        try:
            await self.send_job_update(
                job_id, {"text": f"Asking the model for artists like {artist_name}…"}
            )
            prompt = self._PROMPT.format(
                n=LLM_SUGGESTIONS,
                artist_name=artist_name,
                genre_clause=f" (genre: {genre})" if genre else "",
            )
            client = self._openai()
            completion = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200,
            )
            text = (completion.choices[0].message.content or "").strip()
            names = [line.strip(" -•\t").rstrip(".") for line in text.splitlines() if line.strip()]
            seen: set[str] = set()
            artists: list[dict] = []
            for name in names:
                key = deezer.normalize_name(name)
                if not key or key in seen:
                    continue
                if key == deezer.normalize_name(artist_name):
                    continue
                seen.add(key)
                # No id / image_url — the LLM only gives us the name. The
                # client renders LLM-only rows as plain text.
                artists.append({"id": "", "name": name, "image_url": ""})
            await self.send_job_update(job_id, {"text": f"Got {len(artists)} suggestions."})
            await self.send_job_response(job_id, response={"artists": artists})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._respond_error(job_id, exc)
