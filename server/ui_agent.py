"""UI worker: owns navigation stack and UI state.

Two entry points:

- The inherited ``respond`` job: the voice layer delegates a
  natural-language request. The worker's LLM picks one tool, grounded by
  the current screen (``render_ui_state``), with no conversation history.
- ``@ui_event`` handlers: the client sends a UI event (grid click or
  Detail-screen button) via ``sendUIEvent``. Dispatched directly to
  helper methods without an LLM call; they mutate the nav state so the
  next voice turn sees the new screen.

All UI changes fan out through ``send_command``: ``PipelineWorker``
translates each command into an ``RTVIEvent.UICommand`` for the client.
Every catalog lookup (seed listing, artist fetch, title resolution,
description generation) goes through the long-lived ``CatalogWorker``
via a ``@job``.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.llm import tool
from pipecat.workers.ui import UIWorker, ui_event

import descriptions
from llm import create_llm_service

Screen = Literal["home", "artist", "detail", "trending"]
Kind = Literal["album", "song"]
SectionStatus = Literal["running", "completed", "cancelled", "error"]

# The three discovery workers, in the order they appear on the Discovery
# screen. Single source of truth for ``discover_similar_artists`` —
# ``_DISCOVERY_LABELS`` maps each worker to a UI section heading.
DISCOVERY_WORKERS: tuple[str, ...] = (
    "discovery_deezer",
    "discovery_tracks",
    "discovery_llm",
)
_DISCOVERY_LABELS: dict[str, str] = {
    "discovery_deezer": "Deezer related",
    "discovery_tracks": "Co-occurring on top tracks",
    "discovery_llm": "Model suggestions",
}


SYSTEM_PROMPT = """\
You control a voice-driven music player backed by a live music catalog. \
You never speak to the user directly. You always call exactly one tool \
per turn.

## UI layers
- **Home**: three 8-column grids stacked top to bottom — Trending \
artists, New releases (recent albums with their artist names), and \
Favorites. Position references on Home resolve against the section \
the user names ("the first new release", "bottom left favorite").
- **Artist**: an artist page with three tabs — Albums, Songs, and \
Related artists. Only one tab's grid is visible at a time (8 columns \
wide). The ``<ui_state>`` block describes the currently active \
tab; position references like "top right" resolve against that \
tab's grid.
- **Detail**: an album or song page with Play, More Info, and Add to \
Favorites action buttons.
- **Trending**: a grid of currently-popular artists, optionally \
scoped to a genre.

## Tools
- ``navigate_to_artist(artist_name)``: Push the artist screen. Use \
when the user names an artist ("show me Nirvana", "show me Daft \
Punk") or refers to an artist on the home or trending grid by \
position. Any artist in the catalog is fair game, not just the \
seeded lineup.
- ``select_item(item_title)``: Push the detail screen for an album or \
song. Works from any screen. If the item lives under a different \
artist, the server navigates through that artist's page first so \
"go back" lands on it.
- ``play(item_title)``: Play the named album or song. Works from any \
screen; navigates to its detail page and starts playback.
- ``control_playback(action)``: Control the currently-playing preview. \
``action`` is ``"pause"``, ``"resume"``, or ``"stop"``. Use for "pause", \
"resume", "continue", "stop", "mute this".
- ``show_info(title)``: Show a description toast for a named item. \
``title`` may name an album, a song, or an artist; the server resolves \
all three. Works from any screen. Use when the user asks "tell me \
about X" and X is a specific album/song/artist they named.
- ``answer_about_catalog(question, about=None)``: Answer a factual \
question about the artist in focus using their catalog (latest album, \
first album, release year, track count, duration). Speaks a short \
answer. Pass ``about`` with the item title only if the answer pivots \
on one specific album or song the user should see a toast for.
- ``answer_about_music(question, about=None)``: Answer an opinion or \
trivia question about the current artist (most popular album, best \
entry point, who influenced them, are they still active). Uses the \
model's general music knowledge, grounded by the catalog. Speaks a \
short answer. Pass ``about`` only when the answer centers on a \
specific album or song.
- ``add_to_favorites(item_title)``: Mark an album or song as a \
favorite. Works from any screen.
- ``switch_tab(tab)``: Switch the current Artist page to its Albums, \
Songs, or Related Artists tab. ``tab`` is ``"albums"``, ``"songs"``, \
or ``"related"``. Only valid on an Artist screen. The Related tab \
runs a multi-source similar-artists search the first time it's \
opened (Deezer related, top-track crossref, model suggestions); \
results stream into three sections on the tab as each background \
worker finishes. Use for "show albums", "show songs", "show \
tracks", "who's similar", "show me artists like them", "more like \
this", "show related", "discover similar artists", "find me more \
like this one", "explore artists like them".
- ``show_trending(genre)``: Push a Trending screen. ``genre`` is an \
optional string like "rock", "pop", "hip-hop"; omit for the global \
chart. Use for "what's trending", "what's popular in rock", or \
anything chart-adjacent.
- ``go_back()``: Pop one screen off the navigation stack.
- ``go_home()``: Reset to the home grid.
- ``answer_about_screen(answer)``: Answer a question using only what's \
currently on screen — the ``<ui_state>`` block. Use for "what's the \
first track", the order or position of grid items, the current artist \
or album, "what's on screen", "where am I". You compose the spoken \
answer; it's read aloud verbatim. Read-only.

## Decision rules
1. Every turn picks exactly one tool. Never reply with plain text.
2. If the user refers to an item by position ("top right", "the first \
one", "second album"), resolve the position from the most recent \
``<ui_state>`` grid layout in your context, then pass the resolved \
title to the tool.
3. If the user names a specific artist, album, or song, pass that \
name verbatim to the tool; the server resolves it case-insensitively \
against the live catalog.
4. When the user names a specific album or song title, call \
``select_item``, ``play``, ``show_info``, or ``add_to_favorites`` \
directly. Prefer ``navigate_to_artist`` only when the user names an \
artist without a specific title.
5. Use ``switch_tab(tab)`` to switch the Artist page tab when the \
user asks for one of those categories in the abstract ("show me the \
albums", "who's similar", "show related", "discover similar \
artists"). Use ``show_trending`` for popularity / chart questions.
6. If the answer is visible in the current ``<ui_state>`` (a tracklist, \
the grid contents, the order or position of items, the current artist \
or album), answer it directly with ``answer_about_screen``. Use \
``answer_about_catalog`` or ``answer_about_music`` only for facts that \
are NOT on screen — release years, discography-wide facts, opinions, \
or trivia. For a specific named item, use ``show_info`` or \
``select_item``.

## UI context
A ``<ui_state>`` block in your context describes the current \
screen and grid layouts. Grid descriptions use the form \
"row R col C: <title>". Resolve position references against the \
columns reported in the most recent grid description, for example:
- "top left" is row 1 col 1.
- "top right" is row 1 col N, where N is the last column.
- "bottom left" is row 2 col 1.
- "bottom right" is row 2 col N."""


@dataclass
class NavFrame:
    """One entry on the UI agent's navigation stack."""

    screen: Screen
    artist_id: str | None = None
    kind: Kind | None = None
    item_id: str | None = None
    # Only populated when screen == "trending".
    trending_genre: str | None = None


ArtistTab = Literal["albums", "songs", "related"]


@dataclass
class UIState:
    """Internal UI state mirroring what the client is rendering."""

    stack: list[NavFrame] = field(default_factory=lambda: [NavFrame(screen="home")])
    favorite_keys: set[str] = field(default_factory=set)
    favorites: list[dict] = field(default_factory=list)
    playing: dict | None = None
    playing_artist_id: str | None = None
    # Session-scoped artist cache populated whenever the CatalogWorker
    # hands us a full artist dict. Drained only when the worker restarts.
    artist_cache: dict[str, dict] = field(default_factory=dict)
    # Per-artist active tab on the Artist screen. Missing entries default
    # to "albums". Persists across nav-stack pushes so returning to an
    # artist keeps the tab the user picked.
    active_tab_by_artist: dict[str, ArtistTab] = field(default_factory=dict)


class MusicUIWorker(UIWorker):
    """Owns UI state and routes voice requests / client clicks to UI actions.

    The voice layer dispatches a ``respond`` job per utterance; the
    worker's LLM picks one tool, drives the client with ``send_command``,
    and replies via ``respond_to_job``. Client clicks arrive as
    ``@ui_event`` events and update state directly, without an LLM turn.
    """

    def __init__(self):
        llm = create_llm_service(system_prompt=SYSTEM_PROMPT)
        # This app drives a server-owned UI through custom commands, not the
        # accessibility-snapshot protocol, so the wire-format prompt guide is
        # disabled and ``render_ui_state`` is overridden to surface the current
        # screen from ``self._state``. No conversation history is kept: the
        # voice layer owns the dialogue and sends self-contained queries, so the
        # worker only needs the current screen, which the auto-inject hook
        # re-injects each turn. Client clicks mutate state directly, so their
        # events aren't injected.
        super().__init__(
            "ui",
            llm=llm,
            inject_events=False,
            prompt_guide=None,
        )
        self._state = UIState()
        # The current screen rendered as text (grid layouts + state), kept up to
        # date by the ``_emit_*`` methods and surfaced via ``render_ui_state``.
        self._screen_state = ""

    # ------------------------------------------------------------------
    # Client click events (sent via ``sendUIEvent``; no LLM turn).
    #
    # ``@ui_event`` handlers are dispatched in their own task, so awaiting
    # a catalog ``@job`` here can't deadlock the bus dispatcher.
    # ------------------------------------------------------------------

    @ui_event("hello")
    async def on_hello(self, message) -> None:
        # The client emits ``hello`` after the RTVI handshake completes.
        # Re-emit the top of the nav stack so the view is correct on
        # connect and after reconnect.
        await self._emit_for_top()

    @ui_event("nav")
    async def on_nav(self, message) -> None:
        await self._handle_nav_click(message.payload or {})

    @ui_event("action")
    async def on_action(self, message) -> None:
        await self._handle_action_click(message.payload or {})

    @ui_event("set_tab")
    async def on_set_tab(self, message) -> None:
        await self._handle_set_tab_click(message.payload or {})

    @ui_event("play_track")
    async def on_play_track(self, message) -> None:
        await self._handle_play_track_click(message.payload or {})

    @ui_event("stop_playback")
    async def on_stop_playback(self, message) -> None:
        # Header-level Stop. Click carries no payload — we just stop
        # whatever's currently playing, no-op if nothing is. The
        # corresponding voice path is ``control_playback("stop")``.
        if self._state.playing is None:
            return
        await self.send_command("playback_control", {"action": "stop"})
        await self._do_stop_playback()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool
    async def navigate_to_artist(self, params: FunctionCallParams, artist_name: str):
        """Push the artist screen for the named artist.

        Args:
            artist_name: The artist's display name (e.g. "Nirvana").
        """
        logger.info(f"{self}: navigate_to_artist('{artist_name}')")
        artist = await self._catalog_find_artist(artist_name)
        if not artist:
            await self.respond_to_job(
                f"I could not find {artist_name} in the library.", tts_speak=True
            )
            await params.result_callback(None)
            return
        await self._do_navigate_to_artist(artist)
        await self.respond_to_job(f"Here's {artist['name']}.", tts_speak=True)
        await params.result_callback(None)

    @tool
    async def select_item(self, params: FunctionCallParams, item_title: str):
        """Push the detail screen for an album or song.

        Args:
            item_title: The album or song title.
        """
        logger.info(f"{self}: select_item('{item_title}')")
        resolved = await self._catalog_resolve_item(item_title)
        if not resolved:
            await self.respond_to_job(
                f"I could not find {item_title} in the library.", tts_speak=True
            )
            await params.result_callback(None)
            return
        artist = resolved["artist"]
        kind: Kind = resolved["kind"]
        item = resolved["item"]
        await self._do_select_item(artist, kind, item)
        await self.respond_to_job(f"Here's {item['title']}.", tts_speak=True)
        await params.result_callback(None)

    @tool
    async def play(self, params: FunctionCallParams, item_title: str):
        """Play an album or song, navigating to its detail first.

        Args:
            item_title: The album or song title to play.
        """
        logger.info(f"{self}: play('{item_title}')")
        resolved = await self._catalog_resolve_item(item_title)
        if not resolved:
            await self.respond_to_job(
                f"I could not find {item_title} in the library.", tts_speak=True
            )
            await params.result_callback(None)
            return
        artist = resolved["artist"]
        kind: Kind = resolved["kind"]
        item = resolved["item"]
        # If the user is looking at an album detail, treat "play X" as
        # "play this track from the album" so the album stays in focus
        # and the track row flips to Stop.
        top = self._top()
        if kind == "song" and top.screen == "detail" and top.kind == "album" and top.item_id:
            cached_artist = self._get_cached_artist(top.artist_id or "")
            album = (
                self._find_item_in_artist(cached_artist, "album", top.item_id)
                if cached_artist
                else None
            )
            track = self._find_track_in_album(album, item) if album else None
            if cached_artist and album and track:
                await self._do_play_track(cached_artist, album, track)
                await self.respond_to_job(f"Playing {track['title']}.", tts_speak=True)
                await params.result_callback(None)
                return
        await self._do_play(artist, kind, item)
        await self.respond_to_job(f"Playing {item['title']}.", tts_speak=True)
        await params.result_callback(None)

    @tool
    async def show_info(self, params: FunctionCallParams, title: str):
        """Show a description toast for any album, song, or artist.

        Args:
            title: An album title, a song title, or an artist name.
        """
        logger.info(f"{self}: show_info('{title}')")

        # Try cached artists first so a recently-viewed artist is recognized
        # without another catalog round-trip.
        artist = self._find_cached_artist(title)
        if artist:
            long_desc = await self._catalog_get_description("artist", artist["id"], "long")
            await self._emit_artist_toast(artist, long_desc)
            await self.respond_to_job(
                long_desc or artist.get("short_description") or f"Here's {artist['name']}.",
                tts_speak=True,
            )
            await params.result_callback(None)
            return

        resolved = await self._catalog_resolve_item(title)
        if not resolved:
            await self.respond_to_job(f"I could not find {title} in the library.", tts_speak=True)
            await params.result_callback(None)
            return
        artist = resolved["artist"]
        kind: Kind = resolved["kind"]
        item = resolved["item"]
        long_desc = await self._catalog_get_description(kind, item["id"], "long")
        await self._emit_item_toast(artist, kind, item, long_desc)
        await self.respond_to_job(
            long_desc or item.get("short_description") or f"Here's {item['title']}.",
            tts_speak=True,
        )
        await params.result_callback(None)

    @tool
    async def add_to_favorites(self, params: FunctionCallParams, item_title: str):
        """Add an album or song to favorites.

        Args:
            item_title: The album or song title.
        """
        logger.info(f"{self}: add_to_favorites('{item_title}')")
        resolved = await self._catalog_resolve_item(item_title)
        if not resolved:
            await self.respond_to_job(
                f"I could not find {item_title} in the library.", tts_speak=True
            )
            await params.result_callback(None)
            return
        artist = resolved["artist"]
        kind: Kind = resolved["kind"]
        item = resolved["item"]
        description = await self._do_add_favorite(artist, kind, item)
        await self.respond_to_job(description, tts_speak=True)
        await params.result_callback(None)

    @tool
    async def control_playback(self, params: FunctionCallParams, action: str):
        """Pause, resume, or stop the current preview playback.

        Args:
            action: One of ``"pause"``, ``"resume"``, ``"stop"``.
        """
        logger.info(f"{self}: control_playback('{action}')")
        normalized = action.strip().lower()
        if normalized not in ("pause", "resume", "stop"):
            await self.respond_to_job(f"Unknown playback action: {action}.", tts_speak=True)
            await params.result_callback(None)
            return
        if self._state.playing is None:
            await self.respond_to_job("Nothing is playing.", tts_speak=True)
            await params.result_callback(None)
            return
        await self.send_command("playback_control", {"action": normalized})
        if normalized == "stop":
            await self._do_stop_playback()
            await self.respond_to_job("Stopped.", tts_speak=True)
        elif normalized == "pause":
            await self.respond_to_job("Paused.", tts_speak=True)
        else:
            await self.respond_to_job("Resuming.", tts_speak=True)
        await params.result_callback(None)

    @tool
    async def switch_tab(self, params: FunctionCallParams, tab: str):
        """Switch the current Artist page to its Albums, Songs, or Related Artists tab.

        Only valid on an Artist screen. Related artists are fetched on demand.

        Args:
            tab: One of ``"albums"``, ``"songs"``, or ``"related"``.
        """
        logger.info(f"{self}: switch_tab({tab!r})")
        normalized = tab.strip().lower()
        if normalized not in ("albums", "songs", "related"):
            await self.respond_to_job(f"Unknown tab: {tab}.", tts_speak=True)
            await params.result_callback(None)
            return
        artist = await self._current_artist_for_tab_switch()
        if artist is None:
            await self.respond_to_job(
                "I can only switch tabs while you're on an artist page.", tts_speak=True
            )
            await params.result_callback(None)
            return
        await self._activate_tab(artist, normalized)
        if normalized == "albums":
            speak = f"Here are {artist['name']}'s albums."
        elif normalized == "songs":
            speak = f"Here are {artist['name']}'s songs."
        else:  # related
            # If the Deezer section already completed (i.e. cached from
            # an earlier visit), name a few artists in the ack;
            # otherwise speak a "looking it up" placeholder while the
            # workers run in the background.
            sections = artist.get("related_sections") or []
            primary = next(
                (
                    s
                    for s in sections
                    if s["worker_name"] == "discovery_deezer"
                    and s["status"] == "completed"
                    and s["artists"]
                ),
                None,
            )
            if primary:
                names = ", ".join(a["name"] for a in primary["artists"][:4])
                speak = f"Artists similar to {artist['name']}: {names}."
            else:
                speak = f"Looking up artists similar to {artist['name']}."
        await self.respond_to_job(speak, tts_speak=True)
        await params.result_callback(None)

    # ------------------------------------------------------------------
    # Related-tab discovery: ``_activate_tab(artist, "related")`` kicks
    # off ``_run_related_discovery``. The three workers populate the
    # artist's ``related_sections`` in place and re-emit the Artist
    # screen as each finishes.
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_discovery_sections() -> list[dict]:
        return [
            {
                "worker_name": name,
                "label": _DISCOVERY_LABELS[name],
                "status": "running",
                "artists": [],
            }
            for name in DISCOVERY_WORKERS
        ]

    async def _run_related_discovery(self, artist: dict) -> None:
        """Fan three discovery workers out and stream into the Artist screen.

        Mutates ``artist['related_sections']`` in place. After each
        yielded job-group event we sweep ``tg.responses`` for any
        worker that completed since the previous tick and re-emit the
        Artist screen (if the user is still on it). ``job_group``
        (not ``ui_job_group``) since results land directly on the
        page; no client-side progress card subscribes to the
        lifecycle envelopes.
        """
        sections = artist["related_sections"]
        by_name = {s["worker_name"]: s for s in sections}
        try:
            async with self.job_group(
                *DISCOVERY_WORKERS,
                name="discover_similar",
                payload={
                    "artist_id": artist["id"],
                    "artist_name": artist["name"],
                    "genre": artist.get("genre") or "",
                },
            ) as tg:
                async for _ in tg:
                    await self._fold_in_responses(artist, sections, by_name, tg.responses)
                # Final sweep after the stream closes (catches a worker
                # whose response landed between the last yielded event
                # and group completion). Then mark any still-running
                # sections completed-with-no-results.
                await self._fold_in_responses(artist, sections, by_name, tg.responses)
                if self._finalize_sections(sections, "completed"):
                    await self._emit_artist_if_visible(artist)
        except asyncio.CancelledError:
            if self._finalize_sections(sections, "cancelled"):
                await self._emit_artist_if_visible(artist)
            raise
        except Exception:
            logger.exception(f"{self}: related discovery for {artist['name']} failed")
            if self._finalize_sections(sections, "error"):
                await self._emit_artist_if_visible(artist)

    async def _fold_in_responses(
        self,
        artist: dict,
        sections: list[dict],
        by_name: dict[str, dict],
        responses: dict[str, dict],
    ) -> bool:
        """Update any section whose worker just gained a response. True on change."""
        changed = False
        for worker_name, response in responses.items():
            section = by_name.get(worker_name)
            if section is None or section["status"] != "running":
                continue
            if "error" in (response or {}):
                section["status"] = "error"
                section["artists"] = []
            else:
                section["status"] = "completed"
                section["artists"] = (response or {}).get("artists") or []
            changed = True
        if changed:
            await self._emit_artist_if_visible(artist)
        return changed

    @staticmethod
    def _finalize_sections(sections: list[dict], status: SectionStatus) -> bool:
        """Force any still-running section to ``status``. True on change."""
        changed = False
        for section in sections:
            if section["status"] == "running":
                section["status"] = status
                changed = True
        return changed

    async def _emit_artist_if_visible(self, artist: dict) -> None:
        """Re-emit the Artist screen only if the user is still on this artist.

        Background discovery tasks can outlive the user's navigation
        away from an artist; without this guard, a late completion
        would clobber whatever screen the user is currently looking
        at. Results still land in ``artist['related_sections']`` so
        they show on return.
        """
        top = self._top()
        if top.screen == "artist" and top.artist_id == artist["id"]:
            await self._emit_artist(artist)

    @tool
    async def answer_about_catalog(
        self,
        params: FunctionCallParams,
        question: str,
        about: str | None = None,
    ):
        """Answer a factual question about the current artist's catalog.

        Args:
            question: The user's question, passed verbatim.
            about: Optional album, song, or artist title the answer \
                pivots on. When provided, the server raises a toast \
                for that item alongside the spoken answer.
        """
        await self._answer_question("catalog", question, about, params)

    @tool
    async def answer_about_music(
        self,
        params: FunctionCallParams,
        question: str,
        about: str | None = None,
    ):
        """Answer an opinion or trivia question about the current artist.

        Args:
            question: The user's question, passed verbatim.
            about: Optional album, song, or artist title the answer \
                pivots on. When provided, the server raises a toast \
                for that item alongside the spoken answer.
        """
        await self._answer_question("music", question, about, params)

    @tool
    async def show_trending(self, params: FunctionCallParams, genre: str | None = None):
        """Push a Trending screen. Optional ``genre`` like "rock" or "pop"."""
        logger.info(f"{self}: show_trending(genre={genre!r})")
        result = await self._catalog_get_trending(genre)
        artists = result.get("artists") or []
        label = result.get("label") or "Trending"
        genre_label = result.get("genre")
        self._enter(NavFrame(screen="trending", trending_genre=genre_label))
        await self._emit_trending(label, artists, genre_label)
        if artists:
            top_names = ", ".join(a["name"] for a in artists[:3])
            speak = f"Trending: {top_names}."
        else:
            speak = "I could not find a trending chart right now."
        await self.respond_to_job(speak, tts_speak=True)
        await params.result_callback(None)

    @tool
    async def go_back(self, params: FunctionCallParams):
        """Pop one screen off the navigation stack."""
        logger.info(f"{self}: go_back")
        description = await self._do_go_back()
        await self.respond_to_job(description, tts_speak=True)
        await params.result_callback(None)

    @tool
    async def go_home(self, params: FunctionCallParams):
        """Reset the navigation stack to the home grid."""
        logger.info(f"{self}: go_home")
        description = await self._do_go_home()
        await self.respond_to_job(description, tts_speak=True)
        await params.result_callback(None)

    @tool
    async def answer_about_screen(self, params: FunctionCallParams, answer: str):
        """Answer a question using only what's currently on screen. Read-only.

        Use when the answer is visible in the current ``<ui_state>`` (a
        tracklist, the grid contents, item order or position, the current
        artist or album). Compose the spoken answer from that context.

        Args:
            answer: The spoken answer. One short sentence, plain language.
        """
        logger.info(f"{self}: answer_about_screen('{answer[:60]}...')")
        await self.respond_to_job(answer, tts_speak=True)
        await params.result_callback(None)

    async def _answer_question(
        self,
        mode: str,
        question: str,
        about: str | None,
        params: FunctionCallParams,
    ) -> None:
        logger.info(f"{self}: answer_{mode}('{question}', about={about!r})")
        artist = self._current_context_artist()
        if artist is None:
            await self.respond_to_job("Pick an artist first and ask again.", tts_speak=True)
            await params.result_callback(None)
            return

        answer = await descriptions.answer_question(
            mode=mode,
            question=question,
            artist_name=artist["name"],
            albums=artist.get("albums") or [],
            songs=artist.get("songs") or [],
            about=about,
        )
        if not answer:
            await self.respond_to_job("I'm not sure about that one.", tts_speak=True)
            await params.result_callback(None)
            return

        if about:
            await self._emit_answer_toast(artist, about, answer)
        await self.respond_to_job(answer, tts_speak=True)
        await params.result_callback(None)

    def _current_context_artist(self) -> dict | None:
        """Best-effort: return the artist whose page the user is on."""
        for frame in reversed(self._state.stack):
            if frame.artist_id:
                cached = self._get_cached_artist(frame.artist_id)
                if cached:
                    return cached
        return None

    async def _emit_answer_toast(self, artist: dict, about: str, answer: str) -> bool:
        """Resolve ``about`` and raise a toast for the matching item.

        Returns True if a toast was emitted. Falls back to no toast
        (speech only) when the title can't be resolved.
        """
        target = (about or "").strip().lower()
        if not target:
            return False
        if target == (artist.get("name") or "").strip().lower():
            await self.send_command(
                "toast",
                {
                    "title": artist["name"],
                    "subtitle": artist.get("genre") or "Artist",
                    "image_url": artist.get("image_url") or "",
                    "description": answer,
                },
            )
            return True
        resolved = await self._catalog_resolve_item(about)
        if not resolved:
            return False
        resolved_artist = resolved["artist"]
        kind: Kind = resolved["kind"]
        item = resolved["item"]
        label = "Album" if kind == "album" else "Song"
        year = item.get("year")
        subtitle = f"{resolved_artist['name']} · {label}"
        if kind == "album" and year:
            subtitle = f"{subtitle} · {year}"
        await self.send_command(
            "toast",
            {
                "title": item["title"],
                "subtitle": subtitle,
                "image_url": item.get("cover_url") or resolved_artist.get("image_url") or "",
                "description": answer,
            },
        )
        return True

    # ------------------------------------------------------------------
    # Client click handlers (dispatched from the ``@ui_event`` methods above)
    # ------------------------------------------------------------------

    async def _handle_play_track_click(self, data: dict) -> None:
        artist = await self._catalog_get_artist(data.get("artist_id", ""))
        if not artist:
            return
        album = self._find_item_in_artist(artist, "album", data.get("album_id", ""))
        if not album:
            return
        track_id = data.get("track_id", "")
        # Toggle: re-clicking the active track stops playback.
        if (
            self._state.playing is not None
            and self._state.playing_artist_id == artist["id"]
            and self._state.playing.get("id") == track_id
        ):
            await self.send_command("playback_control", {"action": "stop"})
            await self._do_stop_playback()
            return
        tracks = album.get("tracks") or []
        if not tracks:
            tracks = await self._catalog_get_album_tracks(album["id"])
            album["tracks"] = tracks
        track = next((t for t in tracks if t["id"] == track_id), None)
        if not track:
            return
        await self._do_play_track(artist, album, track)

    async def _handle_set_tab_click(self, data: dict) -> None:
        tab = data.get("tab")
        if tab not in ("albums", "songs", "related"):
            return
        artist_id = data.get("artist_id") or ""
        artist = await self._catalog_get_artist(artist_id)
        if not artist:
            return
        await self._activate_tab(artist, tab)

    async def _handle_nav_click(self, data: dict) -> None:
        view = data.get("view")
        if view == "home":
            await self._do_go_home()
        elif view == "back":
            await self._do_go_back()
        elif view == "artist":
            artist = await self._catalog_get_artist(data.get("artist_id", ""))
            if not artist:
                return
            await self._do_navigate_to_artist(artist)
        elif view == "detail":
            artist = await self._catalog_get_artist(data.get("artist_id", ""))
            kind = data.get("detail_kind")
            item_id = data.get("item_id", "")
            if not artist or kind not in ("album", "song"):
                return
            item = self._find_item_in_artist(artist, kind, item_id)
            if not item:
                return
            await self._do_select_item(artist, kind, item)

    async def _handle_action_click(self, data: dict) -> None:
        action = data.get("action")
        artist = await self._catalog_get_artist(data.get("artist_id", ""))
        if not artist:
            return
        item_id = data.get("item_id", "")
        kind: Kind | None = None
        item: dict | None = None
        for k in ("album", "song"):
            found = self._find_item_in_artist(artist, k, item_id)
            if found:
                kind = k  # type: ignore[assignment]
                item = found
                break
        if not kind or item is None:
            return
        if action == "play":
            # Toggle: re-clicking Play on the currently-playing item is
            # treated as Stop. Mirrors ``_handle_play_track_click``. The
            # voice ``play()`` tool never goes through this path, so a
            # spoken "play X" twice in a row still re-plays (which is
            # what the user means there).
            if (
                self._state.playing is not None
                and self._state.playing_artist_id == artist["id"]
                and self._state.playing.get("id") == item["id"]
            ):
                await self.send_command("playback_control", {"action": "stop"})
                await self._do_stop_playback()
                return
            await self._do_play(artist, kind, item)
        elif action == "show_info":
            long_desc = await self._catalog_get_description(kind, item["id"], "long")
            await self._emit_item_toast(artist, kind, item, long_desc)
        elif action == "add_to_favorites":
            # Toggle: re-clicking on an already-favorited item removes
            # it. Mirrors the play/stop and tracklist toggles. The
            # voice ``add_to_favorites`` tool stays idempotent (saying
            # "add to favorites" twice doesn't undo it).
            if self._favorite_key(artist["id"], kind, item["id"]) in self._state.favorite_keys:
                await self._do_remove_favorite(artist, kind, item)
            else:
                await self._do_add_favorite(artist, kind, item)
        else:
            return

    # ------------------------------------------------------------------
    # Action helpers (shared by tools and click dispatcher)
    # ------------------------------------------------------------------

    async def _do_navigate_to_artist(self, artist: dict) -> None:
        self._enter(NavFrame(screen="artist", artist_id=artist["id"]))
        await self._emit_artist(artist)

    async def _do_select_item(self, artist: dict, kind: Kind, item: dict) -> None:
        top = self._top()
        if top.artist_id != artist["id"]:
            self._enter(NavFrame(screen="artist", artist_id=artist["id"]))
        self._enter(
            NavFrame(screen="detail", artist_id=artist["id"], kind=kind, item_id=item["id"])
        )
        await self._emit_detail(artist, kind, item)

    async def _do_play(self, artist: dict, kind: Kind, item: dict) -> None:
        top = self._top()
        already_on_detail = (
            top.screen == "detail" and top.artist_id == artist["id"] and top.item_id == item["id"]
        )
        if not already_on_detail:
            if top.artist_id != artist["id"]:
                self._enter(NavFrame(screen="artist", artist_id=artist["id"]))
            self._enter(
                NavFrame(screen="detail", artist_id=artist["id"], kind=kind, item_id=item["id"])
            )
            await self._emit_detail(artist, kind, item)
        preview_url = item.get("preview_url") or ""
        if kind == "album" and not preview_url:
            preview_url = await self._catalog_get_album_preview(item["id"])
            if preview_url:
                item["preview_url"] = preview_url
        self._state.playing = item
        self._state.playing_artist_id = artist["id"]
        await self.send_command(
            "playback",
            {
                "state": "playing",
                "item_title": item["title"],
                "item_id": item["id"],
                "preview_url": preview_url,
            },
        )
        await self._emit_detail(artist, kind, item)

    async def _do_play_track(self, artist: dict, album: dict, track: dict) -> None:
        """Play a single track from an album's tracklist, staying on the album page."""
        synthetic = {
            "id": track["id"],
            "title": track["title"],
            "album_id": album["id"],
            "duration_seconds": track.get("duration_seconds") or 0,
            "cover_url": album.get("cover_url") or "",
            "preview_url": track.get("preview_url") or "",
        }
        self._state.playing = synthetic
        self._state.playing_artist_id = artist["id"]
        await self.send_command(
            "playback",
            {
                "state": "playing",
                "item_title": track["title"],
                "item_id": track["id"],
                "preview_url": synthetic["preview_url"],
            },
        )
        await self._emit_detail(artist, "album", album)

    async def _do_stop_playback(self) -> None:
        self._state.playing = None
        self._state.playing_artist_id = None
        top = self._top()
        if top.screen == "detail" and top.artist_id and top.kind and top.item_id:
            artist = self._get_cached_artist(top.artist_id)
            if artist:
                item = self._find_item_in_artist(artist, top.kind, top.item_id)
                if item:
                    await self._emit_detail(artist, top.kind, item)

    async def _do_add_favorite(self, artist: dict, kind: Kind, item: dict) -> str:
        key = self._favorite_key(artist["id"], kind, item["id"])
        is_new = key not in self._state.favorite_keys
        if is_new:
            self._state.favorite_keys.add(key)
            self._state.favorites.append(self._favorite_record(artist, kind, item))
        await self.send_command(
            "favorite_added",
            {
                "favorite": self._favorite_record(artist, kind, item),
                "favorites": list(self._state.favorites),
            },
        )
        top = self._top()
        if top.screen == "detail" and top.artist_id == artist["id"] and top.item_id == item["id"]:
            await self._emit_detail(artist, kind, item)
        if not is_new:
            return f"{item['title']} is already in favorites."
        return f"Added {item['title']} to favorites."

    async def _do_remove_favorite(self, artist: dict, kind: Kind, item: dict) -> str:
        """Remove the item from favorites. Idempotent."""
        key = self._favorite_key(artist["id"], kind, item["id"])
        removed = key in self._state.favorite_keys
        if removed:
            self._state.favorite_keys.discard(key)
            self._state.favorites = [
                f
                for f in self._state.favorites
                if self._favorite_key(f["artist_id"], f["kind"], f["item_id"]) != key
            ]
        await self.send_command(
            "favorite_removed",
            {
                "favorite": self._favorite_record(artist, kind, item),
                "favorites": list(self._state.favorites),
            },
        )
        top = self._top()
        if top.screen == "detail" and top.artist_id == artist["id"] and top.item_id == item["id"]:
            await self._emit_detail(artist, kind, item)
        if not removed:
            return f"{item['title']} was not in favorites."
        return f"Removed {item['title']} from favorites."

    async def _do_go_back(self) -> str:
        if len(self._state.stack) > 1:
            self._state.stack.pop()
        top = self._top()
        await self._emit_for_top()
        if top.screen == "home":
            return "Back at the home grid."
        if top.screen == "artist":
            artist = self._get_cached_artist(top.artist_id or "")
            return f"Back on the {artist['name'] if artist else 'artist'} page."
        if top.screen == "trending":
            label = f"Trending · {top.trending_genre}" if top.trending_genre else "Trending"
            return f"Back on {label}."
        return "Back one screen."

    async def _do_go_home(self) -> str:
        self._state.stack = [NavFrame(screen="home")]
        await self._emit_home()
        return "Home grid is showing."

    # ------------------------------------------------------------------
    # Nav stack + caches
    # ------------------------------------------------------------------

    def _enter(self, frame: NavFrame) -> None:
        top = self._top()
        if top == frame:
            return
        self._state.stack.append(frame)

    def _top(self) -> NavFrame:
        return self._state.stack[-1]

    def _get_cached_artist(self, artist_id: str) -> dict | None:
        return self._state.artist_cache.get(artist_id)

    def _find_cached_artist(self, name: str) -> dict | None:
        target = name.strip().lower()
        for artist in self._state.artist_cache.values():
            if artist["name"].lower() == target or artist["id"] == target:
                return artist
        return None

    @staticmethod
    def _find_item_in_artist(artist: dict, kind: str, item_id: str) -> dict | None:
        coll = artist.get("albums", []) if kind == "album" else artist.get("songs", [])
        return next((i for i in coll if i["id"] == item_id), None)

    @staticmethod
    def _find_track_in_album(album: dict | None, song: dict) -> dict | None:
        """Match a resolved song dict against an album's loaded tracklist."""
        if not album:
            return None
        tracks = album.get("tracks") or []
        song_id = song.get("id")
        for t in tracks:
            if t["id"] == song_id:
                return t
        target = (song.get("title") or "").strip().lower()
        if not target:
            return None
        for t in tracks:
            if (t.get("title") or "").strip().lower() == target:
                return t
        return None

    def _cache_artist(self, artist: dict) -> None:
        self._state.artist_cache[artist["id"]] = artist

    @staticmethod
    def _favorite_key(artist_id: str, kind: Kind, item_id: str) -> str:
        return f"{artist_id}:{kind}:{item_id}"

    @staticmethod
    def _favorite_record(artist: dict, kind: Kind, item: dict) -> dict:
        return {
            "artist_id": artist["id"],
            "artist_name": artist["name"],
            "kind": kind,
            "item_id": item["id"],
            "item_title": item["title"],
            "cover_url": item.get("cover_url"),
        }

    # ------------------------------------------------------------------
    # CatalogWorker job calls
    # ------------------------------------------------------------------

    async def _catalog(self, name: str, payload: dict | None = None, *, timeout: float) -> dict:
        """Run the named catalog ``@job`` and return its response dict."""
        async with self.job("catalog", name=name, payload=payload, timeout=timeout) as t:
            pass
        return t.response or {}

    async def _catalog_list_home(self) -> list[dict]:
        response = await self._catalog("list_home", timeout=30)
        # Home records are minimal (id + name + image_url). Don't cache
        # them in ``_state.artist_cache`` — that cache is for full artist
        # dicts with albums/songs. Clicking a home cell goes through
        # ``_catalog_get_artist`` which triggers a full fetch on miss.
        return response.get("artists") or []

    async def _catalog_list_new_releases(self, limit: int = 12) -> list[dict]:
        response = await self._catalog("list_new_releases", {"limit": limit}, timeout=15)
        return response.get("releases") or []

    async def _catalog_find_artist(self, name: str) -> dict | None:
        response = await self._catalog("find_artist", {"name": name}, timeout=30)
        artist = response.get("artist")
        if artist:
            self._cache_artist(artist)
        return artist

    async def _catalog_get_artist(self, artist_id: str) -> dict | None:
        if not artist_id:
            return None
        cached = self._get_cached_artist(artist_id)
        if cached:
            return cached
        response = await self._catalog("get_artist", {"artist_id": artist_id}, timeout=15)
        artist = response.get("artist")
        if artist:
            self._cache_artist(artist)
            return artist
        # Not in the catalog's cache either — this happens for related /
        # trending artists the user just clicked. Fall back to a live
        # Deezer fetch keyed by id.
        response = await self._catalog("fetch_artist_by_id", {"artist_id": artist_id}, timeout=30)
        artist = response.get("artist")
        if artist:
            self._cache_artist(artist)
        return artist

    async def _catalog_get_trending(self, genre: str | None) -> dict:
        return await self._catalog("get_trending", {"genre": genre, "limit": 16}, timeout=15)

    async def _catalog_resolve_item(self, title: str) -> dict | None:
        prefer = self._top().artist_id
        response = await self._catalog(
            "resolve_item", {"title": title, "prefer_artist_id": prefer}, timeout=15
        )
        resolved = response.get("resolved")
        if resolved and resolved.get("artist"):
            self._cache_artist(resolved["artist"])
        return resolved

    async def _catalog_get_album_preview(self, album_id: str) -> str:
        response = await self._catalog("get_album_preview", {"album_id": album_id}, timeout=15)
        return response.get("preview_url", "") or ""

    async def _catalog_get_album_tracks(self, album_id: str) -> list[dict]:
        response = await self._catalog("get_album_tracks", {"album_id": album_id}, timeout=20)
        return response.get("tracks") or []

    async def _catalog_get_description(self, kind: str, id_: str, depth: str) -> str:
        response = await self._catalog(
            "get_description", {"kind": kind, "id": id_, "depth": depth}, timeout=30
        )
        return response.get("description", "") or ""

    # ------------------------------------------------------------------
    # UI command emission
    # ------------------------------------------------------------------

    async def _emit_home(self) -> None:
        artists, new_releases = await asyncio.gather(
            self._catalog_list_home(),
            self._catalog_list_new_releases(limit=16),
        )
        await self.send_command(
            "screen",
            {
                "screen": "home",
                "artists": artists,
                "new_releases": new_releases,
                "favorites": list(self._state.favorites),
            },
        )
        self._screen_state = self._describe_home_screen(
            artists, new_releases, self._state.favorites
        )

    async def _emit_artist(self, artist: dict) -> None:
        tab = self._get_artist_tab(artist["id"])
        await self.send_command(
            "screen",
            {
                "screen": "artist",
                "artist": artist,
                "active_tab": tab,
                "back_enabled": len(self._state.stack) > 1,
            },
        )
        self._screen_state = self._describe_artist_screen(artist)

    def _get_artist_tab(self, artist_id: str) -> ArtistTab:
        return self._state.active_tab_by_artist.get(artist_id, "albums")

    def _set_artist_tab(self, artist_id: str, tab: ArtistTab) -> None:
        self._state.active_tab_by_artist[artist_id] = tab

    async def _current_artist_for_tab_switch(self) -> dict | None:
        """Return the Artist currently shown on top of the nav stack."""
        top = self._top()
        if top.screen != "artist" or not top.artist_id:
            return None
        artist = self._get_cached_artist(top.artist_id)
        if not artist:
            artist = await self._catalog_get_artist(top.artist_id)
        return artist

    async def _activate_tab(self, artist: dict, tab: ArtistTab) -> None:
        """Flip the active tab.

        For ``related``, kicks off the multi-source discovery fan-out
        the first time the tab is opened for this artist. Sections
        populate in place and re-render via ``_emit_artist_if_visible``
        as each worker finishes; subsequent flips back to ``related``
        show the cached sections.
        """
        self._set_artist_tab(artist["id"], tab)
        if tab == "related" and not artist.get("related_sections"):
            artist["related_sections"] = self._initial_discovery_sections()
            self._cache_artist(artist)
            self.create_task(
                self._run_related_discovery(artist),
                f"related_discovery::{artist['id']}",
            )
        await self._emit_artist(artist)

    async def _emit_detail(self, artist: dict, kind: Kind, item: dict) -> None:
        if kind == "album" and not item.get("tracks"):
            tracks = await self._catalog_get_album_tracks(item["id"])
            if tracks:
                item["tracks"] = tracks
                if not item.get("preview_url"):
                    item["preview_url"] = tracks[0].get("preview_url", "")
        is_playing = (
            self._state.playing is not None
            and self._state.playing_artist_id == artist["id"]
            and self._state.playing.get("id") == item["id"]
        )
        await self.send_command(
            "screen",
            {
                "screen": "detail",
                "kind": kind,
                "item": item,
                "artist": artist,
                "is_favorite": self._favorite_key(artist["id"], kind, item["id"])
                in self._state.favorite_keys,
                "is_playing": is_playing,
                "playing_track_id": (
                    self._state.playing.get("id")
                    if self._state.playing and self._state.playing_artist_id == artist["id"]
                    else None
                ),
                "back_enabled": len(self._state.stack) > 1,
            },
        )
        self._screen_state = self._describe_detail_screen(artist, kind, item)

    async def _emit_trending(self, label: str, artists: list[dict], genre: str | None) -> None:
        await self.send_command(
            "screen",
            {
                "screen": "trending",
                "label": label,
                "genre": genre,
                "artists": artists,
                "back_enabled": len(self._state.stack) > 1,
            },
        )
        self._screen_state = self._describe_trending_screen(label, artists)

    async def _emit_for_top(self) -> None:
        top = self._top()
        if top.screen == "home":
            await self._emit_home()
        elif top.screen == "artist":
            artist = await self._catalog_get_artist(top.artist_id or "")
            if artist:
                await self._emit_artist(artist)
        elif top.screen == "detail":
            artist = await self._catalog_get_artist(top.artist_id or "")
            if artist and top.kind and top.item_id:
                item = self._find_item_in_artist(artist, top.kind, top.item_id)
                if item:
                    await self._emit_detail(artist, top.kind, item)
        elif top.screen == "trending":
            # Re-fetch trending on reconnect; charts change fast enough
            # that the previous list is stale.
            result = await self._catalog_get_trending(top.trending_genre)
            await self._emit_trending(
                result.get("label") or "Trending",
                result.get("artists") or [],
                result.get("genre"),
            )

    async def _emit_artist_toast(self, artist: dict, long_description: str) -> None:
        text = (
            long_description
            or artist.get("long_description")
            or artist.get("short_description")
            or ""
        )
        genre = artist.get("genre") or "Artist"
        await self.send_command(
            "toast",
            {
                "title": artist["name"],
                "subtitle": genre,
                "image_url": artist.get("image_url") or "",
                "description": text,
            },
        )

    async def _emit_item_toast(
        self, artist: dict, kind: Kind, item: dict, long_description: str
    ) -> None:
        text = (
            long_description or item.get("long_description") or item.get("short_description") or ""
        )
        label = "Album" if kind == "album" else "Song"
        year = item.get("year")
        subtitle = f"{artist['name']} · {label}"
        if kind == "album" and year:
            subtitle = f"{subtitle} · {year}"
        await self.send_command(
            "toast",
            {
                "title": item["title"],
                "subtitle": subtitle,
                "image_url": item.get("cover_url") or artist.get("image_url") or "",
                "description": text,
            },
        )

    # ------------------------------------------------------------------
    # LLM context
    # ------------------------------------------------------------------

    def render_ui_state(self) -> str:
        """Surface the current screen to the LLM as a ``<ui_state>`` block.

        This app has no accessibility snapshot; instead the ``_emit_*``
        methods record the current screen (grid layouts + state) in
        ``self._screen_state``. The auto-inject hook calls this before
        each turn, so the LLM can resolve position references ("top
        right", "the first one") and "go back" against what's on screen
        now -- without keeping any conversation history.
        """
        if not self._screen_state:
            return ""
        return f"<ui_state>\n{self._screen_state}\n</ui_state>"

    # ------------------------------------------------------------------
    # Screen descriptions (grid layout + state) for the LLM context
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_home_screen(
        artists: list[dict], new_releases: list[dict], favorites: list[dict]
    ) -> str:
        sections = [MusicUIWorker._describe_grid(artists, "Trending artists")]
        release_items = [
            {"title": f"{r.get('title', '')} — {r.get('artist_name', '')}"} for r in new_releases
        ]
        sections.append(MusicUIWorker._describe_grid(release_items, "New releases"))
        fav_items = [
            {"title": f"{f.get('item_title', '')} — {f.get('artist_name', '')}"} for f in favorites
        ]
        if fav_items:
            sections.append(MusicUIWorker._describe_grid(fav_items, "Favorites"))
        else:
            sections.append("Favorites grid: empty.")
        return "Home screen. " + " ".join(sections)

    @staticmethod
    def _describe_grid(items: list[dict], label: str, cols: int = 8) -> str:
        # Items are albums/songs (keyed by ``title``) or minimal artist
        # records (keyed by ``name``). Accept either.
        parts = [
            f"row {i // cols + 1} col {i % cols + 1}: "
            + str(item.get("title") or item.get("name") or "")
            for i, item in enumerate(items)
        ]
        rows = max(1, (len(items) - 1) // cols + 1) if items else 0
        return f"{label} grid ({rows} rows x {cols} columns): " + "; ".join(parts)

    def _describe_artist_screen(self, artist: dict) -> str:
        tab = self._get_artist_tab(artist["id"])
        if tab == "songs":
            grid_desc = self._describe_grid(artist.get("songs") or [], "Songs")
        elif tab == "related":
            grid_desc = self._describe_related_sections(artist.get("related_sections") or [])
        else:
            grid_desc = self._describe_grid(artist.get("albums") or [], "Albums")
        return (
            f"Artist screen: {artist['name']} ({tab} tab active). {grid_desc} "
            f"Tabs available: Albums, Songs, Related artists."
        )

    @staticmethod
    def _describe_related_sections(sections: list[dict]) -> str:
        """Render the Related tab's per-source sections for the LLM.

        Position references on the Related tab resolve against each
        labeled section, so we surface them the same way the user
        sees them. Running sections still get a placeholder so the
        LLM doesn't try to navigate from one that isn't ready yet.
        """
        if not sections:
            return "Related artists grid: empty (fetching)."
        parts: list[str] = []
        for section in sections:
            status = section.get("status", "running")
            artists = section.get("artists") or []
            label = f"{section['label']} ({status})"
            if status == "running":
                parts.append(f"{label}: still loading.")
            elif not artists:
                parts.append(f"{label}: no results.")
            else:
                parts.append(MusicUIWorker._describe_grid(artists, label, cols=8))
        return " ".join(parts)

    @staticmethod
    def _describe_trending_screen(label: str, artists: list[dict]) -> str:
        return f"{label} screen. " + MusicUIWorker._describe_grid(artists, "Trending", cols=8)

    def _describe_detail_screen(self, artist: dict, kind: Kind, item: dict) -> str:
        is_favorite = (
            self._favorite_key(artist["id"], kind, item["id"]) in self._state.favorite_keys
        )
        is_playing = (
            self._state.playing is not None
            and self._state.playing_artist_id == artist["id"]
            and self._state.playing.get("id") == item["id"]
        )
        flags = []
        if is_playing:
            flags.append("playing")
        if is_favorite:
            flags.append("favorited")
        flags_text = f" ({', '.join(flags)})" if flags else ""
        short = item.get("short_description") or ""
        base = f"Detail screen: {kind} '{item['title']}' by {artist['name']}{flags_text}. {short}"
        if kind == "album":
            tracks = item.get("tracks") or []
            if tracks:
                parts = [f"{i + 1}. {t['title']}" for i, t in enumerate(tracks)]
                base += " Tracklist: " + "; ".join(parts) + "."
        return base
