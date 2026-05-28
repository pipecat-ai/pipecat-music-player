# Pipecat Music Player

A voice-driven music browsing app built on [Pipecat](https://github.com/pipecat-ai/pipecat). Demonstrates the **voice / UI separation-of-concerns pattern** with Pipecat workers: a main pipeline worker handles the spoken conversation, a `UIWorker` owns the UI state, and a non-LLM worker owns the music catalog. Talk to browse trending artists, play 30-second previews, favorite songs, ask conversational questions about a catalog, and more.

**Try it live:** [pipecat-music-player.vercel.app](https://pipecat-music-player.vercel.app/)

Catalog data is live from [Deezer](https://developers.deezer.com/api). Descriptions and answers are generated on demand via OpenAI. No database. No auth. No seeded catalog.

## Architecture

```
main PipelineWorker (transport + RTVI):
  transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
    └── handle_request(query) tool
          └── worker.job("ui", name="respond", payload={query})

MusicUIWorker (UIWorker):
  ├── tools: navigate_to_artist, select_item, switch_tab, play,
  │          control_playback, show_info, add_to_favorites,
  │          show_trending, go_back, go_home, answer_about_catalog,
  │          answer_about_music, answer_about_screen
  ├── send_command(...) → RTVIEvent.UICommand screen/toast/playback updates
  └── @ui_event(hello | nav | action | set_tab | play_track): client clicks (no LLM)

CatalogWorker (runner peer, long-lived, no LLM pipeline)
  ├── Deezer-backed artist + album + track cache
  ├── LLM-generated descriptions + conversational answers
  └── @job handlers: list_home, list_new_releases, find_artist, get_artist,
      fetch_artist_by_id, related_artists, resolve_item, get_trending,
      get_album_preview, get_album_tracks, get_description
```

- **Main worker** (`server/bot.py`): a `PipelineWorker` that owns the Pipecat transport + RTVI and runs the conversation (STT → LLM → TTS). Its one tool, `handle_request`, dispatches a `respond` job to the UI worker and speaks the reply.
- **MusicUIWorker** (`server/ui_agent.py`): a `UIWorker` that owns a navigation stack (home → artist → detail → trending) and drives the client with `send_command` UI commands. Voice requests run through its own LLM (the `respond` job); client clicks arrive as `@ui_event` events and update state directly without an LLM call for low latency.
- **CatalogWorker** (`server/catalog_agent.py`): a process-lifetime `BaseWorker` (no LLM pipeline) that owns the Deezer-backed catalog, description cache, and Q&A inference. Every music lookup goes through one of its `@job(name=...)` handlers.

## What this demonstrates

A tour of the Pipecat worker primitives this app exercises, with pointers into the code:

- **`PipelineWorker` with voice LLM + RTVI** — the conversation pipeline (STT → LLM → TTS), transport, and RTVI processor all live in a single worker. `server/bot.py:run_bot`.
- **`UIWorker` subclass with custom `<ui_state>`** — `MusicUIWorker` overrides `render_ui_state` to surface server-owned screen state (`self._screen_state`) instead of an accessibility snapshot; `keep_history=False` keeps each turn stateless except for the current screen, which the auto-inject hook re-injects before every inference. `server/ui_agent.py:MusicUIWorker`.
- **`BaseWorker` with `@job(name=...)` per action** — `CatalogWorker` exposes one handler per action (`list_home`, `find_artist`, `get_album_tracks`, …); each reads its own payload and responds. Added once as a runner peer so its warm-up and caches live for the whole process. `server/catalog_agent.py`.
- **Voice → UI delegation** — the voice LLM's only tool, `handle_request`, opens a job to the UI worker and awaits the reply: `async with params.pipeline_worker.job("ui", name="respond", payload={"query": query}) as t`. `server/bot.py:handle_request`.
- **Worker speaks the reply verbatim** — every UI tool ends with `respond_to_job(answer, tts_speak=True)`, which publishes a `BusTTSSpeakMessage` the main pipeline speaks directly. The voice LLM never re-phrases, so each user turn is exactly two inferences (voice route → UI act). Any tool in `server/ui_agent.py`.
- **Worker → worker dispatch** — the UI worker calls the catalog the same way the voice LLM calls the UI: `self.job("catalog", name="find_artist", payload={...})` — the `name` argument selects which `@job` handler runs. `server/ui_agent.py:_catalog`.
- **Server → client UI commands** — `self.send_command("screen", {...})` publishes a `BusUICommandMessage`; `PipelineWorker` translates it to an `RTVIUICommandFrame` and the client receives `RTVIEvent.UICommand`. `server/ui_agent.py:_emit_*` methods.
- **Client → server clicks** — `client.sendUIEvent("nav" | "action" | …, payload)` becomes a `BusUIEventMessage` dispatched by name to `@ui_event` handlers. The framework runs each handler in its own task, so they can call catalog jobs without deadlocking the bus dispatcher. `server/ui_agent.py` `@ui_event` handlers.
- **Runner lifecycle** — `PipelineRunner` + `await runner.add_workers(CatalogWorker("catalog"), MusicUIWorker(), worker)` brings all three workers up; `await runner.run()` runs them until the transport disconnects. `server/bot.py:run_bot`.

## Features

- **Home**: Trending artists (live Deezer chart), New releases (Deezer editorial feed), Favorites — three 8-column grids.
- **Artist pages**: Albums / Songs / Related tabs. Full discography (not capped), top 16 songs, lazy-loaded related artists.
- **Album detail with tracklist**: 30-second Deezer previews per track. Click a track to play; click again to stop.
- **Global search**: say "play Bohemian Rhapsody" from anywhere, and the server resolves via Deezer's track search before falling back to album search.
- **Conversational Q&A**: "what's their latest album?", "what's their most iconic album?", "are they still active?" route through dedicated `answer_about_catalog` / `answer_about_music` tools.
- **Descriptions**: LLM-generated, grounded by Deezer metadata, cached in-process. Short line under each detail cover; long form appears in toast cards that auto-dismiss when the bot stops speaking.
- **Trending by genre**: "what's trending in alternative?" works for any Deezer genre, derived from the per-genre track chart (since Deezer's artist chart endpoint ignores genre).
- **Favorites** stored in session memory.

## Running

### Prerequisites

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/)
- Node 20+, `npm`
- API keys: OpenAI, Soniox, Cartesia, Daily

### Environment

Create `server/.env` with:

```
OPENAI_API_KEY=...
SONIOX_API_KEY=...
CARTESIA_API_KEY=...
DAILY_API_KEY=...
```

### Start the server

```bash
cd server
uv sync
uv run bot.py
```

Binds to `http://localhost:7860` (SmallWebRTC) by default. Set `VITE_TRANSPORT` to `daily` for a Daily room.

### Start the client

```bash
cd client
npm install
npm run dev
```

Open http://localhost:5173, click **Connect**, and start talking.

## Deploy to Pipecat Cloud

Once the bot works locally, you can deploy the server to [Pipecat Cloud](https://pipecat.daily.co) and point the client at the hosted bot.

### Prerequisites

1. [Sign up for Pipecat Cloud](https://pipecat.daily.co/sign-up).
2. Install the [Pipecat CLI](https://github.com/pipecat-ai/pipecat-cli) and log in:

   ```bash
   uv tool install pipecat-ai-cli
   pc cloud auth login
   ```

### Review the deployment configuration

Deployment settings live in [`server/pcc-deploy.toml`](server/pcc-deploy.toml):

```toml
agent_name    = "music-player"
secret_set    = "music-player-secrets"
agent_profile = "agent-1x"

[krisp_viva]
audio_filter = "tel"

[scaling]
min_agents = 1
```

Adjust `agent_name` / `secret_set` if you want to deploy multiple variants. The [pcc-deploy.toml docs](https://docs.pipecat.ai/api-reference/cli/cloud/deploy#configuration-file-pcc-deploy-toml) cover the full schema.

### Upload secrets

```bash
cd server
pc cloud secrets set music-player-secrets --file .env
```

This pushes every key from `server/.env` (`OPENAI_API_KEY`, `SONIOX_API_KEY`, `CARTESIA_API_KEY`, `DAILY_API_KEY`) into Pipecat Cloud's secret store. The bot reads them at runtime, so nothing is baked into the image.

### Deploy

```bash
pc cloud deploy
```

Pipecat builds and ships the bot from the current directory. The first build takes a few minutes; subsequent deploys reuse the layer cache. See the [cloud builds guide](https://docs.pipecat.ai/pipecat-cloud/guides/cloud-builds) for more.

### Point the client at the deployed bot

The client picks its bot endpoint up from `VITE_BOT_START_URL` (see [`client/src/config.ts`](client/src/config.ts)). For local dev it defaults to `http://localhost:7860/start`; for Pipecat Cloud, set it to your agent's public start URL and pass a [Public API Key](https://docs.pipecat.ai/pipecat-cloud/fundamentals/authentication) as the bearer token. In `client/.env.local`:

```bash
VITE_BOT_START_URL=https://api.pipecat.daily.co/v1/public/music-player/start
VITE_BOT_START_PUBLIC_API_KEY=pk_...
```

Then `npm run build` and deploy the `client/dist/` directory to any static host (Vercel, Netlify, Cloudflare Pages, …). The client is a vanilla Vite SPA, so nothing about the build is Pipecat-Cloud-specific.

## Things to try

- **"Show me Taylor Swift"** — artist page, 8-col Albums grid.
- **"What's their latest album?"** — catalog Q&A, voice answer + toast for the album.
- **"Show me the songs"** — switches the Artist page tab.
- **"Play London Calling"** — resolves even when the catalog only has "London Calling (Remastered)".
- **"Play Bohemian Rhapsody"** from home — global Deezer search loads Queen and starts playback.
- **"What's trending in metal?"** — genre chart derived from the track feed.
- **"Show me similar artists"** — Related tab fetches on demand.
- **"Tell me about Nevermind"** — long-description toast, auto-dismisses when narration ends.
- **"Most iconic album?"** — music-trivia answer drawn from training knowledge, grounded by the artist's catalog.
