#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice + UI separation-of-concerns example: a voice-driven music player.

The main ``PipelineWorker`` runs the conversation (STT → LLM → TTS) and
owns the transport + RTVI. Its one tool, ``handle_request``, forwards
each request to the ``MusicUIWorker``, which owns the navigation stack
and screen state and drives the client with UI commands. A non-LLM
``CatalogWorker`` serves the live Deezer catalog over the bus.

Architecture::

    main PipelineWorker (transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── handle_request(query) tool
              └── worker.job("ui", name="respond", payload={query})

    MusicUIWorker (UIWorker): tools + @ui_event click handlers
    CatalogWorker (BaseWorker): @job("catalog") Deezer-backed store

Run the server from this directory:

    uv run bot.py

Then open http://localhost:5173 (the Vite client in ``../client/``) to
talk to the bot.

Requirements:
- OPENAI_API_KEY
- SONIOX_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.job_context import JobError
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from catalog_agent import CatalogWorker
from discovery_agent import (
    DeezerRelatedWorker,
    LLMSuggestionsWorker,
    TopTracksCrossrefWorker,
)
from llm import create_llm_service
from ui_agent import MusicUIWorker

load_dotenv(override=True)

MAIN_NAME = "main"
UI_NAME = "ui"


VOICE_PROMPT = """\
You are the voice layer for a music player backed by a live music \
catalog. A separate UI layer owns all screen state. You do not know \
what is on screen. You do not navigate, play, favorite, or change \
state on your own. Every request that involves the UI goes through \
the ``handle_request`` tool.

## Absolute routing rule
You MUST call ``handle_request`` for every user utterance that implies \
a UI action, including:

- Any navigation: "show me Nirvana", "show me Daft Punk", "go back", \
"go home", "take me back", "the first one", "top right".
- Any action on an item: play, pause, stop, add to favorites, more \
info, tell me about.
- Any discovery: "who's similar", "show me artists like them", \
"what's trending", "what's popular in rock".
- Any question about what's on screen or where the user is.

Never answer these with your own words, not even short confirmations \
like "Back to home." or "Here's Radiohead." ``handle_request`` delivers \
the spoken reply itself, so you never voice the result — call the tool \
and stay silent.

Call the tool every time, even when the user repeats themselves. "Go \
back" five times in a row is five ``handle_request`` calls. Do not \
predict the result and skip the tool. Do not reuse a previous result \
to answer a new turn.

## When not to call the tool
Only respond directly for:

- Small talk that doesn't touch the UI ("hello", "thanks", "you too").
- Clarifying questions when the request is genuinely ambiguous ("what \
should I listen to", "play something fun"). Ask one short question, \
then call ``handle_request`` once the user commits.

## Voice rules
- Plain spoken language only. No markdown, no lists, no symbols.
- Very short. One short sentence by default. Under fifteen words.
- Do not confirm after ``handle_request``; the UI layer speaks the \
reply itself. Stay silent and let it play.
- Do not ask "anything else?" or similar follow-ups.

## handle_request arguments
Pass the user's request as a self-contained query. Leave anything that \
could refer to what's currently on screen verbatim — "this", "that", \
"the first one", "top right", and personal or possessive pronouns like \
"his", "her", "their", "him", "this album", "this artist". The user is \
pointing at what they're looking at, and the UI layer sees the current \
screen and resolves these against it; you cannot see the screen and \
must not guess. The user may have clicked to a different artist since \
you last heard about one, so do NOT rewrite "his/her/their" using the \
previous conversation topic. Only rewrite genuinely cross-turn \
references that name a different entity ("yes" answering a question \
you just asked, "the album we discussed earlier")."""


async def handle_request(params: FunctionCallParams, query: str):
    """Delegate the user's request to the UI layer.

    Args:
        query: The user's request, passed verbatim. Resolve \
            conversation pronouns but leave UI-state references \
            ("top right", "this", "the first one") untouched.
    """
    logger.info(f"handle_request('{query}')")
    try:
        async with params.pipeline_worker.job(
            UI_NAME, name="respond", payload={"query": query}, timeout=30
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"ui job failed: {e}")
        await params.result_callback("Something went wrong on my side.")
        return

    # The UI worker either spoke verbatim (``tts_speak`` → ``t.response``
    # is None) or returned text for the voice LLM to phrase
    # (``{"answer": ...}``). Either way, hand the result straight back.
    await params.result_callback(t.response)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting music player bot")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    stt = SonioxSTTService(
        api_key=os.getenv("SONIOX_API_KEY"),
        settings=SonioxSTTService.Settings(
            language_hints=[Language.EN],
            language_hints_strict=True,
        ),
    )
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSSettings(
            voice=os.getenv("CARTESIA_VOICE_ID"),
        ),
        text_aggregation_mode=TextAggregationMode.TOKEN,
    )
    llm = create_llm_service(system_prompt=VOICE_PROMPT)
    llm.register_direct_function(handle_request, cancel_on_interruption=False, timeout_secs=30)

    context = LLMContext(tools=ToolsSchema(standard_tools=[handle_request]))
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user. Welcome them to the voice music "
                    "player and mention they can ask to see any artist, "
                    "play a track, or get more info. One short sentence."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    # CatalogWorker is added alongside the others so its Deezer warm-up
    # starts as soon as the runner brings the workers up. The three
    # discovery_* workers are peers fanned out from
    # ``MusicUIWorker.discover_similar_artists`` via ``start_ui_job_group``;
    # idle when not invoked.
    await runner.add_workers(
        CatalogWorker("catalog"),
        MusicUIWorker(),
        DeezerRelatedWorker("discovery_deezer"),
        TopTracksCrossrefWorker("discovery_tracks"),
        LLMSuggestionsWorker("discovery_llm"),
        worker,
    )

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Pipecat Cloud Client entry point."""

    if os.environ.get("ENV") != "local":
        from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter

        krisp_filter = KrispVivaFilter()
    else:
        krisp_filter = None

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_in_filter=krisp_filter,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_in_filter=krisp_filter,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
