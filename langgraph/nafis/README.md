# LangGraph platform — Nafis server (8.213.40.74)

Infrastructure for an agent project that is **not written yet**. Everything
here waits: nothing starts, and the deploy does nothing, until a project is
configured and pushed.

## What the project must provide

The AI engineer's repo needs these at its **root** (not in a subfolder — the
deploy syncs the repo root into `/opt/langgraph`):

    langgraph.json     graph registry. Required -- the deploy refuses to run
                       without it, and the Studio service will not start.
    requirements.txt   installed into the venv on every deploy.
    app.py             optional. A FastAPI app exposing POST /chat, if n8n is
                       to call it. Without it, only the Studio service runs.

Two things learned the hard way on the other server, worth building in from
the start:

1. **Do not compile the graph with a checkpointer.** `langgraph dev` refuses
   to load a graph carrying its own and exits with `GraphLoadError`. Detect the
   server instead:

       _UNDER_API = "langgraph_api" in sys.modules
       graph = builder.compile() if _UNDER_API else builder.compile(checkpointer=MemorySaver())

2. **Add a CORS block to `langgraph.json`,** or Studio fails with a bare
   "Failed to fetch". Allow the headers wildcard — Studio sends headers beyond
   the obvious ones and any unlisted one rejects the whole preflight:

       "http": { "cors": {
           "allow_origins": ["https://smith.langchain.com"],
           "allow_credentials": true,
           "allow_methods": ["*"],
           "allow_headers": ["*"]
       }}

## Layout

    /etc/langgraph/deploy.env   REPO + BRANCH to track, bind address  (not in git)
    /opt/langgraph/.env         the project's runtime secrets          (not in git)
    /opt/agent-src              clone of the project repo
    /opt/langgraph              runtime: synced from the clone, holds .venv
    /opt/langgraph-platform     these scripts

    langgraph-dev     127.0.0.1:2025   Studio backend (auth "noop" - never expose)
    langgraph-api     172.18.0.1:8000  app.py, the endpoint n8n calls
    langgraph-deploy  timer, every 60s

## Why 172.18.0.1

That is the `n8n_web` docker bridge gateway: the n8n **container** reaches the
host on it, so n8n calls `http://172.18.0.1:8000/chat` with no hop off the box.
Deliberately not `0.0.0.0` — ufw is inactive here, so that would publish the
agent to the internet.

If the n8n compose stack is ever recreated with a different subnet, update
`BIND_ADDR` in `/etc/langgraph/deploy.env` and restart `langgraph-api`.

## This box also runs production n8n

`automation.tanasuq.med.sa`, plus its Postgres and Caddy. The deploy never
touches Caddy or certificates, and the agent services are separate units, so a
bad agent commit cannot affect n8n. Reload Caddy only after
`caddy validate` passes.
