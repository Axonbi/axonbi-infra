# LangGraph on the `studio` server

Flow definitions for the LangGraph Studio instance at
`https://langchain.axonbi.com:2024`.

Push to `main` and the server picks the change up within ~60s: a systemd timer
polls this repo, syncs `langgraph/` into `/opt/langgraph`, and restarts the
service. Reload Studio to see the new graph.

## Layout

    graphs/hello_flow.py    no LLM -- branching + a retry loop, runs with no API key
    graphs/agent.py         tool-calling agent on Claude (needs ANTHROPIC_API_KEY)
    langgraph.json          graph registry + CORS config for Studio
    render_graph.py         print a graph as ascii/mermaid without a browser
    proxy/                  nginx TLS + basic auth, systemd units, deploy script

## Adding a graph

1. Write `graphs/my_flow.py` exposing a compiled graph named `graph`.
2. Register it in `langgraph.json` under `"graphs"`.
3. Commit and push.

## What is NOT in this repo

This repo is public, so these live only on the server:

    /opt/langgraph/.env          ANTHROPIC_API_KEY
    /etc/langgraph/proxy.env     basic-auth password + secret path
    /opt/langgraph/.venv         the virtualenv (absolute paths, not portable)
    /opt/langgraph/.langgraph_api thread + run state

## Server notes

- `langgraph dev` binds `127.0.0.1:2025` and runs with auth `noop`. It must
  never be exposed directly -- nginx on 2024 is the only way in.
- `deploy.sh` deliberately leaves nginx and the certificate alone, so a bad
  commit cannot take down TLS. After editing anything in `proxy/`, run
  `setup-proxy.sh` by hand.
- Python is a uv-managed 3.13; the distro 3.14 is too new for the dep tree.
