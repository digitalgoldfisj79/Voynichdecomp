# PAL MCP Server

This repo is configured to use the [PAL MCP Server](https://github.com/BeehiveInnovations/pal-mcp-server)
(Provider Abstraction Layer). PAL lets your CLI (Claude Code, Gemini CLI, Codex CLI, …)
orchestrate multiple AI models — routing each task to the model best suited for it —
and provides tools like `chat`, `thinkdeep`, `planner`, `consensus`, `codereview`,
`precommit`, `debug`, `apilookup`, and `challenge`.

The configuration lives in [`.mcp.json`](../.mcp.json) at the repo root and is
picked up automatically by MCP clients that read project-scoped config (e.g. Claude Code).

## One-time setup

### 1. Install `uv` / `uvx`

PAL is launched on demand via `uvx`, so you need the [`uv`](https://docs.astral.sh/uv/)
toolchain (which provides `uvx`) on your PATH:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No manual clone is required — `uvx` fetches and runs PAL directly from the Git repo.

### 2. Add your provider API key(s)

```bash
cp .env.example .env
# then edit .env and fill in at least one provider key
```

`.env` is gitignored, so keys are never committed. `.mcp.json` reads them via
`${VAR:-}` expansion, so unset providers simply resolve to empty and don't
break server startup. You need **at least one** provider key.

| Variable | Provider |
|---|---|
| `GEMINI_API_KEY` | Google Gemini |
| `OPENAI_API_KEY` | OpenAI / GPT models |
| `OPENROUTER_API_KEY` | OpenRouter (multi-model gateway) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI deployments |
| `XAI_API_KEY` | X.AI Grok |
| `DIAL_API_KEY` | DIAL vendor-agnostic routing |

Make sure the variables from `.env` are exported into the environment your MCP
client runs in (for example, `set -a; source .env; set +a` before launching the
client, or load `.env` via your shell profile / process manager).

### 3. Restart your MCP client

Restart Claude Code (or your MCP client) so it re-reads `.mcp.json` and starts
the `pal` server. In Claude Code you can check with `/mcp`.

## Behaviour options

Both are set in `.mcp.json` and overridable from `.env`:

- `PAL_DEFAULT_MODEL` (default `auto`) — model-selection strategy.
- `PAL_DISABLED_TOOLS` (default `analyze,refactor,testgen,secaudit,docgen,tracer`)
  — comma-separated PAL tools to keep disabled. Remove entries to enable more tools.

See the [PAL README](https://github.com/BeehiveInnovations/pal-mcp-server) for the
full list of tools and environment variables.
