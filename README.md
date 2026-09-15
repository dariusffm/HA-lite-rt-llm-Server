# homassist-addons

Home Assistant Add-on Repository by Darius Pauly.

[![Add repository to my Home Assistant](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fdariusffm%2FHA-lite-rt-lm)

## Add-ons

| Add-on | Status | Purpose |
|---|---|---|
| [litert-llm-server](litert-llm-server/) | in development | Local LLM inference via Google LiteRT-LM, exposing OpenAI- and Ollama-compatible HTTP APIs |

## Installation

1. In Home Assistant open **Settings → Apps → App Store → ⋮ → Repositories**
   (older versions: **Settings → Add-ons → Add-on Store**).
2. Add `https://github.com/dariusffm/HA-lite-rt-lm` and close the dialog.
3. Search the store for the add-on, open it and click **Install**.
   The Supervisor builds the image locally; expect 5–15 minutes on first install.

Or click the badge above to have Home Assistant open the dialog for you.

## Supported Architectures

`amd64` and `aarch64`. Images are built on the Home Assistant host; pre-built
images via GHCR are planned.

## Development

See [`CLAUDE.md`](CLAUDE.md) for the architectural rules every add-on in this
repo follows. Per-add-on docs live in each add-on's directory
(`README.md`, `DOCS.md`, `CHANGELOG.md`).

## License

[MIT](LICENSE). Models downloaded by the add-ons are subject to their own
licenses (e.g. the Gemma Terms of Use for Gemma models).
