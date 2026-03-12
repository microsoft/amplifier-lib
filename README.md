# Amplifier Lib

Core library for building Amplifier applications: bundle composition, source resolution, session manipulation, and module activation.

## Install

```bash
pip install git+https://github.com/microsoft/amplifier-lib
```

Requires Python 3.12+. Depends on `amplifier-core>=1.1.1` and `pyyaml>=6.0`.

## Quick Start

```python
import asyncio
from amplifier_lib import load_bundle

async def main():
    # Load a bundle from a git URL or local path
    bundle = await load_bundle("git+https://github.com/microsoft/amplifier-foundation@main")
    provider = await load_bundle("./providers/anthropic.yaml")

    # Compose bundles (later overrides earlier)
    composed = bundle.compose(provider)

    # Prepare: resolves sources, activates modules, returns a PreparedBundle
    prepared = await composed.prepare()

    # Create a session and execute
    session = await prepared.create_session()
    response = await session.execute("Hello!")
    print(response)

asyncio.run(main())
```

## What's Included

### Bundle System

| Export | Purpose |
|--------|---------|
| `Bundle` | Core composable unit — load, compose, prepare |
| `PreparedBundle` | Fully activated bundle ready for session creation |
| `BundleModuleResolver` | Maps module IDs to activated local paths |
| `BundleRegistry` | Register, load, cache, and track bundles |
| `load_bundle(uri)` | Load a bundle from any supported URI |

### Session Manipulation

Pure functions operating on message lists — no framework coupling:

| Export | Purpose |
|--------|---------|
| `fork_session_in_memory()` | Fork a conversation at a specific turn |
| `fork_session()` | Fork with filesystem persistence |
| `get_turn_boundaries()` | Find where each turn starts in a message list |
| `count_turns()` | Count user turns |
| `find_orphaned_tool_calls()` | Detect tool calls without results |
| `add_synthetic_tool_results()` | Patch orphaned tool calls for valid message lists |
| `get_fork_preview()` | Preview a fork without writing |
| `get_session_lineage()` | Walk parent chain |
| `list_session_forks()` | Find all forks of a session |
| `set_working_dir()` | Update session working directory capability |

### Bundle Updates

| Export | Purpose |
|--------|---------|
| `check_bundle_status()` | Check for upstream changes via `git ls-remote` |
| `update_bundle()` | Force re-download all sources |
| `BundleStatus` | Status result with per-source details |

### Provider Preferences

| Export | Purpose |
|--------|---------|
| `apply_provider_preferences_with_resolution()` | Match provider/model preferences with glob support |

### Utilities

| Export | Purpose |
|--------|---------|
| `sanitize_message()` | Make LLM messages safe for JSON serialization |
| `write_with_backup()` | Atomic file writes with `.bak` backup |
| `get_amplifier_home()` | Resolve `$AMPLIFIER_HOME` or `~/.amplifier` |
| `generate_sub_session_id()` | W3C Trace Context session IDs for spawn trees |

### Exceptions

```
BundleError
├── BundleNotFoundError      — cannot locate bundle at source
├── BundleLoadError          — found but failed to parse
├── BundleValidationError    — parsed but invalid
└── BundleDependencyError    — circular or missing dependency
```

## Sub-packages

These are not re-exported from the top level — import them directly.

### Sources (`amplifier_lib.sources`)

URI resolution with pluggable handlers:

```python
from amplifier_lib.sources import SimpleSourceResolver, ParsedURI, parse_uri

parsed = parse_uri("git+https://github.com/org/repo@main#subdirectory=pkg")
# ParsedURI(scheme='git+https', host='github.com', ref='main', subpath='pkg')
```

| Handler | Schemes |
|---------|---------|
| `GitSourceHandler` | `git+https://` — shallow clone, `@ref`, `#subdirectory=`, `AMPLIFIER_GIT_HOST` rewriting |
| `FileSourceHandler` | `file://`, local paths |
| `HttpSourceHandler` | `https://`, `http://` |
| `ZipSourceHandler` | `zip+https://`, `zip+file://` |

### Mentions (`amplifier_lib.mentions`)

Parse and resolve `@namespace:path` references in instructions:

```python
from amplifier_lib.mentions import parse_mentions, load_mentions

mentions = parse_mentions("Load @mybundle:context/guide.md for reference")
results = await load_mentions(text, resolver)
```

### Modules (`amplifier_lib.modules`)

Module activation — download, install dependencies, wire into `sys.path`:

```python
from amplifier_lib.modules import ModuleActivator, InstallStateManager
```

`InstallStateManager` fingerprints dependency files to skip redundant installs.

### Session (`amplifier_lib.session`)

The session sub-package exports are available at the top level (see Session Manipulation above). Direct import is also supported:

```python
from amplifier_lib.session import fork_session_in_memory, get_turn_boundaries
```

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
