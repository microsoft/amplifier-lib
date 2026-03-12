"""Tests for amplifier_lib.updates — BundleStatus, check_bundle_status, update_bundle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_lib.bundle import Bundle
from amplifier_lib.sources.protocol import SourceStatus
from amplifier_lib.updates import BundleStatus, check_bundle_status, update_bundle

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _make_status(uri: str, has_update: bool | None, summary: str = "") -> SourceStatus:
    return SourceStatus(
        source_uri=uri,
        is_cached=True,
        has_update=has_update,
        summary=summary or f"status for {uri}",
    )


def _make_bundle(
    name: str = "test-bundle",
    providers: list | None = None,
    tools: list | None = None,
    hooks: list | None = None,
    session: dict | None = None,
    source_uri: str | None = None,
    base_path: Path | None = None,
) -> Bundle:
    bundle = Bundle(
        name=name,
        providers=providers or [],
        tools=tools or [],
        hooks=hooks or [],
        session=session or {},
        base_path=base_path,
    )
    if source_uri:
        bundle._source_uri = source_uri  # type: ignore[attr-defined]
    return bundle


# ===========================================================================
# BundleStatus properties
# ===========================================================================


class TestBundleStatusProperties:
    def _status_with(self, *updates: bool | None) -> BundleStatus:
        sources = [
            _make_status(f"git+https://example.com/repo{i}", u) for i, u in enumerate(updates)
        ]
        return BundleStatus(bundle_name="my-bundle", bundle_source=None, sources=sources)

    # --- has_updates ---

    def test_has_updates_true_when_any_update(self):
        bs = self._status_with(True, False)
        assert bs.has_updates is True

    def test_has_updates_false_when_all_up_to_date(self):
        bs = self._status_with(False, False)
        assert bs.has_updates is False

    def test_has_updates_false_when_all_unknown(self):
        bs = self._status_with(None, None)
        assert bs.has_updates is False

    def test_has_updates_true_when_mixed_with_update(self):
        bs = self._status_with(None, True, False)
        assert bs.has_updates is True

    # --- updateable_sources ---

    def test_updateable_sources_filters_true(self):
        bs = self._status_with(True, False, None, True)
        assert len(bs.updateable_sources) == 2
        assert all(s.has_update is True for s in bs.updateable_sources)

    def test_updateable_sources_empty_when_none_available(self):
        bs = self._status_with(False, None)
        assert bs.updateable_sources == []

    # --- up_to_date_sources ---

    def test_up_to_date_sources_filters_false(self):
        bs = self._status_with(True, False, None)
        assert len(bs.up_to_date_sources) == 1
        assert bs.up_to_date_sources[0].has_update is False

    def test_up_to_date_sources_empty_when_none(self):
        bs = self._status_with(True, None)
        assert bs.up_to_date_sources == []

    # --- unknown_sources ---

    def test_unknown_sources_filters_none(self):
        bs = self._status_with(True, False, None, None)
        assert len(bs.unknown_sources) == 2
        assert all(s.has_update is None for s in bs.unknown_sources)

    def test_unknown_sources_empty_when_all_known(self):
        bs = self._status_with(True, False)
        assert bs.unknown_sources == []

    # --- summary ---

    def test_summary_updates_available(self):
        bs = self._status_with(True, False)
        assert "update" in bs.summary.lower()
        assert "my-bundle" in bs.summary

    def test_summary_all_up_to_date(self):
        bs = self._status_with(False, False)
        assert "up to date" in bs.summary.lower()
        assert "my-bundle" in bs.summary

    def test_summary_unknown_sources(self):
        bs = self._status_with(None, None)
        assert "unknown" in bs.summary.lower()
        assert "my-bundle" in bs.summary

    def test_summary_no_sources(self):
        bs = BundleStatus(bundle_name="empty", bundle_source=None, sources=[])
        assert "empty" in bs.summary
        assert "no tracked" in bs.summary.lower()

    def test_summary_updates_includes_counts(self):
        bs = self._status_with(True, True, False)
        summary = bs.summary
        # Should mention 2 updates and 1 up to date
        assert "2" in summary
        assert "1" in summary

    def test_summary_mixed_with_updates_and_unknown(self):
        bs = self._status_with(True, None)
        summary = bs.summary.lower()
        assert "update" in summary
        assert "unknown" in summary


# ===========================================================================
# check_bundle_status
# ===========================================================================


class TestCheckBundleStatus:
    """Tests for check_bundle_status using a mocked GitSourceHandler."""

    @pytest.fixture
    def git_status_update(self):
        return SourceStatus(
            source_uri="git+https://github.com/org/repo",
            is_cached=True,
            has_update=True,
            summary="Update available (abc12345 → def67890)",
        )

    @pytest.fixture
    def git_status_up_to_date(self):
        return SourceStatus(
            source_uri="git+https://github.com/org/other@main",
            is_cached=True,
            has_update=False,
            summary="Up to date (abc12345)",
        )

    @pytest.mark.asyncio
    async def test_returns_bundle_status_type(self):
        bundle = _make_bundle()
        result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))
        assert isinstance(result, BundleStatus)

    @pytest.mark.asyncio
    async def test_bundle_name_propagated(self):
        bundle = _make_bundle(name="my-bundle")
        result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))
        assert result.bundle_name == "my-bundle"

    @pytest.mark.asyncio
    async def test_bundle_source_uri_propagated(self):
        bundle = _make_bundle(source_uri="git+https://github.com/org/bundle")
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=_make_status("git+https://github.com/org/bundle", False),
        ):
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))
        assert result.bundle_source == "git+https://github.com/org/bundle"

    @pytest.mark.asyncio
    async def test_no_sources_returns_empty(self):
        bundle = _make_bundle()
        result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_git_source_in_providers_checked(self, git_status_update):
        bundle = _make_bundle(
            providers=[{"module": "my-provider", "source": "git+https://github.com/org/repo"}]
        )
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=git_status_update,
        ) as mock_get_status:
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert mock_get_status.call_count == 1
        assert len(result.sources) == 1
        assert result.sources[0].has_update is True

    @pytest.mark.asyncio
    async def test_git_source_in_tools_checked(self, git_status_up_to_date):
        bundle = _make_bundle(
            tools=[{"module": "my-tool", "source": "git+https://github.com/org/other@main"}]
        )
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=git_status_up_to_date,
        ):
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1
        assert result.sources[0].has_update is False

    @pytest.mark.asyncio
    async def test_git_source_in_hooks_checked(self, git_status_up_to_date):
        bundle = _make_bundle(
            hooks=[{"module": "my-hook", "source": "git+https://github.com/org/other@main"}]
        )
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=git_status_up_to_date,
        ):
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_git_source_in_session_orchestrator_checked(self, git_status_up_to_date):
        bundle = _make_bundle(
            session={"orchestrator": {"source": "git+https://github.com/org/other@main"}}
        )
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=git_status_up_to_date,
        ):
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_git_source_in_session_context_checked(self, git_status_up_to_date):
        bundle = _make_bundle(
            session={"context": {"source": "git+https://github.com/org/other@main"}}
        )
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=git_status_up_to_date,
        ):
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_non_git_source_reported_as_unknown(self):
        bundle = _make_bundle(
            providers=[{"module": "my-provider", "source": "/local/path/to/module"}]
        )
        result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1
        assert result.sources[0].has_update is None

    @pytest.mark.asyncio
    async def test_non_git_http_source_reported_as_unknown(self):
        bundle = _make_bundle(
            tools=[{"module": "my-tool", "source": "https://example.com/module.zip"}]
        )
        result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1
        assert result.sources[0].has_update is None

    @pytest.mark.asyncio
    async def test_duplicate_sources_deduplicated(self, git_status_up_to_date):
        shared_uri = "git+https://github.com/org/other@main"
        bundle = _make_bundle(
            providers=[{"module": "a", "source": shared_uri}],
            tools=[{"module": "b", "source": shared_uri}],
        )
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=git_status_up_to_date,
        ) as mock_get_status:
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        # Should only check once despite appearing in two sections
        assert mock_get_status.call_count == 1
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_get_status_exception_returns_unknown_status(self):
        bundle = _make_bundle(
            providers=[{"module": "my-provider", "source": "git+https://github.com/org/repo"}]
        )
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network error"),
        ):
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1
        status = result.sources[0]
        assert status.has_update is None
        assert "network error" in (status.error or "")

    @pytest.mark.asyncio
    async def test_bundle_source_uri_included_in_sources(self):
        bundle = _make_bundle(source_uri="git+https://github.com/org/bundle-itself")
        expected_status = _make_status("git+https://github.com/org/bundle-itself", False)
        with patch(
            "amplifier_lib.updates.GitSourceHandler.get_status",
            new_callable=AsyncMock,
            return_value=expected_status,
        ):
            result = await check_bundle_status(bundle, cache_dir=Path("/tmp/cache"))

        assert len(result.sources) == 1
        assert result.sources[0].source_uri == "git+https://github.com/org/bundle-itself"


# ===========================================================================
# update_bundle
# ===========================================================================


class TestUpdateBundle:
    """Tests for update_bundle using mocked GitSourceHandler and ModuleActivator."""

    _PROVIDER_URI = "git+https://github.com/org/provider@main"
    _TOOL_URI = "git+https://github.com/org/tool@main"

    def _make_update_status(self, uri: str) -> SourceStatus:
        return SourceStatus(
            source_uri=uri,
            is_cached=True,
            has_update=True,
            summary="Update available",
        )

    def _make_up_to_date_status(self, uri: str) -> SourceStatus:
        return SourceStatus(
            source_uri=uri,
            is_cached=True,
            has_update=False,
            summary="Up to date",
        )

    @pytest.mark.asyncio
    async def test_returns_bundle(self):
        bundle = _make_bundle()
        result = await update_bundle(bundle, cache_dir=Path("/tmp/cache"), install_deps=False)
        assert result is bundle

    @pytest.mark.asyncio
    async def test_updates_sources_with_available_updates(self):
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
            tools=[{"module": "tool", "source": self._TOOL_URI}],
        )
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[
                self._make_update_status(self._PROVIDER_URI),
                self._make_update_status(self._TOOL_URI),
            ],
        )

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.updates.GitSourceHandler.update",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_update,
        ):
            await update_bundle(bundle, cache_dir=Path("/tmp/cache"), install_deps=False)

        assert mock_update.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_up_to_date_sources(self):
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
        )
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[self._make_up_to_date_status(self._PROVIDER_URI)],
        )

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.updates.GitSourceHandler.update",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await update_bundle(bundle, cache_dir=Path("/tmp/cache"), install_deps=False)

        mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_selective_update_only_updates_specified_uris(self):
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
            tools=[{"module": "tool", "source": self._TOOL_URI}],
        )
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[
                self._make_update_status(self._PROVIDER_URI),
                self._make_update_status(self._TOOL_URI),
            ],
        )

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.updates.GitSourceHandler.update",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_update,
        ):
            await update_bundle(
                bundle,
                cache_dir=Path("/tmp/cache"),
                selective=[self._PROVIDER_URI],  # Only update provider
                install_deps=False,
            )

        # Only one update should have been called (provider only)
        assert mock_update.call_count == 1

    @pytest.mark.asyncio
    async def test_selective_empty_list_updates_nothing(self):
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
        )
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[self._make_update_status(self._PROVIDER_URI)],
        )

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.updates.GitSourceHandler.update",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            await update_bundle(
                bundle,
                cache_dir=Path("/tmp/cache"),
                selective=[],  # Empty — nothing selected
                install_deps=False,
            )

        mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_install_deps_false_skips_dependency_installation(self):
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
            base_path=Path("/tmp/bundle"),
        )
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[self._make_update_status(self._PROVIDER_URI)],
        )

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.updates.GitSourceHandler.update",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "amplifier_lib.modules.activator.ModuleActivator._install_dependencies",
                new_callable=AsyncMock,
            ) as mock_install,
        ):
            await update_bundle(bundle, cache_dir=Path("/tmp/cache"), install_deps=False)

        mock_install.assert_not_called()

    @pytest.mark.asyncio
    async def test_install_deps_true_calls_install_dependencies(self):
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
        )
        bundle.base_path = Path("/tmp/bundle")
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[self._make_update_status(self._PROVIDER_URI)],
        )

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.updates.GitSourceHandler.update",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "amplifier_lib.modules.activator.ModuleActivator._install_dependencies",
                new_callable=AsyncMock,
            ) as mock_install,
        ):
            await update_bundle(bundle, cache_dir=Path("/tmp/cache"), install_deps=True)

        mock_install.assert_called_once_with(Path("/tmp/bundle"))

    @pytest.mark.asyncio
    async def test_install_deps_skipped_when_nothing_updated(self):
        """install_deps should not be called if there were no updates to apply."""
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
        )
        bundle.base_path = Path("/tmp/bundle")
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[self._make_up_to_date_status(self._PROVIDER_URI)],
        )

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.modules.activator.ModuleActivator._install_dependencies",
                new_callable=AsyncMock,
            ) as mock_install,
        ):
            await update_bundle(bundle, cache_dir=Path("/tmp/cache"), install_deps=True)

        mock_install.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_failure_logged_but_continues(self):
        """A failing update should not abort the entire update process."""
        bundle = _make_bundle(
            providers=[{"module": "prov", "source": self._PROVIDER_URI}],
            tools=[{"module": "tool", "source": self._TOOL_URI}],
        )
        bundle_status = BundleStatus(
            bundle_name="test-bundle",
            bundle_source=None,
            sources=[
                self._make_update_status(self._PROVIDER_URI),
                self._make_update_status(self._TOOL_URI),
            ],
        )

        call_count = 0

        async def _update_side_effect(parsed, cache_dir):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Clone failed")
            return MagicMock()

        with (
            patch(
                "amplifier_lib.updates.check_bundle_status",
                new_callable=AsyncMock,
                return_value=bundle_status,
            ),
            patch(
                "amplifier_lib.updates.GitSourceHandler.update",
                side_effect=_update_side_effect,
            ),
        ):
            # Should not raise even though first update fails
            result = await update_bundle(bundle, cache_dir=Path("/tmp/cache"), install_deps=False)

        assert result is bundle
        assert call_count == 2  # Both were attempted
