"""Tests for amplifier_lib.exceptions — hierarchy, instantiation, str representation."""

import pytest

from amplifier_lib.exceptions import (
    BundleDependencyError,
    BundleError,
    BundleLoadError,
    BundleNotFoundError,
    BundleValidationError,
)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_bundle_error_is_exception(self):
        assert issubclass(BundleError, Exception)

    def test_bundle_not_found_is_bundle_error(self):
        assert issubclass(BundleNotFoundError, BundleError)

    def test_bundle_load_error_is_bundle_error(self):
        assert issubclass(BundleLoadError, BundleError)

    def test_bundle_validation_error_is_bundle_error(self):
        assert issubclass(BundleValidationError, BundleError)

    def test_bundle_dependency_error_is_bundle_error(self):
        assert issubclass(BundleDependencyError, BundleError)

    def test_subclasses_are_not_siblings(self):
        """Concrete subclasses should not inherit from each other."""
        assert not issubclass(BundleNotFoundError, BundleLoadError)
        assert not issubclass(BundleLoadError, BundleNotFoundError)
        assert not issubclass(BundleValidationError, BundleNotFoundError)
        assert not issubclass(BundleDependencyError, BundleValidationError)

    def test_all_catchable_as_bundle_error(self):
        """All subclasses should be catchable as BundleError."""
        for exc_class in (
            BundleNotFoundError,
            BundleLoadError,
            BundleValidationError,
            BundleDependencyError,
        ):
            with pytest.raises(BundleError):
                raise exc_class("test")

    def test_all_catchable_as_exception(self):
        """All exceptions should be catchable as base Exception."""
        for exc_class in (
            BundleError,
            BundleNotFoundError,
            BundleLoadError,
            BundleValidationError,
            BundleDependencyError,
        ):
            with pytest.raises(Exception):
                raise exc_class("test")


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    @pytest.mark.parametrize(
        "exc_class",
        [
            BundleError,
            BundleNotFoundError,
            BundleLoadError,
            BundleValidationError,
            BundleDependencyError,
        ],
    )
    def test_instantiate_no_args(self, exc_class):
        exc = exc_class()
        assert isinstance(exc, exc_class)

    @pytest.mark.parametrize(
        "exc_class",
        [
            BundleError,
            BundleNotFoundError,
            BundleLoadError,
            BundleValidationError,
            BundleDependencyError,
        ],
    )
    def test_instantiate_with_message(self, exc_class):
        msg = "something went wrong"
        exc = exc_class(msg)
        assert exc.args[0] == msg

    @pytest.mark.parametrize(
        "exc_class",
        [
            BundleError,
            BundleNotFoundError,
            BundleLoadError,
            BundleValidationError,
            BundleDependencyError,
        ],
    )
    def test_instantiate_with_multiple_args(self, exc_class):
        exc = exc_class("message", "extra", 42)
        assert exc.args == ("message", "extra", 42)


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


class TestStrRepresentation:
    @pytest.mark.parametrize(
        "exc_class",
        [
            BundleError,
            BundleNotFoundError,
            BundleLoadError,
            BundleValidationError,
            BundleDependencyError,
        ],
    )
    def test_str_contains_message(self, exc_class):
        msg = "bundle at /some/path not found"
        exc = exc_class(msg)
        assert msg in str(exc)

    def test_str_empty_when_no_args(self):
        exc = BundleError()
        assert str(exc) == ""

    def test_repr_includes_class_name(self):
        exc = BundleNotFoundError("missing bundle")
        assert "BundleNotFoundError" in repr(exc)

    def test_not_found_specific_message(self):
        exc = BundleNotFoundError("bundle 'my-bundle' not found at source 'github:org/repo'")
        assert "my-bundle" in str(exc)

    def test_load_error_specific_message(self):
        exc = BundleLoadError("YAML parse error at line 42")
        assert "42" in str(exc)

    def test_validation_error_specific_message(self):
        exc = BundleValidationError("missing required field: 'name'")
        assert "name" in str(exc)

    def test_dependency_error_specific_message(self):
        exc = BundleDependencyError("circular dependency detected: a -> b -> a")
        assert "circular" in str(exc)
