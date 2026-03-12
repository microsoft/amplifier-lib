"""Source resolution for bundles (git, file, http, zip)."""

from amplifier_lib.sources.file import FileSourceHandler
from amplifier_lib.sources.git import GitSourceHandler
from amplifier_lib.sources.http import HttpSourceHandler
from amplifier_lib.sources.protocol import (
    SourceHandlerProtocol,
    SourceHandlerWithStatusProtocol,
    SourceResolverProtocol,
    SourceStatus,
)
from amplifier_lib.sources.resolver import SimpleSourceResolver
from amplifier_lib.sources.uri import ParsedURI, ResolvedSource, parse_uri
from amplifier_lib.sources.zip import ZipSourceHandler

__all__ = [
    "SimpleSourceResolver",
    "FileSourceHandler",
    "GitSourceHandler",
    "HttpSourceHandler",
    "ZipSourceHandler",
    "SourceResolverProtocol",
    "SourceHandlerProtocol",
    "SourceHandlerWithStatusProtocol",
    "SourceStatus",
    "ParsedURI",
    "ResolvedSource",
    "parse_uri",
]
