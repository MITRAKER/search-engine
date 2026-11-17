"""A stand-in for the Elasticsearch client, for unit tests without a server.

It records every call and answers with canned responses, so a test can check
both what the engine sends and how it reads the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import Elasticsearch, NotFoundError


@dataclass
class Response:
    """What the real client returns: the parsed JSON is in ``body``."""

    body: Any


def not_found(body: dict[str, Any]) -> NotFoundError:
    """The error the real client raises for an HTTP 404."""
    meta = ApiResponseMeta(
        status=404,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=NodeConfig("http", "localhost", 9200),
    )
    return NotFoundError("not found", meta, body)


def search_response(
    hits: list[dict[str, Any]] | None = None,
    facets: dict[str, int] | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    """A search response body in Elasticsearch's format."""
    hits = hits or []
    buckets = [{"key": key, "doc_count": count} for key, count in (facets or {}).items()]
    return {
        "took": 3,
        "hits": {"total": {"value": len(hits) if total is None else total}, "hits": hits},
        "aggregations": {"facets": {"buckets": buckets}},
    }


class FakeIndices:
    def __init__(self, es: FakeElasticsearch) -> None:
        self.es = es

    def exists(self, **kwargs: Any) -> bool:
        self.es.record("indices.exists", kwargs)
        return self.es.existing

    def create(self, **kwargs: Any) -> Response:
        self.es.record("indices.create", kwargs)
        self.es.existing = True
        return Response({"acknowledged": True})

    def delete(self, **kwargs: Any) -> Response:
        self.es.record("indices.delete", kwargs)
        self.es.existing = False
        return Response({"acknowledged": True})

    def get_settings(self, **kwargs: Any) -> Response:
        self.es.record("indices.get_settings", kwargs)
        return Response({kwargs["index"]: {"settings": self.es.settings}})

    def refresh(self, **kwargs: Any) -> Response:
        self.es.record("indices.refresh", kwargs)
        return Response({})

    def analyze(self, **kwargs: Any) -> Response:
        self.es.record("indices.analyze", kwargs)
        return Response({"tokens": self.es.tokens})


class FakeElasticsearch:
    """Answers like Elasticsearch would, from the attributes a test sets."""

    def __init__(self, *, existing: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.indices = FakeIndices(self)
        self.existing = existing
        self.settings: dict[str, str] = {
            "index.similarity.default.k1": "1.2",
            "index.similarity.default.b": "0.75",
        }
        self.bulk_errors: dict[str, str] = {}  # document id -> error reason
        self.search_responses: list[dict[str, Any]] = []
        self.explain_result: dict[str, Any] | Exception = {"matched": True, "explanation": {}}
        self.tokens: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None  # raised by the next call

    def record(self, name: str, kwargs: dict[str, Any]) -> None:
        if self.fail_with is not None:
            error, self.fail_with = self.fail_with, None
            raise error
        self.calls.append((name, kwargs))

    def called(self, name: str) -> list[dict[str, Any]]:
        """The arguments of every call to ``name``, in order."""
        return [kwargs for call, kwargs in self.calls if call == name]

    def options(self, **kwargs: Any) -> FakeElasticsearch:
        self.record("options", kwargs)
        return self

    def bulk(self, **kwargs: Any) -> Response:
        self.record("bulk", kwargs)
        actions = kwargs["operations"][::2]
        items = []
        for action in actions:
            doc_id = action["create"]["_id"]
            item: dict[str, Any] = {"_id": doc_id, "status": 201}
            if doc_id in self.bulk_errors:
                item = {"_id": doc_id, "status": 409, "error": {"reason": self.bulk_errors[doc_id]}}
            items.append({"create": item})
        return Response({"errors": bool(self.bulk_errors), "items": items})

    def count(self, **kwargs: Any) -> Response:
        self.record("count", kwargs)
        return Response({"count": 25})

    def search(self, **kwargs: Any) -> Response:
        self.record("search", kwargs)
        return Response(self.search_responses.pop(0))

    def explain(self, **kwargs: Any) -> Response:
        self.record("explain", kwargs)
        if isinstance(self.explain_result, Exception):
            raise self.explain_result
        return Response(self.explain_result)


def fake_client(**kwargs: Any) -> tuple[Elasticsearch, FakeElasticsearch]:
    """The fake, typed as a real client for the engine, and as itself for the test."""
    fake = FakeElasticsearch(**kwargs)
    return cast(Elasticsearch, fake), fake
