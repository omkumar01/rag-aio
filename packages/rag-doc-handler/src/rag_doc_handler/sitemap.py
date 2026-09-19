"""Web site discovery: sitemap fetching and same-domain BFS crawling."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as _ET

import httpx
from bs4 import BeautifulSoup
from rag_core.errors import CrawlError

from rag_doc_handler.security import validate_public_url

__all__ = ["CrawlLimits", "SiteCrawler", "SitemapCrawler"]


@dataclass
class CrawlLimits:
    """Limits for crawler behaviour and resource use."""

    max_pages: int = 500
    max_depth: int = 1
    max_response_bytes: int = 10 * 1024 * 1024
    timeout_s: float = 30.0
    max_concurrency: int = 8
    same_domain_only: bool = True


def _same_domain(url: str, netloc: str) -> bool:
    return urlsplit(url).netloc == netloc


def _parse_locs(xml: str) -> list[str]:
    """Extract ``<loc>`` text from a sitemap (namespace-agnostic)."""
    locs: list[str] = []
    try:
        root = _ET.fromstring(xml.encode("utf-8"))
    except _ET.ParseError:
        return locs
    for el in root.iter():
        if el.tag.endswith("loc") and el.text:
            locs.append(el.text.strip())
    return locs


def _is_sitemap_index(xml: str) -> bool:
    return "<sitemapindex" in xml.lower()


async def _fetch_capped(
    client: httpx.AsyncClient, limits: CrawlLimits, url: str
) -> tuple[bytes, str | None]:
    """Validate (SSRF), stream, and enforce ``max_response_bytes`` for *url*."""
    validated = validate_public_url(url)
    chunks: list[bytes] = []
    total = 0
    async with client.stream(
        "GET", validated, follow_redirects=True, timeout=limits.timeout_s
    ) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type")
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > limits.max_response_bytes:
                raise CrawlError(
                    f"response exceeds max_response_bytes for {url}",
                    details={"url": validated, "bytes": total},
                )
            chunks.append(chunk)
    return b"".join(chunks), content_type


class SitemapCrawler:
    """Fetches an XML sitemap (or sitemap index) for a host."""

    def __init__(self, client: httpx.AsyncClient, limits: CrawlLimits | None = None) -> None:
        self.client = client
        self.limits = limits or CrawlLimits()

    async def discover(self, url: str) -> list[str]:
        sitemap_url = self._sitemap_url(url)
        try:
            body, _ct = await _fetch_capped(self.client, self.limits, sitemap_url)
        except (CrawlError, httpx.HTTPError):
            return []
        text = body.decode("utf-8", "replace")
        locs = _parse_locs(text)
        if not _is_sitemap_index(text):
            return locs[: self.limits.max_pages]
        # Sitemap index: follow each child sitemap once (one level of recursion).
        result: list[str] = []
        for loc in locs:
            if len(result) >= self.limits.max_pages:
                break
            try:
                sub_body, _ = await _fetch_capped(self.client, self.limits, loc)
            except (CrawlError, httpx.HTTPError):
                continue
            result.extend(_parse_locs(sub_body.decode("utf-8", "replace")))
        return result[: self.limits.max_pages]

    def _sitemap_url(self, url: str) -> str:
        lowered = url.lower()
        if "/sitemap" in lowered or lowered.endswith(".xml"):
            return url
        return urljoin(url, "/sitemap.xml")


class SiteCrawler:
    """Same-domain BFS crawler honoring robots.txt, depth and page caps."""

    def __init__(self, client: httpx.AsyncClient, limits: CrawlLimits | None = None) -> None:
        self.client = client
        self.limits = limits or CrawlLimits()

    async def crawl(self, start_url: str) -> list[tuple[str, bytes]]:
        start = validate_public_url(start_url)
        base_netloc = urlsplit(start).netloc
        robots = await self._fetch_robot_parser(urljoin(start, "/robots.txt"))

        results: list[tuple[str, bytes]] = []
        visited: set[str] = {start}
        pending: deque[tuple[str, int]] = deque([(start, 1)])
        sem = asyncio.Semaphore(self.limits.max_concurrency)

        async def process(url: str, depth: int) -> None:
            if len(results) >= self.limits.max_pages:
                return
            try:
                body, ctype = await self._fetch_html(url, sem, robots)
            except (CrawlError, httpx.HTTPError):
                return
            if not ctype or "html" not in ctype.lower():
                return
            results.append((url, body))
            if depth >= self.limits.max_depth:
                return
            soup = BeautifulSoup(body, "lxml")
            for anchor in soup.find_all("a", href=True):
                raw_href = anchor["href"]
                href = raw_href if isinstance(raw_href, str) else str(raw_href)
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue
                link = urljoin(url, href)
                if self.limits.same_domain_only and not _same_domain(link, base_netloc):
                    continue
                if link in visited:
                    continue
                visited.add(link)
                pending.append((link, depth + 1))

        while pending and len(results) < self.limits.max_pages:
            batch: list[tuple[str, int]] = []
            budget = min(self.limits.max_concurrency, self.limits.max_pages - len(results))
            while pending and len(batch) < budget:
                batch.append(pending.popleft())
            await asyncio.gather(*(process(u, d) for u, d in batch))

        return results

    async def _fetch_robot_parser(self, robots_url: str) -> RobotFileParser:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            body, _ = await _fetch_capped(self.client, self.limits, robots_url)
        except (CrawlError, httpx.HTTPError):
            return rp  # unreachable robots.txt -> permissive
        rp.parse(body.decode("utf-8", "replace").splitlines())
        return rp

    @staticmethod
    def _can_fetch(rp: RobotFileParser, url: str) -> bool:
        try:
            return bool(rp.can_fetch("*", url))
        except Exception:  # pragma: no cover - defensive
            return True

    async def _fetch_html(
        self, url: str, sem: asyncio.Semaphore, robots: RobotFileParser
    ) -> tuple[bytes, str | None]:
        validated = validate_public_url(url)
        if not self._can_fetch(robots, validated):
            raise CrawlError("disallowed by robots.txt", details={"url": validated})
        async with sem:
            body, content_type = await _fetch_capped(self.client, self.limits, validated)
        if len(body) > self.limits.max_response_bytes:
            raise CrawlError("response too large", details={"url": validated})
        return body, content_type
