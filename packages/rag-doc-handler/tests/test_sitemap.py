"""SiteCrawler and SitemapCrawler tests with httpx.MockTransport (no real network)."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx
from rag_doc_handler import CrawlLimits, SiteCrawler, SitemapCrawler


def _site_handler(request: httpx.Request) -> httpx.Response:
    path = urlsplit(str(request.url)).path.rstrip("/") or "/"
    pages = {
        "/": '<html><body><a href="/a">A</a><a href="/b">B</a>'
        '<a href="https://other.test/x">external</a></body></html>',
        "/a": '<html><body><a href="/b">B</a><a href="/">home</a></body></html>',
        "/b": '<html><body><a href="/">home</a></body></html>',
    }
    if path == "/robots.txt":
        return httpx.Response(
            200, content=b"User-agent: *\nAllow: /\n", headers={"content-type": "text/plain"}
        )
    if path in pages:
        return httpx.Response(
            200, content=pages[path].encode(), headers={"content-type": "text/html"}
        )
    return httpx.Response(404)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_site_crawler_discovers_same_domain() -> None:
    limits = CrawlLimits(max_pages=10, max_depth=5, max_concurrency=4)
    async with _client(_site_handler) as client:
        results = await SiteCrawler(client, limits).crawl("http://test.local/")
    urls = {u for u, _ in results}
    assert "http://test.local/" in urls
    assert "http://test.local/a" in urls
    assert "http://test.local/b" in urls
    # External domain is filtered out (same_domain_only default True).
    assert not any("other.test" in u for u, _ in results)
    assert len(results) == 3


async def test_site_crawler_respects_page_cap() -> None:
    limits = CrawlLimits(max_pages=2, max_depth=5, max_concurrency=4)
    async with _client(_site_handler) as client:
        results = await SiteCrawler(client, limits).crawl("http://test.local/")
    assert len(results) == 2
    assert all(u.startswith("http://test.local") for u, _ in results)


async def test_site_crawler_respects_depth() -> None:
    # max_depth=1 means only the start page is fetched (no links followed).
    limits = CrawlLimits(max_pages=10, max_depth=1, max_concurrency=4)
    async with _client(_site_handler) as client:
        results = await SiteCrawler(client, limits).crawl("http://test.local/")
    assert len(results) == 1
    assert results[0][0] == "http://test.local/"


async def test_sitemap_crawler_discovers_urls() -> None:
    sitemap_xml = (
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://test.local/1</loc></url>"
        b"<url><loc>http://test.local/2</loc></url>"
        b"</urlset>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlsplit(str(request.url)).path
        if path == "/sitemap.xml":
            return httpx.Response(
                200, content=sitemap_xml, headers={"content-type": "application/xml"}
            )
        return httpx.Response(404)

    limits = CrawlLimits()
    async with _client(handler) as client:
        urls = await SitemapCrawler(client, limits).discover("http://test.local/sitemap.xml")
    assert urls == ["http://test.local/1", "http://test.local/2"]


async def test_sitemap_crawler_recurses_index() -> None:
    index = (
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://test.local/child.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    child = (
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://test.local/p1</loc></url>"
        b"</urlset>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlsplit(str(request.url)).path
        if path == "/sitemap_index.xml":
            return httpx.Response(200, content=index, headers={"content-type": "application/xml"})
        if path == "/child.xml":
            return httpx.Response(200, content=child, headers={"content-type": "application/xml"})
        return httpx.Response(404)

    limits = CrawlLimits()
    async with _client(handler) as client:
        urls = await SitemapCrawler(client, limits).discover("http://test.local/sitemap_index.xml")
    assert urls == ["http://test.local/p1"]
