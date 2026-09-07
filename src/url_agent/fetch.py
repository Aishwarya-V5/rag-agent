from curl_cffi import requests as cffi_requests
import trafilatura

def fetch_and_extract(url: str, timeout: int = 10):
    """
    Fetches a URL and extracts clean article text.
    Returns (success: bool, text_or_error: str)
    """
    try:
        response = cffi_requests.get(
            url,
            timeout=timeout,
            impersonate="chrome124"
        )
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code} error"
    except Exception as e:
        return False, f"Failed to fetch: {e}"

    extracted = trafilatura.extract(response.text, include_comments=False, include_tables=True)

    if not extracted or not extracted.strip():
        return False, "No readable content extracted from page."

    return True, extracted


def fetch_all(urls: list, timeout: int = 10):
    """
    Fetches and extracts content from a list of URLs.
    Returns a list of dicts: {"url": ..., "success": bool, "content": str}
    """
    results = []
    for url in urls:
        success, content = fetch_and_extract(url, timeout=timeout)
        results.append({
            "url": url,
            "success": success,
            "content": content if success else None,
            "error": content if not success else None
        })
        print(f"{'OK' if success else 'FAILED'}: {url}")
    return results


if __name__ == "__main__":
    test_urls = [
        "https://www.netapp.com/",
    ]
    results = fetch_all(test_urls)
    for r in results:
        print(f"\nURL: {r['url']}")
        if r["success"]:
            print(f"Content preview: {r['content'][:300]}")
        else:
            print(f"Error: {r['error']}")