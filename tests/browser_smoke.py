"""End-to-end browser smoke check for the Streamlit dashboard.

This drives the running app in a real Chromium and fails on any Streamlit
`stException` rendered on screen. It complements the pytest suite: several
defects (for example a KeyError in the Quantum Lab "Run on Simulator" path that
only triggers when the editor code already contains `measure_all()`) are not
reachable through Streamlit's headless AppTest harness, but do surface here.

Usage
-----
    # terminal 1
    streamlit run app/app.py --server.headless true --server.port 8530
    # terminal 2
    python tests/browser_smoke.py --url http://localhost:8530

Requires: playwright (python) with the chromium browser installed.
Exit code 0 = every exercised path rendered without an exception.
"""
from __future__ import annotations

import argparse
import sys

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    print("playwright is not installed: python -m pip install playwright && playwright install chromium")
    raise SystemExit(2)


def exceptions_on_page(page) -> list[str]:
    out = []
    for el in page.query_selector_all('[data-testid="stException"]'):
        txt = (el.inner_text() or "").strip()
        if txt:
            out.append(txt)
    return out


class Smoke:
    def __init__(self) -> None:
        self.steps: list[bool] = []

    def check(self, page, label: str) -> None:
        exc = exceptions_on_page(page)
        self.steps.append(not exc)
        print(f"  [{'CLEAN' if not exc else str(len(exc)) + ' EXCEPTION(S)'}] {label}")
        for e in exc:
            print("     >>", e.replace("\n", " | ")[:400])


def wait_for_ready(page, selector: str, attempts: int = 40, interval_ms: int = 3_000) -> bool:
    """Wait until `selector` exists. The first load runs a full script (torch model
    loading), so navigating away too early can overlap two script runs in the same
    server process and surface spurious partially-initialised-import errors."""
    for _ in range(attempts):
        page.wait_for_timeout(interval_ms)
        if page.query_selector(selector):
            return True
    return False


def run(url: str) -> int:
    s = Smoke()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        print("=== dashboard (/) ===")
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)
        ready = wait_for_ready(page, 'button:has-text("Generate")')
        print(f"  dashboard ready: {ready}")
        s.check(page, "initial render")

        btn = page.query_selector('button:has-text("Generate")')
        if btn:
            btn.scroll_into_view_if_needed()
            btn.click()
            page.wait_for_timeout(22_000)
            s.check(page, "Generate & Infer")
            for tab in page.query_selector_all('[role="tab"]'):
                name = (tab.inner_text() or "").strip()
                tab.click()
                page.wait_for_timeout(4_500)
                s.check(page, f"result tab '{name}'")
        else:
            print("  NOTE 'Generate & Infer' button not found")
            s.steps.append(False)

        print("=== Quantum Lab (/Quantum_Lab) ===")
        page.goto(url.rstrip("/") + "/Quantum_Lab", wait_until="domcontentloaded", timeout=90_000)
        ready = wait_for_ready(page, 'button:has-text("Run on Simulator")')
        print(f"  lab ready: {ready}")
        page.wait_for_timeout(3_000)
        s.check(page, "initial render")

        rs = page.query_selector('button:has-text("Run on Simulator")')
        if rs:
            rs.click()
            page.wait_for_timeout(16_000)
            s.check(page, "Run on Simulator (default editor code)")
            for view in ("Counts", "Probabilities", "Statevector"):
                lbl = page.query_selector(f'label:has-text("{view}")')
                if lbl:
                    lbl.click()
                    page.wait_for_timeout(4_000)
                    s.check(page, f"measurement view '{view}'")
        else:
            print("  NOTE 'Run on Simulator' button not found")
            s.steps.append(False)

        rp = page.query_selector('button:has-text("Run Pipeline")')
        if rp:
            rp.click()
            page.wait_for_timeout(20_000)
            s.check(page, "Run Pipeline — Real Model")
        else:
            print("  NOTE 'Run Pipeline' button not found")
            s.steps.append(False)

        browser.close()

    print("=" * 70)
    print(f"steps clean : {sum(s.steps)}/{len(s.steps)}")
    print(f"page errors : {len(page_errors)}")
    for e in page_errors[:5]:
        print("   ", e[:250])
    return 0 if all(s.steps) and not page_errors else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8530",
                    help="base URL of the running Streamlit app")
    return run(ap.parse_args().url)


if __name__ == "__main__":
    raise SystemExit(main())
