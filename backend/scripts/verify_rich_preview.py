#!/usr/bin/env python
"""Verify rich preview machinery for a job (and optionally a clip).

Usage:
    python -m backend.scripts.verify_rich_preview <job_id> [clip_id]

Checks:
  1. Thumbnail endpoint returns 200 with correct headers
  2. Crawler user-agent on /analysis/{job_id} returns OG tags
  3. Dedicated share route returns OG tags
  4. Regular browser request passes through to SPA
  5. (If clip_id given) /seo/{job_id}/{clip_id} returns full OG tags
  6. (If clip_id given) /share/clip/{job_id}/{clip_id} returns full OG tags
  7. Per-clip thumbnail endpoint with fallback
  8. Per-platform requirement validation
"""

import subprocess
import sys
import os
import re


CRAWLER_UAS = {
    "Facebook":  "facebookexternalhit/1.1",
    "Twitter":   "Twitterbot/1.0",
    "Discord":   "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)",
    "iMessage":  "Applebot/0.1 +http://www.apple.com/go/applebot",
    "WhatsApp":  "WhatsApp/2.21.4.22 A",
    "LinkedIn":  "LinkedInBot/1.0",
    "Telegram":  "TelegramBot (like TwitterBot)",
    "Slack":     "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
    "Reddit":    "Redditbot/1.0",
}


def _header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _curl_get(url, ua=None, head_only=False):
    cmd = ["curl", "-sL"]
    if head_only:
        cmd.append("-I")
    if ua:
        cmd.extend(["-A", ua])
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout or ""
    except Exception as e:
        return f"(error: {e})"


def _check_og(body, checks, label=""):
    """Check for required OG tags in HTML body. Returns (pass_count, fail_count)."""
    p, f = 0, 0
    for tag, desc in checks:
        if tag in body:
            p += 1
        else:
            print(f"  [FAIL] {label}: missing {desc} ({tag})")
            f += 1
    return p, f


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.verify_rich_preview <job_id> [clip_id]")
        sys.exit(1)

    job_id = sys.argv[1]
    clip_id = sys.argv[2] if len(sys.argv) > 2 else None
    base = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

    print(f"Verifying rich preview for job: {job_id}")
    if clip_id:
        print(f"Clip: {clip_id}")
    print(f"Base URL: {base}")

    total_pass = 0
    total_fail = 0

    # 1. Thumbnail endpoint
    _header("1. Thumbnail endpoint")
    resp = _curl_get(f"{base}/thumbnails/{job_id}.jpg", head_only=True)
    print(resp[:500])
    if "image/jpeg" in resp.lower() or "200" in resp:
        print("  [PASS] Job thumbnail returns image/jpeg")
        total_pass += 1
    else:
        print("  [FAIL] Job thumbnail not returning correctly")
        total_fail += 1

    # 2. Crawler-injected analysis page
    _header("2. Crawler-injected analysis page (Slackbot UA)")
    body = _curl_get(f"{base}/analysis/{job_id}", ua=CRAWLER_UAS["Slack"])
    p, f = _check_og(body, [
        ("og:image", "og:image"),
        ("og:title", "og:title"),
        ("og:description", "og:description"),
    ], "analysis")
    total_pass += p
    total_fail += f

    # 3. Dedicated share route
    _header("3. Share route /share/analysis/{job_id}")
    body = _curl_get(f"{base}/share/analysis/{job_id}")
    p, f = _check_og(body, [("og:title", "og:title")], "share/analysis")
    total_pass += p
    total_fail += f

    # 4. Browser pass-through
    _header("4. Browser request (should be SPA, not stub)")
    body = _curl_get(
        f"{base}/analysis/{job_id}",
        ua="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        head_only=True,
    )
    print(body[:300])

    # ── Clip-specific checks ──
    if clip_id:
        _header(f"5. Per-clip thumbnail /thumbnails/{job_id}/{clip_id}.jpg")
        resp = _curl_get(f"{base}/thumbnails/{job_id}/{clip_id}.jpg", head_only=True)
        if "200" in resp:
            print("  [PASS] Per-clip thumbnail returns 200")
            total_pass += 1
        else:
            print("  [INFO] Per-clip thumbnail not found (fallback expected)")

        _header("6. Per-platform crawler matrix on /seo path")
        seo_url = f"{base}/seo/{job_id}/{clip_id}"
        share_url = f"{base}/share/clip/{job_id}/{clip_id}"

        platform_results = {}
        for platform, ua in CRAWLER_UAS.items():
            body = _curl_get(seo_url, ua=ua)
            checks = [
                ("og:image", "og:image"),
                ("og:title", "og:title"),
                ("og:image:secure_url", "og:image:secure_url"),
                ("theme-color", "theme-color"),
            ]
            # Platform-specific checks
            if platform == "Twitter":
                checks.append(("twitter:card", "twitter:card"))
            if platform == "Discord":
                checks.append(("og:video:type", "og:video:type (for inline play)"))

            p, f = 0, 0
            for tag, desc in checks:
                if tag in body:
                    p += 1
                else:
                    f += 1
            passed = f == 0
            platform_results[platform] = "PASS" if passed else "FAIL"
            status = "PASS" if passed else f"FAIL ({f} missing)"
            print(f"  {platform:12s} [{status}]")

        _header("7. Share clip route /share/clip/{job_id}/{clip_id}")
        body = _curl_get(share_url)
        p, f = _check_og(body, [
            ("og:title", "og:title"),
            ("og:image", "og:image"),
            ("og:image:secure_url", "og:image:secure_url"),
        ], "share/clip")
        total_pass += p
        total_fail += f

        # Validate og:image URL returns 200 image/jpeg
        _header("8. Validate og:image URL is reachable")
        img_match = re.search(r'og:image" content="([^"]+)"', body)
        if img_match:
            img_url = img_match.group(1)
            img_resp = _curl_get(img_url, head_only=True)
            if "image/jpeg" in img_resp.lower() or "200" in img_resp:
                print(f"  [PASS] og:image URL returns image/jpeg: {img_url}")
                total_pass += 1
            else:
                print(f"  [FAIL] og:image URL not reachable: {img_url}")
                total_fail += 1

        # Check og:video if present
        vid_match = re.search(r'og:video" content="([^"]+)"', body)
        if vid_match:
            vid_url = vid_match.group(1)
            vid_resp = _curl_get(vid_url, head_only=True)
            if "video/mp4" in vid_resp.lower() or "200" in vid_resp:
                print(f"  [PASS] og:video URL returns video/mp4: {vid_url}")
                total_pass += 1
            else:
                print(f"  [WARN] og:video URL not reachable: {vid_url}")

    # Summary
    _header("Summary")
    print(f"  Passed: {total_pass}")
    print(f"  Failed: {total_fail}")

    _header("Manual verification links")
    if clip_id:
        print(f"  Facebook debugger:")
        print(f"    https://developers.facebook.com/tools/debug/?q={base}/share/clip/{job_id}/{clip_id}")
        print(f"  LinkedIn Post Inspector:")
        print(f"    https://www.linkedin.com/post-inspector/inspect/{base}/seo/{job_id}/{clip_id}")
    else:
        print(f"  Facebook debugger:")
        print(f"    https://developers.facebook.com/tools/debug/?q={base}/share/analysis/{job_id}")
    print(f"  Twitter validator:")
    print(f"    https://cards-dev.twitter.com/validator")
    print(f"  Slack: paste into a channel:")
    if clip_id:
        print(f"    {base}/seo/{job_id}/{clip_id}")
    else:
        print(f"    {base}/share/analysis/{job_id}")
    print(f"  Discord: paste into a DM (append ?v=2 to bust cache):")
    if clip_id:
        print(f"    {base}/seo/{job_id}/{clip_id}?v=2")
    print()

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
