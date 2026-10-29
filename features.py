import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account", "bank",
                    "confirm", "password", "signin", "paypal", "free", "bonus"]
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd"]
RISKY_TLDS = {"zip", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "support"}

def extract_features(url: str) -> dict:
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.lower()
    host_no_port = host.split(":")[0]
    low = url.lower()
    digits = sum(c.isdigit() for c in url)

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "num_dots": url.count("."),
        "num_hyphens_host": host.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "digit_ratio": digits / max(len(url), 1),
        "num_special": len(re.findall(r"[^\w]", url)),
        "num_subdomains": max(host_no_port.count(".") - 1, 0),
        "has_at": int("@" in url),
        "has_ip": int(bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host_no_port))),
        "uses_https": int(parsed.scheme == "https"),
        "has_port": int(":" in host),
        "num_params": parsed.query.count("&") + (1 if parsed.query else 0),
        "suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "is_shortener": int(any(s in host_no_port for s in SHORTENERS)),
        "risky_tld": int(host_no_port.rsplit(".", 1)[-1] in RISKY_TLDS),
    }