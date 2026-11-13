#!/usr/bin/env python3
"""NewsLens site builder: pulls 20 outlets, groups the same story, writes a static news site.
Needs template.html beside it. Optional: ANTHROPIC_API_KEY for neutral AI summaries."""
import argparse, collections, datetime, hashlib, html, json, os, re, socket, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
import feedparser, numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

GN = "https://news.google.com/rss/search?q=site:{}+when:1d&hl=en-US&gl=US&ceid=US:en"
OUTLETS = [
    # --- Global / cross-regional ---
    dict(name="Reuters", region="Global", country="UK/Global", lean=0, url=GN.format("reuters.com")),
    dict(name="Al Jazeera", region="Global", country="Qatar", lean=-0.6, url="https://www.aljazeera.com/xml/rss/all.xml"),
    # --- Americas ---
    dict(name="New York Times", region="Americas", country="USA", lean=-1.1, url="https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    dict(name="Fox News", region="Americas", country="USA", lean=1.9, url="https://moxie.foxnews.com/google-publisher/world.xml"),
    dict(name="The Globe and Mail", region="Americas", country="Canada", lean=0.3, url=GN.format("theglobeandmail.com")),
    dict(name="Mexico News Daily", region="Americas", country="Mexico", lean=0, url="https://mexiconewsdaily.com/feed/"),
    dict(name="MercoPress", region="Americas", country="Uruguay/S. America", lean=0.3, url=GN.format("en.mercopress.com")),
    # --- Europe ---
    dict(name="BBC News", region="Europe", country="UK", lean=-0.4, url="http://feeds.bbci.co.uk/news/rss.xml"),
    dict(name="The Guardian", region="Europe", country="UK", lean=-1.2, url="https://www.theguardian.com/world/rss"),
    dict(name="The Telegraph", region="Europe", country="UK", lean=1.3, url="https://www.telegraph.co.uk/rss.xml"),
    dict(name="Deutsche Welle", region="Europe", country="Germany", lean=-0.3, url="https://rss.dw.com/xml/rss-en-world"),
    # --- Southeast Asia ---
    dict(name="The Straits Times", region="Southeast Asia", country="Singapore", lean=0.2, url="https://www.straitstimes.com/news/world/rss.xml"),
    dict(name="CNA (Channel NewsAsia)", region="Southeast Asia", country="Singapore", lean=0.1, url="https://www.channelnewsasia.com/rssfeeds/8395986"),
    dict(name="Bangkok Post", region="Southeast Asia", country="Thailand", lean=0.3, url="https://www.bangkokpost.com/rss/data/topstories.xml"),
    dict(name="The Jakarta Post", region="Southeast Asia", country="Indonesia", lean=-0.2, url=GN.format("thejakartapost.com")),
    dict(name="Rappler", region="Southeast Asia", country="Philippines", lean=-0.8, url="https://www.rappler.com/feed/"),
    # --- Pacific ---
    dict(name="ABC News (Australia)", region="Pacific", country="Australia", lean=-0.5, url="https://www.abc.net.au/news/feed/2942460/rss.xml"),
    dict(name="Sydney Morning Herald", region="Pacific", country="Australia", lean=-0.5, url="https://www.smh.com.au/rss/world.xml"),
    dict(name="RNZ (Pacific)", region="Pacific", country="NZ / Pacific Islands", lean=-0.3, url="https://www.rnz.co.nz/rss/pacific.xml"),
    dict(name="NZ Herald", region="Pacific", country="New Zealand", lean=0.4, url=GN.format("nzherald.co.nz")),
    # --- More US, UK, Canadian and international outlets ---
    dict(name="Associated Press", region="Americas", country="USA", lean=-0.3, url=GN.format("apnews.com")),
    dict(name="CNN", region="Americas", country="USA", lean=-1.1, url="http://rss.cnn.com/rss/edition.rss"),
    dict(name="Washington Post", region="Americas", country="USA", lean=-1.0, url="https://feeds.washingtonpost.com/rss/world"),
    dict(name="NPR", region="Americas", country="USA", lean=-0.8, url="https://feeds.npr.org/1001/rss.xml"),
    dict(name="Wall Street Journal", region="Americas", country="USA", lean=0.8, url="https://feeds.a.dowjones.com/rss/RSSWorldNews.xml"),
    dict(name="New York Post", region="Americas", country="USA", lean=1.4, url="https://nypost.com/feed/"),
    dict(name="NBC News", region="Americas", country="USA", lean=-0.7, url="https://feeds.nbcnews.com/nbcnews/public/news"),
    dict(name="CBS News", region="Americas", country="USA", lean=-0.3, url="https://www.cbsnews.com/latest/rss/main"),
    dict(name="ABC News (US)", region="Americas", country="USA", lean=-0.6, url="https://abcnews.go.com/abcnews/topstories"),
    dict(name="Politico", region="Americas", country="USA", lean=-0.4, url="https://rss.politico.com/politics-news.xml"),
    dict(name="USA Today", region="Americas", country="USA", lean=-0.3, url="http://rssfeeds.usatoday.com/usatoday-NewsTopStories"),
    dict(name="The Hill", region="Americas", country="USA", lean=0.2, url="https://thehill.com/feed/"),
    dict(name="ESPN", region="Americas", country="USA", lean=-0.2, url="https://www.espn.com/espn/rss/news"),
    dict(name="CBC News", region="Americas", country="Canada", lean=-0.4, url="https://www.cbc.ca/webfeed/rss/rss-topstories"),
    dict(name="Sky News", region="Europe", country="UK", lean=-0.1, url="https://feeds.skynews.com/feeds/rss/world.xml"),
    dict(name="The Independent", region="Europe", country="UK", lean=-0.7, url="https://www.independent.co.uk/news/world/rss"),
    dict(name="Daily Mail", region="Europe", country="UK", lean=1.7, url="https://www.dailymail.co.uk/news/index.rss"),
    dict(name="France 24", region="Europe", country="France", lean=-0.2, url="https://www.france24.com/en/rss"),
    dict(name="Times of India", region="East & South Asia", country="India", lean=0.6, url="https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"),
    dict(name="South China Morning Post", region="East & South Asia", country="Hong Kong", lean=0.1, url="https://www.scmp.com/rss/91/feed"),
    dict(name="The Japan Times", region="East & South Asia", country="Japan", lean=-0.1, url="https://www.japantimes.co.jp/feed/"),
    # --- UK broadcasters, more right-leaning outlets, extra balance ---
    dict(name="ITV News", region="Europe", country="UK", lean=-0.1, url=GN.format("itv.com")),
    dict(name="Channel 4 News", region="Europe", country="UK", lean=-0.5, url=GN.format("channel4.com")),
    dict(name="GB News", region="Europe", country="UK", lean=1.7, url=GN.format("gbnews.com")),
    dict(name="Daily Express", region="Europe", country="UK", lean=1.5, url=GN.format("express.co.uk")),
    dict(name="The Times", region="Europe", country="UK", lean=0.6, url=GN.format("thetimes.com+OR+site:thetimes.co.uk")),
    dict(name="Daily Mirror", region="Europe", country="UK", lean=-1.3, url=GN.format("mirror.co.uk")),
    dict(name="Euronews", region="Europe", country="France", lean=-0.2, url=GN.format("euronews.com")),
    dict(name="Newsmax", region="Americas", country="USA", lean=1.9, url=GN.format("newsmax.com")),
    dict(name="Washington Examiner", region="Americas", country="USA", lean=1.5, url=GN.format("washingtonexaminer.com")),
    dict(name="Washington Times", region="Americas", country="USA", lean=1.4, url=GN.format("washingtontimes.com")),
    dict(name="HuffPost", region="Americas", country="USA", lean=-1.6, url=GN.format("huffpost.com")),
    dict(name="Bloomberg", region="Global", country="USA", lean=-0.1, url=GN.format("bloomberg.com")),
]

CATS = ["World", "Politics", "Business", "Sport", "Culture", "Environment", "Health", "Tech", "Crime & Justice"]
KW = {
 "Politics": "election president minister government parliament congress senate vote party policy lawmakers campaign coalition cabinet",
 "Business": "economy market inflation stocks bank trade jobs rate company profit recession oil prices growth debt tariff earnings",
 "Sport": "football soccer cricket tennis olympic league match goal coach tournament championship rugby golf athlete cup",
 "Culture": "film music festival album artist museum theatre award book celebrity royal fashion series actor",
 "Environment": "climate warming emissions wildfire flood storm typhoon cyclone earthquake species reef energy drought",
 "Health": "health virus vaccine hospital disease cancer drug outbreak medical doctors",
 "Tech": "tech ai artificial intelligence apple google microsoft meta openai cyber chip software robot",
 "Crime & Justice": "police arrest murder trial jailed prison charged shooting suspect court",
}
REG = {
 "UK": "uk britain british england scotland scottish wales welsh london westminster nhs starmer holyrood tory tories labour",
 "Americas": "america american canada canadian mexico mexican brazil argentina chile colombia venezuela peru washington trump",
 "Europe": "france french germany german paris berlin italy spain ukraine russia poland brussels",
 "Southeast Asia": "singapore thailand thai bangkok indonesia jakarta philippines manila vietnam malaysia myanmar cambodia asean",
 "Pacific": "australia australian sydney fiji samoa tonga pacific papua canberra new-zealand",
 "Middle East & Africa": "israel gaza iran iraq syria saudi qatar egypt turkey africa nigeria kenya sudan lebanon yemen",
 "East & South Asia": "china chinese beijing japan japanese korea india indian pakistan taiwan",
}
LOADED = set("horror horrifying devastating brutal stunning explosive savage fury fumes erupts rages rips torches mocks panic terrifying tragic heartbreaking unbelievable chilling shocked slammed blasted disgraceful despicable epic slams blasts destroys shocking outrage furious chaos disaster radical extremist crushes savages humiliates bombshell meltdown scandal rampage".split())
HEDGE = ("reportedly", "allegedly", "claims", "sources say", "unconfirmed", "rumour", "rumor", "alleged")
WIRE = ("Reuters", "BBC News", "Deutsche Welle")
vader = SentimentIntensityAnalyzer()


def bucket(o):
    return "State-run" if o.get("state") else "Left" if o["lean"] <= -0.7 else "Right" if o["lean"] >= 0.7 else "Centre"


def clean(t):
    return html.unescape(re.sub(r"<[^>]+>", " ", t or "")).strip()


def toks(t):
    t = t.lower().replace("new zealand", "new-zealand")
    return set(re.findall(r"[a-z][a-z\-]+", t))


def image_of(e):
    for k in ("media_thumbnail", "media_content"):
        for m in e.get(k, []) or []:
            if m.get("url"):
                return m["url"]
    for l in e.get("links", []):
        if "image" in l.get("type", ""):
            return l.get("href")
    m = re.search(r'<img[^>]+src="([^"]+)"', e.get("summary", "") or "")
    return m.group(1) if m else ""


def https(u):
    return "https:" + u if u.startswith("//") else u.replace("http://", "https://", 1)


def fetch(o, n):
    try:
        out = []
        for e in feedparser.parse(o["url"], agent="Mozilla/5.0 (NewsLens)").entries[:n]:
            t = clean(e.get("title"))
            if "news.google" in o["url"]:
                t = re.sub(r"\s+-\s+[^-]+$", "", t)
            if t:
                out.append(dict(outlet=o["name"], region=o["region"], bucket=bucket(o), title=t,
                                excerpt=clean(e.get("summary"))[:240], link=e.get("link", "#"), img=https(image_of(e))))
        return out
    except Exception as ex:
        print("  !", o["name"], ex)
        return []


def category(text):
    w = toks(text)
    s = {k: len(w & set(v.split())) for k, v in KW.items()}
    b = max(s, key=s.get)
    return b if s[b] else "World"


def regions(text):
    w = toks(text)
    return [k for k, v in REG.items() if w & set(v.split())]


def flags(i, o_state):
    f = []
    bad = sorted(toks(i["title"]) & LOADED)
    if bad:
        f.append("Emotive wording in headline: " + ", ".join(bad))
    if any(h in (i["title"] + " " + i["excerpt"]).lower() for h in HEDGE):
        f.append("Contains hedged or unconfirmed claim language")
    if o_state:
        f.append("State-run outlet: consider independent sources")
    return f


def is_loaded(t):
    tl = t.lower()
    return bool(toks(t) & LOADED) or any(c in t for c in '?!\u201c\u201d"') or any(h in tl for h in HEDGE) or " you " in f" {tl} "


def best_headline(items):
    return min(items, key=lambda i: (is_loaded(i["title"]), i["outlet"] not in WIRE, abs(len(i["title"]) - 70)))["title"]


def generic_title(items):
    """General overview built from the terms most outlets share, used when no clean headline exists."""
    c = collections.Counter()
    for o in {i["outlet"] for i in items}:
        ws = set()
        for i in items:
            if i["outlet"] == o:
                ws |= toks(i["title"])
        c.update(w for w in ws if len(w) > 3 and w not in STOP and w not in LOADED)
    top = [w.title() for w, _ in c.most_common(3)]
    return "Latest reports: " + ", ".join(top) if top else items[0]["title"]


def neutral_title(items):
    b = best_headline(items)
    return generic_title(items) if is_loaded(b) else b


def facts_fallback(items):
    sents = {}
    for i in items:
        for s in re.split(r"(?<=[.!?])\s+", i["excerpt"]):
            if len(s) > 30:
                sents[s] = toks(s) - STOP
    scored = []
    for s, w in sents.items():
        sup = {i["outlet"] for i in items if w and len(w & toks(i["title"] + " " + i["excerpt"])) / len(w) >= .5}
        scored.append((len(sup), s, sorted(sup)))
    return [dict(t=s, n=n, by=by) for n, s, by in sorted(scored, reverse=True)[:3]]


SITES = {"BBC News": "bbc.co.uk", "ITV News": "itv.com", "Channel 4 News": "channel4.com", "Sky News": "news.sky.com",
 "The Guardian": "theguardian.com", "The Telegraph": "telegraph.co.uk", "Daily Mail": "dailymail.co.uk",
 "Daily Express": "express.co.uk", "The Times": "thetimes.com", "GB News": "gbnews.com", "The Independent": "independent.co.uk",
 "Daily Mirror": "mirror.co.uk", "Fox News": "foxnews.com", "New York Post": "nypost.com", "Newsmax": "newsmax.com",
 "Washington Examiner": "washingtonexaminer.com", "Washington Times": "washingtontimes.com", "Wall Street Journal": "wsj.com",
 "CNN": "cnn.com", "New York Times": "nytimes.com", "Washington Post": "washingtonpost.com", "NBC News": "nbcnews.com",
 "CBS News": "cbsnews.com", "ABC News (US)": "abcnews.go.com", "NPR": "npr.org", "Politico": "politico.com",
 "USA Today": "usatoday.com", "The Hill": "thehill.com", "Reuters": "reuters.com", "Associated Press": "apnews.com",
 "Al Jazeera": "aljazeera.com", "Deutsche Welle": "dw.com", "France 24": "france24.com", "CBC News": "cbc.ca",
 "Bloomberg": "bloomberg.com", "HuffPost": "huffpost.com", "Euronews": "euronews.com"}


def sample(items, n=24):
    """Pick up to n items, rotating through Left/Centre/Right/State so the AI sees every side."""
    by = collections.defaultdict(list)
    for i in items:
        by[i["bucket"]].append(i)
    out = []
    while len(out) < min(n, len(items)):
        for b in list(by):
            if by[b] and len(out) < n:
                out.append(by[b].pop(0))
    return out


def backfill(groups):
    """For each big story, search for the major outlets whose feed did not list it, so every big story
    shows as many outlets (and political sides) as possible."""
    by = {o["name"]: o for o in OUTLETS}
    jobs = []
    for g in groups:
        have = {i["outlet"] for i in g}
        q = best_headline(g)
        jobs += [(g, n, site, q) for n, site in SITES.items() if n not in have and n in by]

    def run(j):
        g, name, site, q = j
        url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote_plus(f"{q} site:{site} when:2d")
               + "&hl=en-US&gl=US&ceid=US:en")
        try:
            ents = feedparser.parse(url, agent="Mozilla/5.0 (NewsLens)").entries[:3]
        except Exception:
            return None
        vocab = set().union(*[toks(i["title"] + " " + i["excerpt"]) for i in g]) - STOP
        best, score = None, 0
        for e in ents:
            t = re.sub(r"\s+-\s+[^-]+$", "", clean(e.get("title")))
            w = toks(t) - STOP
            hit = len(w & vocab)
            if w and hit >= 3 and hit / len(w) >= .5 and hit > score:
                best, score = dict(outlet=name, region=by[name]["region"], bucket=bucket(by[name]), title=t,
                                   excerpt="", link=e.get("link", "#"), img=""), hit
        return (g, best) if best else None

    with ThreadPoolExecutor(16) as ex:
        for r in ex.map(run, jobs):
            if r:
                r[0].append(r[1])
    print(f"  backfill: {sum(len({i['outlet'] for i in g}) for g in groups)} outlet-story links across {len(groups)} big stories")


UA_B = {"User-Agent": "Mozilla/5.0 (compatible; NewsLens/1.0)"}


def meta(page, *names):
    for n in names:
        for pat in (r'<meta[^>]+(?:property|name)=["\']%s["\'][^>]+content=["\']([^"\']+)' % n,
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']%s["\']' % n):
            m = re.search(pat, page, re.I)
            if m:
                return html.unescape(m.group(1)).strip()
    return ""


def enrich(groups, limit=90):
    """Fill missing photos and snippets from each article's own preview tags (og:image / og:description)."""
    jobs = []
    for g in groups:
        n = 0
        for i in g:
            if (not i["img"] or not i["excerpt"]) and "news.google" not in i["link"] and n < 3 and len(jobs) < limit:
                jobs.append(i)
                n += 1

    def run(i):
        try:
            page = urllib.request.urlopen(urllib.request.Request(i["link"], headers=UA_B), timeout=10).read(200000).decode("utf-8", "ignore")
        except Exception:
            return
        if not i["img"]:
            u = meta(page, "og:image", "twitter:image")
            if u.startswith(("http", "//")):
                i["img"] = https(u)
        if not i["excerpt"]:
            i["excerpt"] = meta(page, "og:description", "description")[:240]

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(run, jobs))


def validate_imgs(groups, limit=500):
    """Drop image links that are dead or not really images, so the site falls back to its placeholder."""
    urls = list(dict.fromkeys(i["img"] for g in groups for i in g if i["img"]))[:limit]

    def ok(u):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers={**UA_B, "Range": "bytes=0-1023"}), timeout=8)
            return r.status in (200, 206) and "image" in r.headers.get("Content-Type", "")
        except Exception:
            return False

    with ThreadPoolExecutor(16) as ex:
        good = {u for u, k in zip(urls, ex.map(ok, urls)) if k}
    bad = 0
    for g in groups:
        for i in g:
            if i["img"] and i["img"] not in good:
                i["img"], bad = "", bad + 1
    print(f"  images: {len(good)} working links, {bad} dropped")


def llm(items):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    lines = "\n".join(f"[{i['outlet']} | {i['bucket']}] {i['title']} - {i['excerpt']}" for i in sample(items))
    prompt = ("Headlines and snippets about ONE news story from outlets across the political spectrum. Return ONLY JSON: "
              '{"title": a calm, general overview of what is happening in under 12 words: describe events, no adjectives or emotive verbs, no quotes, no questions, no blame or motives, nothing stated as fact that is only claimed, "summary": 2-3 sentences of verifiable facts only, no opinion, '
              '"facts": [{"text": fact, "outlets": [names of outlets whose text supports it]}], '
              '"unconfirmed": [claims only one outlet makes or that are hedged], "framing_differs": true only if left/centre/right '
              'outlets clearly frame it differently, "framing_note": one sentence or "", "per_outlet": {outlet name: one sentence on what that outlet text reports}}. Use only the text given.\n\n' + lines)
    body = json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 1600, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(req, timeout=60))["content"][0]["text"]
        return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    except Exception as ex:
        print("  AI summary failed:", ex)
        return None


def build_story(g, use_ai, states):
    for i in g:
        i["tone"] = round(vader.polarity_scores(i["title"])["compound"], 2)
        i["flags"] = flags(i, i["bucket"] == "State-run")
    text = " ".join(i["title"] + " " + i["excerpt"] for i in g)
    outs = {i["outlet"] for i in g}
    ai = llm(g) if use_ai else None
    t_ai = (ai or {}).get("title")
    title = t_ai if t_ai and not is_loaded(t_ai) else neutral_title(g)
    if ai:
        facts = []
        for f in ai.get("facts", [])[:4]:
            by = sorted(set(f.get("outlets", [])) & outs)
            facts.append(dict(t=f["text"], n=len(by), by=by))
        facts += [dict(t=u, n=1, by=[]) for u in ai.get("unconfirmed", [])[:2]]
        summary = ai.get("summary", "")
    else:
        facts = facts_fallback(g)
        wire = next((i for i in g if i["outlet"] in WIRE and i["excerpt"]), None)
        summary = (f"Reported by {len(outs)} outlets. " + (f"{wire['outlet']} reports: {wire['excerpt']}" if wire else ""))
    per = (ai or {}).get("per_outlet", {})
    for i in g:
        i["summary"] = per.get(i["outlet"], "") if isinstance(per, dict) else ""
    for f in facts:
        f["conf"] = "confirmed" if f["n"] >= 3 else "partial" if f["n"] == 2 else "unconfirmed"
    tones = {}
    for b in ("Left", "Centre", "Right"):
        v = [i["tone"] for i in g if i["bucket"] == b]
        if v:
            tones[b] = round(float(np.mean(v)), 2)
    words = {}
    if len(tones) > 1:
        sets = {b: set().union(*[toks(i["title"]) - STOP for i in g if i["bucket"] == b]) for b in tones}
        for b in tones:
            others = set().union(*[s for k, s in sets.items() if k != b])
            words[b] = sorted(sets[b] - others, key=len, reverse=True)[:4]
    clear = len(tones) > 1 and (max(tones.values()) - min(tones.values()) >= .3 or (ai or {}).get("framing_differs"))
    cat = category(text)
    reg = regions(text) or ([g[0]["region"]] if len(outs) <= 2 and g[0]["region"] != "Global" else ["Global"])
    return dict(id=hashlib.sha1(min(i["link"] for i in g).encode()).hexdigest()[:8], title=title, summary=summary,
                facts=facts, cat=cat, regions=reg, outlets=len(outs), limited=len(outs) < 3,
                imgs=list(dict.fromkeys(i["img"] for i in sorted(g, key=lambda i: i["outlet"] not in WIRE) if i["img"]))[:4],
                items=[{k: i[k] for k in ("outlet", "bucket", "region", "title", "excerpt", "summary", "link", "img", "tone", "flags")} for i in g],
                analysis=dict(show=bool(clear), tones=tones, words=words, note=(ai or {}).get("framing_note", "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public")
    ap.add_argument("--prev", default="prev_roundup.json")
    ap.add_argument("--tz", default="Europe/London", help="time zone for the 6am round-up")
    ap.add_argument("--per-outlet", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.78)
    ap.add_argument("--ai-stories", type=int, default=14)
    ap.add_argument("--min-home", type=int, default=14, help="minimum stories on the homepage")
    ap.add_argument("--backfill-stories", type=int, default=10, help="how many top stories to search extra outlets for")
    ap.add_argument("--once", action="store_true")  # accepted for compatibility
    a = ap.parse_args()
    socket.setdefaulttimeout(20)

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda o: fetch(o, a.per_outlet), OUTLETS))
    for o, r in zip(OUTLETS, res):
        print(f"  {o['name']:28s} {len(r)}")
    items = [i for r in res for i in r]
    if len(items) < 20:
        raise SystemExit("Too few articles fetched.")
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [f"{i['title']} {i['title']} {i['excerpt']}" for i in items])
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=a.threshold).fit_predict(X.toarray())
    groups = collections.defaultdict(list)
    for i, l in zip(items, labels):
        groups[l].append(i)
    cl = sorted(groups.values(), key=lambda g: (-len({i["outlet"] for i in g}), -len({i["bucket"] for i in g})))
    multi = [g for g in cl if len({i["outlet"] for i in g}) >= 3][:80]
    backfill(multi[:a.backfill_stories])
    multi.sort(key=lambda g: (-len({i["outlet"] for i in g}), -len({i["bucket"] for i in g})))
    if len(multi) < a.min_home:  # quiet news day: top up with the widest 2-outlet stories
        multi += [g for g in cl if len({i["outlet"] for i in g}) == 2][: a.min_home - len(multi)]
    hidden, per = [], collections.Counter()
    for g in cl:  # stories few outlets cover: shown only under "Hidden stories"
        if len({i["outlet"] for i in g}) <= 2 and len(g[0]["title"]) > 25 and not any(g is m for m in multi):
            c = category(g[0]["title"] + " " + g[0]["excerpt"])
            if per[c] < 5:
                per[c] += 1
                hidden.append(g)
    print(f"  {len(multi)} main stories, {len(hidden)} hidden stories")
    enrich(multi + hidden)
    validate_imgs(multi + hidden)
    use_ai = bool(os.environ.get("ANTHROPIC_API_KEY"))
    stories = [build_story(g, use_ai and n < a.ai_stories, None) for n, g in enumerate(multi)]
    stories += [dict(build_story(g, False, None), hidden=True) for g in hidden]

    now = datetime.datetime.now(ZoneInfo(a.tz))
    edition = (now.date() if now.hour >= 6 else now.date() - datetime.timedelta(days=1)).isoformat()
    try:
        prev = json.load(open(a.prev))
    except Exception:
        prev = {}
    if prev.get("v") == 2 and prev.get("edition") == edition and len(prev.get("stories", [])) >= 6:
        roundup = prev  # frozen until the next 6am edition
    else:
        roundup = dict(v=2, edition=edition, built=now.strftime("%H:%M"), stories=[
            dict(id=s["id"], title=s["title"], summary=s["summary"], cat=s["cat"], outlets=s["outlets"])
            for s in stories if not s.get("hidden")][:10])

    os.makedirs(a.out, exist_ok=True)
    data = dict(updated=now.strftime("%d %b %Y, %H:%M"), cats=CATS, regions=list(REG) + ["Global"], stories=stories,
                roundup=roundup, outlets=[dict(name=o["name"], bucket=bucket(o), region=o["region"]) for o in OUTLETS], ai=use_ai)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"), encoding="utf-8").read()
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", blob))
    json.dump(roundup, open(os.path.join(a.out, "roundup.json"), "w"))
    print(f"Built {len(stories)} stories -> {a.out}/index.html")


if __name__ == "__main__":
    main()
