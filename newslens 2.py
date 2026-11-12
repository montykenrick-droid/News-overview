#!/usr/bin/env python3
"""NewsLens site builder: pulls 20 outlets, groups the same story, writes a static news site.
Needs template.html beside it. Optional: ANTHROPIC_API_KEY for neutral AI summaries."""
import argparse, collections, datetime, hashlib, html, json, os, re, urllib.request
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
    dict(name="BBC News", region="Europe", country="UK", lean=-0.4, url="http://feeds.bbci.co.uk/news/world/rss.xml"),
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
 "Americas": "america american canada canadian mexico mexican brazil argentina chile colombia venezuela peru washington trump",
 "Europe": "britain british london france french germany german paris berlin italy spain ukraine russia poland brussels",
 "Southeast Asia": "singapore thailand thai bangkok indonesia jakarta philippines manila vietnam malaysia myanmar cambodia asean",
 "Pacific": "australia australian sydney fiji samoa tonga pacific papua canberra new-zealand",
 "Middle East & Africa": "israel gaza iran iraq syria saudi qatar egypt turkey africa nigeria kenya sudan lebanon yemen",
 "East & South Asia": "china chinese beijing japan japanese korea india indian pakistan taiwan",
}
LOADED = set("slams blasts destroys shocking outrage furious chaos disaster radical extremist crushes savages humiliates bombshell meltdown scandal rampage".split())
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


def fetch(o, n):
    try:
        out = []
        for e in feedparser.parse(o["url"], agent="Mozilla/5.0 (NewsLens)").entries[:n]:
            t = clean(e.get("title"))
            if "news.google" in o["url"]:
                t = re.sub(r"\s+-\s+[^-]+$", "", t)
            if t:
                out.append(dict(outlet=o["name"], region=o["region"], bucket=bucket(o), title=t,
                                excerpt=clean(e.get("summary"))[:240], link=e.get("link", "#"), img=image_of(e)))
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


def neutral_title(items):
    def key(i):
        return (len(toks(i["title"]) & LOADED) + ("?" in i["title"]) + any(h in i["title"].lower() for h in HEDGE),
                i["outlet"] not in WIRE, abs(len(i["title"]) - 70))
    return min(items, key=key)["title"]


def facts_fallback(items):
    sents = {}
    for i in items:
        for s in re.split(r"(?<=[.!?])\s+", i["excerpt"]):
            if len(s) > 30:
                sents[s] = toks(s) - STOP
    scored = []
    for s, w in sents.items():
        sup = {i["outlet"] for i in items if w and len(w & toks(i["title"] + " " + i["excerpt"])) / len(w) >= .5}
        scored.append((len(sup), s))
    return [dict(t=s, n=n) for n, s in sorted(scored, reverse=True)[:3]]


def llm(items):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    lines = "\n".join(f"[{i['outlet']} | {i['bucket']}] {i['title']} - {i['excerpt']}" for i in items)
    prompt = ("Headlines and snippets about ONE news story from outlets across the political spectrum. Return ONLY JSON: "
              '{"title": neutral factual headline, "summary": 2-3 sentences of verifiable facts only, no opinion, '
              '"facts": [{"text": fact, "outlets": [names of outlets whose text supports it]}], '
              '"unconfirmed": [claims only one outlet makes or that are hedged], "framing_differs": true only if left/centre/right '
              'outlets clearly frame it differently, "framing_note": one sentence or ""}. Use only the text given.\n\n' + lines)
    body = json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 700, "messages": [{"role": "user", "content": prompt}]}).encode()
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
    title = (ai or {}).get("title") or neutral_title(g)
    if ai:
        facts = [dict(t=f["text"], n=len(set(f.get("outlets", [])) & outs)) for f in ai.get("facts", [])][:4]
        facts += [dict(t=u, n=1) for u in ai.get("unconfirmed", [])[:2]]
        summary = ai.get("summary", "")
    else:
        facts = facts_fallback(g)
        wire = next((i for i in g if i["outlet"] in WIRE and i["excerpt"]), None)
        summary = (f"Reported by {len(outs)} outlets. " + (f"{wire['outlet']} reports: {wire['excerpt']}" if wire else ""))
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
                img=next((i["img"] for i in sorted(g, key=lambda i: i["outlet"] not in WIRE) if i["img"]), ""),
                items=[{k: i[k] for k in ("outlet", "bucket", "region", "title", "excerpt", "link", "img", "tone", "flags")} for i in g],
                analysis=dict(show=bool(clear), tones=tones, words=words, note=(ai or {}).get("framing_note", "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public")
    ap.add_argument("--prev", default="prev_roundup.json")
    ap.add_argument("--tz", default="Europe/London", help="time zone for the 6am round-up")
    ap.add_argument("--per-outlet", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--ai-stories", type=int, default=14)
    ap.add_argument("--once", action="store_true")  # accepted for compatibility
    a = ap.parse_args()

    with ThreadPoolExecutor(10) as ex:
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
    multi = [g for g in cl if len({i["outlet"] for i in g}) >= 2][:60]
    singles, per = [], collections.Counter()
    for g in cl:  # niche stories: keep a few per category so every section has content
        if len({i["outlet"] for i in g}) == 1:
            c = category(g[0]["title"] + " " + g[0]["excerpt"])
            if per[c] < 4:
                per[c] += 1
                singles.append(g)
    use_ai = bool(os.environ.get("ANTHROPIC_API_KEY"))
    stories = [build_story(g, use_ai and n < a.ai_stories, None) for n, g in enumerate(multi + singles)]

    now = datetime.datetime.now(ZoneInfo(a.tz))
    edition = (now.date() if now.hour >= 6 else now.date() - datetime.timedelta(days=1)).isoformat()
    try:
        prev = json.load(open(a.prev))
    except Exception:
        prev = {}
    if prev.get("edition") == edition and prev.get("stories"):
        roundup = prev  # frozen until the next 6am edition
    else:
        roundup = dict(edition=edition, built=now.strftime("%H:%M"), stories=[
            dict(id=s["id"], title=s["title"], summary=s["summary"], cat=s["cat"], outlets=s["outlets"])
            for s in stories if not s["limited"]][:8])

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
