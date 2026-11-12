#!/usr/bin/env python3
"""
NewsLens - compare 20 major news outlets worldwide.

Pulls live RSS feeds, then reports:
  1. Topic coverage per outlet (heatmap)
  2. Political-lean spectrum + average headline tone per outlet
  3. Shared stories: the same event as framed by left / centre / right / state-run outlets
  4. Words that left- vs right-leaning outlets use disproportionately
  5. (optional) an AI summary of framing differences per story

Setup:   pip install feedparser scikit-learn vaderSentiment numpy
Run:     python newslens.py            -> builds newslens_report.html, opens it, then refreshes
                                         everything automatically every 60 minutes (Ctrl+C to stop)
Options: --interval 60 (minutes)  --once (single run, no loop)  --per-outlet 40
         --threshold 0.72  --stories 12  --no-open
Optional AI framing summaries: set ANTHROPIC_API_KEY before running.
"""
import argparse, collections, datetime, html, json, math, os, re, time, urllib.request, webbrowser
from concurrent.futures import ThreadPoolExecutor

import feedparser
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# lean: -2 left ... 0 centre ... +2 right. ROUGH, EDITABLE estimates informed by public
# raters (AllSides, Media Bias/Fact Check, Ad Fontes). Adjust to your own judgement.
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

TOPICS = {
    "Politics": "election president minister government parliament congress senate vote party trump biden "
                "policy law court supreme ruling campaign tariff sanctions diplomat asean beijing coalition",
    "Conflict & Security": "war attack strike military troops missile gaza ukraine russia israel iran army "
                           "ceasefire hostage terror killed bomb drone",
    "Economy & Business": "economy market inflation stocks bank trade jobs rate company ceo profit "
                          "recession oil prices growth debt",
    "Health": "health virus vaccine hospital disease cancer drug outbreak medical covid doctors",
    "Climate & Science": "climate warming emissions science space nasa energy storm flood wildfire "
                         "heatwave species research study typhoon earthquake cyclone reef",
    "Tech": "tech ai artificial intelligence apple google microsoft meta openai cyber data chip "
            "software robot",
    "Crime & Justice": "police arrest murder trial jailed prison charged shooting suspect investigation",
    "Society & Culture": "migrants immigration protest rights school film music festival culture "
                         "religion church royal celebrity sport football",
}
TOPIC_SETS = {k: set(v.split()) for k, v in TOPICS.items()}
UA = "Mozilla/5.0 (NewsLens research tool)"
vader = SentimentIntensityAnalyzer()


def bucket(o):
    if o.get("state"):
        return "State-run"
    return "Left" if o["lean"] <= -0.7 else "Right" if o["lean"] >= 0.7 else "Centre"


def clean(t):
    return html.unescape(re.sub(r"<[^>]+>", " ", t or "")).strip()


def fetch(o, n):
    try:
        feed = feedparser.parse(o["url"], agent=UA)
        items = []
        for e in feed.entries[:n]:
            title = re.sub(r"\s+-\s+[^-]+$", "", clean(e.get("title"))) if "news.google" in o["url"] else clean(e.get("title"))
            summ = clean(e.get("summary"))[:300]
            if title:
                items.append(dict(outlet=o["name"], region=o["region"], bucket=bucket(o), title=title, summary=summ,
                                  link=e.get("link", "#")))
        return items
    except Exception as ex:  # one broken feed must not stop the run
        print(f"  ! {o['name']}: {ex}")
        return []


def topic_of(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    scores = {k: len(words & v) for k, v in TOPIC_SETS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "Other"


def cluster(items, threshold):
    texts = [f"{i['title']} {i['title']} {i['summary']}" for i in items]
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=threshold).fit_predict(X.toarray())
    groups = collections.defaultdict(list)
    for item, lab in zip(items, labels):
        groups[lab].append(item)
    return list(groups.values())


def distinctive_words(items, a="Left", b="Right", k=12):
    stop = ENGLISH_STOP_WORDS | {"says", "say", "new", "news", "live", "year", "amid", "after"}
    cnt = {a: collections.Counter(), b: collections.Counter()}
    for i in items:
        if i["bucket"] in cnt:
            cnt[i["bucket"]].update({w for w in re.findall(r"[a-z]{4,}", i["title"].lower()) if w not in stop})
    na, nb = sum(cnt[a].values()) + 1, sum(cnt[b].values()) + 1
    vocab = {w for w in set(cnt[a]) | set(cnt[b]) if cnt[a][w] + cnt[b][w] >= 3}
    score = {w: math.log((cnt[a][w] + .5) / na) - math.log((cnt[b][w] + .5) / nb) for w in vocab}
    ranked = sorted(score, key=score.get)
    return ranked[-k:][::-1], ranked[:k]


def ai_framing(story):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ""
    lines = "\n".join(f"[{i['bucket']}] {i['outlet']}: {i['title']} - {i['summary']}" for i in story)
    body = json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 300, "messages": [{"role": "user", "content":
        "Below are headlines about the same story from outlets across the political spectrum. In 3 short "
        "sentences, describe (1) the facts they agree on and (2) how framing, emphasis or omissions differ "
        "by group. Be even-handed and describe only what the text shows.\n\n" + lines}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["content"][0]["text"]
    except Exception as ex:
        return f"(AI summary unavailable: {ex})"


def esc(s):
    return html.escape(str(s))


def render(outlets, items, stories, lean_words, ai, refresh=0):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    nxt = f" Auto-refreshes hourly; last data pull {stamp}." if refresh else f" Data pulled {stamp}."
    by_outlet = collections.defaultdict(list)
    for i in items:
        by_outlet[i["outlet"]].append(i)
    topics = list(TOPICS) + ["Other"]

    heat = "<tr><th></th>" + "".join(f"<th>{esc(t)}</th>" for t in topics) + "</tr>"
    for o in outlets:
        its = by_outlet.get(o["name"], [])
        c = collections.Counter(i["topic"] for i in its)
        cells = ""
        for t in topics:
            share = c[t] / len(its) if its else 0
            cells += f'<td style="background:rgba(55,138,221,{min(share * 2.2, .85):.2f})">{share:.0%}</td>'
        heat += f"<tr><th>{esc(o['name'])}</th>{cells}</tr>"

    spec = ""
    for o in sorted(outlets, key=lambda o: o["lean"]):
        its = by_outlet.get(o["name"], [])
        tone = np.mean([i["tone"] for i in its]) if its else 0
        pos = (o["lean"] + 2) / 4 * 100
        flag = " <span class='tag'>state-run</span>" if o.get("state") else ""
        spec += (f"<tr><td>{esc(o['name'])}{flag}<br><small>{esc(o['region'])} · {esc(o['country'])}</small></td>"
                 f"<td><div class='track'><i style='left:{pos:.0f}%'></i></div></td>"
                 f"<td class='num'>{tone:+.2f}</td><td class='num'>{len(its)}</td></tr>")

    regions = []
    for o in outlets:
        if o["region"] not in regions:
            regions.append(o["region"])
    reg = "<tr><th>Region</th>" + "".join(f"<th>{esc(t)}</th>" for t in topics) + "<th class='num'>Avg tone</th></tr>"
    for r in regions:
        its = [i for o in outlets if o["region"] == r for i in by_outlet.get(o["name"], [])]
        c = collections.Counter(i["topic"] for i in its)
        cells = "".join(
            f'<td style="background:rgba(55,138,221,{min((c[t] / len(its) if its else 0) * 2.2, .85):.2f})">'
            f'{(c[t] / len(its) if its else 0):.0%}</td>' for t in topics)
        tone = np.mean([i["tone"] for i in its]) if its else 0
        reg += f"<tr><th>{esc(r)}</th>{cells}<td class='num'>{tone:+.2f}</td></tr>"

    cards = ""
    for s, summary in stories:
        cols = ""
        for b in ("Left", "Centre", "Right", "State-run"):
            rows = [i for i in s if i["bucket"] == b]
            if not rows:
                continue
            li = "".join(f"<li><a href='{esc(i['link'])}' target='_blank' rel='noopener'><b>{esc(i['outlet'])}</b> <small>({esc(i['region'])})</small>: "
                         f"{esc(i['title'])}</a> <span class='tone {'pos' if i['tone'] > .05 else 'neg' if i['tone'] < -.05 else ''}'>"
                         f"{i['tone']:+.2f}</span></li>" for i in rows)
            cols += f"<div class='col'><h4>{b}</h4><ul>{li}</ul></div>"
        n_out = len({i['outlet'] for i in s})
        title = min(s, key=lambda i: len(i["title"]))["title"]
        ai_html = f"<p class='ai'>{esc(summary)}</p>" if summary else ""
        cards += (f"<section class='card'><h3>{esc(title)}</h3><small>{n_out} outlets · "
                  f"{esc(collections.Counter(i['topic'] for i in s).most_common(1)[0][0])}</small>{ai_html}"
                  f"<div class='cols'>{cols}</div></section>")

    left_w, right_w = lean_words
    chips = lambda ws: "".join(f"<span class='chip'>{esc(w)}</span>" for w in ws)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{meta}<title>NewsLens report</title><style>
:root{{--bg:#fff;--fg:#1c1c1a;--mut:#6b6a65;--line:#dddbd3;--card:#f7f6f2}}
@media(prefers-color-scheme:dark){{:root{{--bg:#1c1c1a;--fg:#eceae3;--mut:#a09f98;--line:#3a3935;--card:#262523}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin:0 auto;padding:24px}}
h1{{margin:0}}h2{{margin-top:40px;border-bottom:1px solid var(--line);padding-bottom:6px}}small,.mut{{color:var(--mut)}}
.wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;font-weight:normal}}
tr>th:first-child{{font-weight:600;white-space:nowrap}}thead th,tr:first-child th{{font-size:12px;color:var(--mut)}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}.track{{position:relative;height:6px;background:linear-gradient(90deg,#378add,#b4b2a9,#d85a30);border-radius:3px;min-width:220px}}
.track i{{position:absolute;top:-5px;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:var(--fg);border:2px solid var(--bg)}}
.tag{{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--mut)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}}.card h3{{margin:0 0 2px}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:10px}}.col h4{{margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}}
.col ul{{margin:0;padding-left:16px}}.col li{{margin-bottom:6px}}a{{color:inherit}}.tone{{font-size:12px;color:var(--mut)}}.tone.pos{{color:#3b8a2e}}.tone.neg{{color:#c0392b}}
.ai{{background:var(--bg);border-left:3px solid #378add;padding:8px 10px;margin:10px 0 0}}.chip{{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:1px 10px;margin:2px}}
</style></head><body>
<h1>NewsLens</h1><p class="mut">{len(items)} headlines from {len(outlets)} outlets. Lean ratings are rough, editable estimates, and
headline tone is a lexicon score (-1 negative, +1 positive), so treat everything here as indicative, not definitive.{nxt}</p>
<h2>Political spectrum and tone</h2><div class="wrap"><table><tr><th>Outlet</th><th>Left &rarr; Right</th><th class="num">Avg tone</th><th class="num">Items</th></tr>{spec}</table></div>
<h2>Coverage by region</h2><div class="wrap"><table>{reg}</table></div>
<h2>What each outlet covers</h2><div class="wrap"><table>{heat}</table></div>
<h2>Same story, different framing</h2><p class="mut">Stories covered by 3+ outlets across at least two groups.</p>{cards or "<p>No shared stories found. Try a lower --threshold.</p>"}
<h2>Words that differ by lean</h2><p><b>Over-used by left-leaning outlets:</b> {chips(left_w)}</p><p><b>Over-used by right-leaning outlets:</b> {chips(right_w)}</p>
</body></html>"""


def run_once(a, refresh=0):
    print(f"[{datetime.datetime.now():%H:%M:%S}] Fetching feeds from 20 outlets across 5 regions...")
    with ThreadPoolExecutor(10) as ex:
        results = list(ex.map(lambda o: fetch(o, a.per_outlet), OUTLETS))
    for o, r in zip(OUTLETS, results):
        print(f"  {o['name']:28s} {len(r)} items")
    items = [i for r in results for i in r]
    if len(items) < 10:
        raise RuntimeError("Too few articles fetched - check your internet connection.")

    for i in items:
        i["topic"] = topic_of(i["title"] + " " + i["summary"])
        i["tone"] = vader.polarity_scores(i["title"])["compound"]

    print("Clustering stories...")
    shared = [g for g in cluster(items, a.threshold)
              if len({i["outlet"] for i in g}) >= 3 and len({i["bucket"] for i in g}) >= 2]
    shared.sort(key=lambda g: (-len({i["bucket"] for i in g}), -len({i["outlet"] for i in g})))
    stories = [(g, ai_framing(g)) for g in shared[:a.stories]]

    tmp = a.out + ".tmp"  # write then swap, so a browser reload never sees a half-written file
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(render(OUTLETS, items, stories, distinctive_words(items), None, refresh))
    os.replace(tmp, a.out)
    print(f"Updated -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-outlet", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.72, help="lower = stricter story matching")
    ap.add_argument("--stories", type=int, default=12)
    ap.add_argument("--interval", type=float, default=60, help="minutes between automatic updates")
    ap.add_argument("--once", action="store_true", help="run a single time instead of looping")
    ap.add_argument("--out", default="newslens_report.html")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    refresh = 0 if a.once else int(min(a.interval * 60, 300))  # open page re-reads the file every <=5 min
    opened = False
    while True:
        started = time.time()
        try:
            run_once(a, refresh)
            if not a.no_open and not opened:
                webbrowser.open("file://" + os.path.abspath(a.out))
                opened = True
        except KeyboardInterrupt:
            break
        except Exception as ex:  # keep the service alive and retry next cycle
            print(f"Update failed: {ex}")
        if a.once:
            break
        wait = max(60, a.interval * 60 - (time.time() - started))
        print(f"Next update at {datetime.datetime.now() + datetime.timedelta(seconds=wait):%H:%M} (Ctrl+C to stop)\n")
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            break
    print("Stopped.")


if __name__ == "__main__":
    main()
