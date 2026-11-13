#!/usr/bin/env python3
"""Sightwire site builder: pulls 20 outlets, groups the same story, writes a static news site.
Needs template.html beside it. Optional: ANTHROPIC_API_KEY for neutral AI summaries."""
import argparse, collections, datetime, hashlib, html, json, math, os, re, socket, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
import feedparser, numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
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

CATS = ["World", "Politics", "Business", "Health", "Science & Tech", "Climate", "Sport", "Culture"]
KW = {
 "Politics": "election elections president minister government parliament congress senate vote votes voters party policy lawmakers campaign coalition cabinet legislation opposition referendum downing-street white-house chancellor",
 "Business": "economy economic market markets inflation stocks shares bank banks trade jobs unemployment interest company profit profits recession oil prices growth debt tariff tariffs earnings investors budget sales merger pay wages strike union unions",
 "Health": "health nhs hospital hospitals disease cancer drug vaccine outbreak medical doctors nurses patients virus surgery",
 "Science & Tech": "tech technology ai artificial intelligence apple google microsoft meta openai nvidia cyber chip software robot space nasa scientists research satellite",
 "Climate": "climate warming emissions wildfire wildfires flood floods storm hurricane typhoon cyclone drought heatwave carbon renewable biodiversity pollution",
 "Sport": "football soccer cricket tennis olympic olympics league match goal coach tournament championship rugby golf athlete cup premier-league fifa nba nfl",
 "Culture": "film films movie music festival album artist museum theatre award awards book celebrity royal fashion actor actress oscar concert",
}
REG = {
 "UK": "uk britain british england scotland scottish wales welsh westminster nhs starmer holyrood downing-street labour-party farage badenoch",
 "US": "usa american americans america washington congress senate white-house pentagon gop republican republicans democrat democrats trump vance biden harris supreme-court",
 "Americas": "canada canadian mexico mexican brazil argentina chile colombia venezuela peru",
 "Europe": "france french germany german paris berlin italy spain ukraine russia poland brussels",
 "Southeast Asia": "singapore thailand thai bangkok indonesia jakarta philippines manila vietnam malaysia myanmar cambodia asean",
 "Pacific": "australia australian sydney fiji samoa tonga pacific papua canberra new-zealand",
 "Middle East & Africa": "israel gaza iran iraq syria saudi qatar egypt turkey africa nigeria kenya sudan lebanon yemen",
 "East & South Asia": "china chinese beijing japan japanese korea india indian pakistan taiwan",
}
COUNTRIES = {"UK": REG["UK"], "US": REG["US"], "Canada": "canada canadian ottawa trudeau carney", "Mexico": "mexico mexican sheinbaum",
 "Brazil": "brazil brazilian lula brasilia", "Argentina": "argentina argentine milei", "France": "france french paris macron",
 "Germany": "germany german berlin merz scholz", "Italy": "italy italian rome meloni", "Spain": "spain spanish madrid",
 "Ukraine": "ukraine ukrainian kyiv zelensky", "Russia": "russia russian moscow putin kremlin", "Poland": "poland polish warsaw",
 "Israel": "israel israeli netanyahu knesset", "Palestine": "gaza palestinian palestinians hamas west-bank", "Iran": "iran iranian tehran",
 "Turkey": "turkey turkish ankara erdogan", "Saudi Arabia": "saudi-arabia saudi riyadh", "Egypt": "egypt egyptian cairo",
 "South Africa": "south-africa pretoria ramaphosa", "Nigeria": "nigeria nigerian abuja", "Kenya": "kenya kenyan nairobi",
 "China": "china chinese beijing", "Japan": "japan japanese tokyo", "South Korea": "south-korea seoul", "North Korea": "north-korea pyongyang",
 "India": "india indian delhi modi", "Pakistan": "pakistan pakistani islamabad", "Taiwan": "taiwan taiwanese taipei", "Hong Kong": "hong-kong",
 "Australia": "australia australian canberra albanese", "New Zealand": "new-zealand luxon wellington", "Indonesia": "indonesia indonesian jakarta prabowo",
 "Philippines": "philippines philippine manila marcos", "Thailand": "thailand thai bangkok", "Singapore": "singapore singaporean",
 "Vietnam": "vietnam vietnamese hanoi", "Malaysia": "malaysia malaysian kuala-lumpur anwar", "Myanmar": "myanmar burma"}
BYNAME = {o["name"]: o for o in OUTLETS}
BBC, GU, NYT, FOX, CNNF = "http://feeds.bbci.co.uk/", "https://www.theguardian.com/", "https://rss.nytimes.com/services/xml/rss/nyt/", "https://moxie.foxnews.com/google-publisher/", "http://rss.cnn.com/rss/"
# (outlet, feed url, region tag, category hint): the UK and US sections of these outlets drive the UK and US tabs
EXTRA = [("BBC News", BBC + "news/uk/rss.xml", "UK", None), ("BBC News", BBC + "news/politics/rss.xml", "UK", "Politics"),
 ("The Guardian", GU + "uk/rss", "UK", None), ("The Guardian", GU + "politics/rss", "UK", "Politics"),
 ("Sky News", "https://feeds.skynews.com/feeds/rss/uk.xml", "UK", None), ("Sky News", "https://feeds.skynews.com/feeds/rss/politics.xml", "UK", "Politics"),
 ("The Independent", "https://www.independent.co.uk/news/uk/rss", "UK", None), ("The Telegraph", "https://www.telegraph.co.uk/news/rss.xml", "UK", None),
 ("CNN", CNNF + "cnn_us.rss", "US", None), ("CNN", CNNF + "cnn_allpolitics.rss", "US", "Politics"),
 ("New York Times", NYT + "US.xml", "US", None), ("New York Times", NYT + "Politics.xml", "US", "Politics"),
 ("Fox News", FOX + "us.xml", "US", None), ("Fox News", FOX + "politics.xml", "US", "Politics"),
 ("NPR", "https://feeds.npr.org/1014/rss.xml", "US", "Politics"), ("Washington Post", "https://feeds.washingtonpost.com/rss/national", "US", None),
 ("BBC News", BBC + "sport/rss.xml", None, "Sport"), ("BBC News", BBC + "news/entertainment_and_arts/rss.xml", None, "Culture"),
 ("BBC News", BBC + "news/science_and_environment/rss.xml", None, "Environment"), ("BBC News", BBC + "news/health/rss.xml", None, "Health"),
 ("BBC News", BBC + "news/technology/rss.xml", None, "Tech"), ("BBC News", BBC + "news/business/rss.xml", None, "Business"),
 ("The Guardian", GU + "sport/rss", None, "Sport"), ("The Guardian", GU + "culture/rss", None, "Culture"),
 ("The Guardian", GU + "environment/rss", None, "Environment"), ("The Guardian", GU + "technology/rss", None, "Tech"),
 ("The Guardian", GU + "business/rss", None, "Business"), ("New York Times", NYT + "Sports.xml", None, "Sport"),
 ("New York Times", NYT + "Arts.xml", None, "Culture"), ("New York Times", NYT + "Science.xml", None, "Environment"),
 ("New York Times", NYT + "Health.xml", None, "Health"), ("New York Times", NYT + "Technology.xml", None, "Tech"),
 ("New York Times", NYT + "Business.xml", None, "Business"), ("Sky News", "https://feeds.skynews.com/feeds/rss/business.xml", None, "Business"),
 ("Sky News", "https://feeds.skynews.com/feeds/rss/technology.xml", None, "Tech"), ("Sky News", "https://feeds.skynews.com/feeds/rss/entertainment.xml", None, "Culture"),
 ("CNN", CNNF + "edition_sport.rss", None, "Sport"), ("CNN", CNNF + "edition_technology.rss", None, "Tech"), ("CNN", CNNF + "edition_entertainment.rss", None, "Culture"),
 ("Fox News", FOX + "sports.xml", None, "Sport"), ("Fox News", FOX + "tech.xml", None, "Tech"), ("Fox News", FOX + "health.xml", None, "Health"),
 ("Fox News", FOX + "science.xml", None, "Environment"), ("NPR", "https://feeds.npr.org/1019/rss.xml", None, "Tech"),
 ("NPR", "https://feeds.npr.org/1007/rss.xml", None, "Environment"), ("NPR", "https://feeds.npr.org/1128/rss.xml", None, "Health"),
 ("NPR", "https://feeds.npr.org/1006/rss.xml", None, "Business"), ("NPR", "https://feeds.npr.org/1008/rss.xml", None, "Culture")]
HINT = {"Environment": "Climate", "Tech": "Science & Tech"}
EXTRA = [(n, u, t, "Science & Tech" if c == "Environment" and any(k in u.lower() for k in ("science", "1007")) else HINT.get(c, c)) for n, u, t, c in EXTRA]
SUBJECTS = {"health": "nhs hospital hospitals doctors nurses patients", "prisons": "prison prisons prisoner prisoners inmates", "police": "police policing constable constables",
 "education": "teachers teacher schools school pupils university students", "rail": "rail railway railways train trains", "military": "army troops soldiers military navy",
 "courts": "court judge jury trial verdict", "migration": "migrants asylum immigration border", "energy": "energy gas electricity oil power",
 "banking": "bank banks lender mortgage", "farming": "farmers farming agriculture", "airlines": "airline airlines flights airport"}
SUBJ_IDS = {k: n for n, k in enumerate(SUBJECTS)}
LOADED = set("horror horrifying devastating brutal stunning explosive savage fury fumes erupts rages rips torches mocks panic terrifying tragic heartbreaking unbelievable chilling shocked slammed blasted disgraceful despicable epic slams blasts destroys shocking outrage furious chaos disaster radical extremist crushes savages humiliates bombshell meltdown scandal rampage".split())
HEDGE = ("reportedly", "allegedly", "claims", "sources say", "unconfirmed", "rumour", "rumor", "alleged")
TIER1 = {"Reuters", "Associated Press", "BBC News", "The Guardian", "New York Times", "Washington Post", "Wall Street Journal", "CNN", "Fox News",
 "NBC News", "CBS News", "ABC News (US)", "NPR", "Politico", "USA Today", "Al Jazeera", "Deutsche Welle", "France 24", "Sky News", "ITV News",
 "Channel 4 News", "The Telegraph", "The Times", "The Independent", "Daily Mail", "Bloomberg", "ABC News (Australia)", "CBC News"}


def majors(g):
    return len({i["outlet"] for i in g} & TIER1)


def qualifies(g):
    m, n = majors(g), len({i["outlet"] for i in g})
    return m >= 2 or (n >= 3 and m >= 1)


CKEY = {o["name"]: ("Global" if o["country"] == "UK/Global" else o["country"]) for o in OUTLETS}
_CN = collections.Counter(CKEY.values())
WEIGHT = {n: (1.0 if n in TIER1 else .4) / math.sqrt(_CN[c]) for n, c in CKEY.items()}
WIRE = ("Reuters", "BBC News", "Deutsche Welle")
vader = SentimentIntensityAnalyzer()


def bucket(o):
    return "State-run" if o.get("state") else "Left" if o["lean"] <= -0.7 else "Right" if o["lean"] >= 0.7 else "Centre"


def clean(t):
    return html.unescape(re.sub(r"<[^>]+>", " ", t or "")).strip()


def toks(t):
    t = t.lower().replace("new zealand", "new-zealand").replace("u.s.", "usa").replace("white house", "white-house").replace("downing street", "downing-street").replace("labour party", "labour-party").replace("new england", "new-england").replace("south africa", "south-africa").replace("saudi arabia", "saudi-arabia").replace("south korea", "south-korea").replace("north korea", "north-korea").replace("hong kong", "hong-kong").replace("west bank", "west-bank").replace("kuala lumpur", "kuala-lumpur").replace("supreme court", "supreme-court")
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
        for e in feedparser.parse(o["url"], agent="Mozilla/5.0 (Sightwire)").entries[:n]:
            t = clean(e.get("title"))
            if "news.google" in o["url"]:
                t = re.sub(r"\s+-\s+[^-]+$", "", t)
            if t:
                out.append(dict(outlet=o["name"], region=o["region"], bucket=bucket(o), title=t,
                                excerpt=clean(e.get("summary"))[:240], link=e.get("link", "#"), img=https(image_of(e)), tag=o.get("_tag"), cat=o.get("_cat")))
        return out
    except Exception as ex:
        print("  !", o["name"], ex)
        return []


def dedupe_items(items):
    seen = {}
    for i in items:
        k = i["link"]
        if k in seen:
            for f in ("tag", "cat", "img", "excerpt"):
                if i.get(f) and not seen[k].get(f):
                    seen[k][f] = i[f]
        else:
            seen[k] = i
    return list(seen.values())


def cat_of(title, excerpt=""):
    tw, ew = toks(title), toks(excerpt)
    sc = {k: 2 * len(tw & set(v.split())) + len(ew & set(v.split())) for k, v in KW.items()}
    b = max(sc, key=sc.get)
    return b if sc[b] >= 2 else None


def category(text):
    return cat_of(text) or "World"


def story_category(g):
    """Each article votes (its section feed, else its own wording); the story takes the clear majority, otherwise World."""
    votes = collections.Counter(c for c in (i.get("cat") or cat_of(i["title"], i["excerpt"]) for i in g) if c)
    if votes:
        best, n = votes.most_common(1)[0]
        if n / sum(votes.values()) >= .5 and n >= max(1, .2 * len(g)):
            return best
    return "World"


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


def facts_fallback(items, k=5):
    """Key points without AI: the sentences most outlets back, from snippets and major outlets' clean headlines."""
    cands = {}
    for i in items:
        for sn in re.split(r"(?<=[.!?])\s+", i["excerpt"]):
            if len(sn) > 30:
                cands.setdefault(sn, toks(sn) - STOP)
        if i["outlet"] in TIER1 and not is_loaded(i["title"]):
            cands.setdefault(i["title"], toks(i["title"]) - STOP)
    scored = []
    for sn, w in cands.items():
        if w:
            sup = {i["outlet"] for i in items if len(w & toks(i["title"] + " " + i["excerpt"])) / len(w) >= .5}
            scored.append((len(sup), len(sn), sn, w, sorted(sup)))
    out = []
    for n, _, sn, w, by in sorted(scored, key=lambda x: (-x[0], -x[1])):
        if all(len(w & o["_w"]) / max(1, len(w | o["_w"])) < .5 for o in out):
            out.append(dict(t=sn, n=n, by=by, _w=w))
        if len(out) == k:
            break
    for o in out:
        o.pop("_w")
    return out


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
            ents = feedparser.parse(url, agent="Mozilla/5.0 (Sightwire)").entries[:3]
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


UA_B = {"User-Agent": "Mozilla/5.0 (compatible; Sightwire/1.0)"}


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


def story_regions(g, text):
    places, out = story_places(g), []
    for r, kw in REG.items():
        if r in ("UK", "US"):
            if r in places:
                out.append(r)
        elif sum(bool(toks(i["title"] + " " + i["excerpt"]) & set(kw.split())) for i in g) / len(g) >= .3:
            out.append(r)
    return out


def item_subject(i):
    w = toks(i["title"] + " " + i["excerpt"])
    sc = {k: len(w & set(v.split())) for k, v in SUBJECTS.items()}
    b = max(sc, key=sc.get)
    return b if sc[b] and sorted(sc.values())[-2] < sc[b] else None


def split_subjects(idx_groups, items):
    """If one cluster mixes stories about clearly different subjects (e.g. prison staff vs NHS staff), split it."""
    out = []
    for ix in idx_groups:
        subj = {n: item_subject(items[n]) for n in ix}
        cnt = collections.Counter(v for v in subj.values() if v)
        big = [k for k, c in cnt.items() if c >= 2]
        if len(ix) < 4 or len(big) < 2:
            out.append(ix)
            continue
        main, parts = cnt.most_common(1)[0][0], collections.defaultdict(list)
        for n in ix:
            parts[subj[n] if subj[n] in big else main].append(n)
        out.extend(parts.values())
    return out


def group_subject(ix, items):
    cnt = collections.Counter(item_subject(items[n]) for n in ix)
    cnt.pop(None, None)
    return cnt.most_common(1)[0][0] if cnt and cnt.most_common(1)[0][1] >= .4 * len(ix) else None


def top_terms(ix, items, k=6):
    c = collections.Counter()
    for n in ix:
        c.update(toks(items[n]["title"] + " " + items[n]["excerpt"]) - STOP - LOADED)
    return {w for w, _ in c.most_common(k) if len(w) > 3}


def story_places(g):
    """Countries a story is mainly about. UK/US need their own news-section feeds or to be the clearest subject (so a Canadian story is not tagged UK)."""
    sh = {n: sum(bool(toks(i["title"] + " " + i["excerpt"]) & set(kw.split())) for i in g) / len(g) for n, kw in COUNTRIES.items()}
    top, tags, out = max(sh.values()), collections.Counter(i.get("tag") for i in g), []
    for n, v in sh.items():
        if n in ("UK", "US"):
            if tags[n] / len(g) >= .3 or (v >= .4 and v >= top):
                out.append(n)
        elif v >= .3:
            out.append(n)
    return out


def dedupe(idx_groups, X, items, subj, merge, related):
    """Merge clusters that are the same event under different headlines; link clusters that are follow-ups."""
    C = normalize(sp.vstack([sp.csr_matrix(X[ix].sum(axis=0)) for ix in idx_groups]))
    S = (C @ C.T).toarray()
    np.fill_diagonal(S, 0)
    S = np.triu(S)
    terms = [top_terms(ix, items) if len(ix) >= 2 else set() for ix in idx_groups]  # second check: shared key terms
    big = [k for k in range(len(idx_groups)) if len(terms[k]) >= 3]
    for x in range(len(big)):
        for y in range(x + 1, len(big)):
            a_, b_ = big[x], big[y]
            shared = len(terms[a_] & terms[b_])
            ov = shared / min(len(terms[a_]), len(terms[b_]))
            if shared >= 2 and ov >= .4:
                S[a_, b_] = max(S[a_, b_], merge if ov >= .6 else related)
    codes = np.array([SUBJ_IDS.get(sb, -1) if sb else -1 for sb in subj])
    S[(codes[:, None] >= 0) & (codes[None, :] >= 0) & (codes[:, None] != codes[None, :])] = 0
    parent = list(range(len(idx_groups)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_, b_ in zip(*np.where(S >= merge)):
        parent[find(a_)] = find(b_)
    comp = collections.defaultdict(list)
    for k in range(len(idx_groups)):
        comp[find(k)].append(k)
    merged, owner = [], {}
    for members in comp.values():
        for k in members:
            owner[k] = len(merged)
        merged.append([n for k in members for n in idx_groups[k]])
    links = collections.defaultdict(dict)
    for a_, b_ in zip(*np.where(S >= related)):
        oa, ob = owner[a_], owner[b_]
        if oa != ob:
            links[oa][ob] = links[ob][oa] = max(links[oa].get(ob, 0), float(S[a_, b_]))
    return merged, links


def llm(items):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    lines = "\n".join(f"[{i['outlet']} | {i['bucket']}] {i['title']} - {i['excerpt']}" for i in sample(items))
    prompt = ("Headlines and snippets about ONE news story from outlets across the political spectrum. Return ONLY JSON: "
              '{"title": a calm, general overview of what is happening in under 12 words: describe events, no adjectives or emotive verbs, no quotes, no questions, no blame or motives, nothing stated as fact that is only claimed, "summary": 3-4 specific sentences saying what happened, who is involved, where and when, with key numbers and why it matters - no vague wording, no opinion, '
              '"facts": [up to 6 items like {"text": one concrete key point with names, numbers, places or dates, "outlets": [names of outlets whose text supports it]}], '
              '"unconfirmed": [claims only one outlet makes or that are hedged], "framing_differs": true only if left/centre/right '
              'outlets clearly frame it differently, "framing_note": one sentence or "", "per_outlet": {outlet name: one sentence on what that outlet text reports}}. Use only the text given.\n\n' + lines)
    body = json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 2200, "messages": [{"role": "user", "content": prompt}]}).encode()
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
        for f in ai.get("facts", [])[:6]:
            by = sorted(set(f.get("outlets", [])) & outs)
            facts.append(dict(t=f["text"], n=len(by), by=by))
        facts += [dict(t=u, n=1, by=[]) for u in ai.get("unconfirmed", [])[:2]]
        summary = ai.get("summary", "")
    else:
        facts = facts_fallback(g)
        summary = " ".join(f["t"] if f["t"][-1] in ".!?" else f["t"] + "." for f in facts[:2]) or f"Reported by {len(outs)} outlets."
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
    cat = story_category(g)
    reg = story_regions(g, text) or ["Global"]
    return dict(id=hashlib.sha1(min(i["link"] for i in g).encode()).hexdigest()[:8], title=title, summary=summary,
                facts=facts, cat=cat, regions=reg, outlets=len(outs), limited=len(outs) < 3,
                imgs=list(dict.fromkeys(i["img"] for i in sorted(g, key=lambda i: i["outlet"] not in WIRE) if i["img"]))[:4],
                items=[{k: i[k] for k in ("outlet", "bucket", "region", "title", "excerpt", "summary", "link", "img", "tone", "flags")} for i in g],
                places=story_places(g), gs=round(sum(WEIGHT.get(o, .3) for o in outs), 2), countries=len({CKEY.get(o, o) for o in outs}), majors=len(outs & TIER1),
                analysis=dict(show=bool(clear), tones=tones, words=words, note=(ai or {}).get("framing_note", "")))


def load_prev(a):
    src = a.prev_data
    if not src and os.environ.get("GITHUB_REPOSITORY"):
        owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")
        src = f"https://{owner}.github.io/{repo}/data.json?t={int(time.time())}"
    try:
        if src.startswith("http"):
            return json.load(urllib.request.urlopen(src, timeout=20))
        return json.load(open(src))
    except Exception as ex:
        print("  no previous data (first run?):", ex)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public")
    ap.add_argument("--prev", default="prev_roundup.json")
    ap.add_argument("--prev-data", default="", help="previous data.json (path or URL); defaults to the live site")
    ap.add_argument("--keep-hours", type=float, default=20, help="keep earlier stories this long after they last appeared")
    ap.add_argument("--tz", default="Europe/London", help="time zone for the 6am round-up")
    ap.add_argument("--per-outlet", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--ai-stories", type=int, default=14)
    ap.add_argument("--merge", type=float, default=0.5, help="similarity above which two stories are merged as duplicates")
    ap.add_argument("--related", type=float, default=0.3, help="similarity above which two stories are linked as related")
    ap.add_argument("--min-home", type=int, default=14, help="minimum stories on the homepage")
    ap.add_argument("--backfill-stories", type=int, default=10, help="how many top stories to search extra outlets for")
    ap.add_argument("--once", action="store_true")  # accepted for compatibility
    a = ap.parse_args()
    socket.setdefaulttimeout(20)
    now = datetime.datetime.now(ZoneInfo(a.tz))
    nowiso = now.isoformat(timespec="minutes")
    prev_data = load_prev(a)
    if prev_data.get("v") != 3:
        prev_data = {}  # categories changed: rebuild everything once

    jobs = list(OUTLETS) + [dict(BYNAME[n], url=u, _tag=t, _cat=c) for n, u, t, c in EXTRA if n in BYNAME]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda o: fetch(o, 80 if o['name'] in TIER1 else a.per_outlet), jobs))
    for o, r in zip(OUTLETS, res):
        print(f"  {o['name']:28s} {len(r)}")
    dead = [f"{o['name']} ({o['url'].split('/')[-1] or o['url'][-20:]})" for o, r in zip(jobs[len(OUTLETS):], res[len(OUTLETS):]) if not r]
    print(f"  section feeds: {sum(len(r) for r in res[len(OUTLETS):])} items from {len(EXTRA) - len(dead)}/{len(EXTRA)} working feeds" + (f"; empty: {', '.join(dead)}" if dead else ""))
    items = dedupe_items([i for r in res for i in r])
    if len(items) < 20:
        raise SystemExit("Too few articles fetched.")
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [f"{i['title']} {i['title']} {i['excerpt']}" for i in items])
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=a.threshold).fit_predict(X.toarray())
    idx = collections.defaultdict(list)
    for n, l in enumerate(labels):
        idx[l].append(n)
    gidx = split_subjects(list(idx.values()), items)
    merged, links = dedupe(gidx, X, items, [group_subject(ix, items) for ix in gidx], a.merge, a.related)
    print(f"  {len(gidx)} clusters merged into {len(merged)} stories after duplicate checks; {len(links)} have related follow-ups")
    cl = [[items[n] for n in ix] for ix in merged]
    rel = {id(cl[k]): [(cl[j], sim) for j, sim in lk.items()] for k, lk in links.items()}
    cl = sorted(cl, key=lambda g: (-len({i["outlet"] for i in g}), -len({i["bucket"] for i in g})))
    multi = [g for g in cl if qualifies(g)][:80]
    backfill(multi[:a.backfill_stories])
    multi.sort(key=lambda g: (-len({i["outlet"] for i in g}), -len({i["bucket"] for i in g})))
    if len(multi) < a.min_home:  # quiet news day: top up with the widest 2-outlet stories
        multi += [g for g in cl if majors(g) >= 1 and len({i["outlet"] for i in g}) == 2 and not any(g is m for m in multi)][: a.min_home - len(multi)]
    hidden, per = [], collections.Counter()
    for g in cl:  # stories few outlets cover: shown only under "Hidden stories" (or when a section is thin)
        if len(g[0]["title"]) > 25 and not any(g is m for m in multi):
            key = (category(g[0]["title"] + " " + g[0]["excerpt"]), g[0].get("tag") or "")
            if per[key] < 6:
                per[key] += 1
                hidden.append(g)
    print(f"  {len(multi)} main stories, {len(hidden)} hidden stories")

    # ---- incremental update: match today's stories to the previous build by shared article links
    pool, hid_ids = multi + hidden, {id(g) for g in hidden}
    old = prev_data.get("stories", [])
    old_by_link = {i["link"]: k for k, o in enumerate(old) for i in o["items"]}
    used, match, fresh = set(), {}, {}
    for g in pool:
        votes = collections.Counter(old_by_link[i["link"]] for i in g if i["link"] in old_by_link)
        if votes:
            k, n = votes.most_common(1)[0]
            if k not in used and n >= max(1, .25 * min(len(g), len(old[k]["items"]))):
                used.add(k)
                match[id(g)] = old[k]
    for g in pool:
        o = match.get(id(g))
        if o:
            oldl = {i["link"] for i in o["items"]}
            fresh[id(g)] = any(i["link"] not in oldl for i in g)
            have = {i["link"] for i in g}
            g += [dict(i) for i in o["items"] if i["link"] not in have]  # keep coverage that has rolled out of the feeds
        else:
            fresh[id(g)] = True
    todo = [g for g in pool if fresh[id(g)]]
    enrich(todo)
    validate_imgs(todo)
    use_ai, ai_n, built = bool(os.environ.get("ANTHROPIC_API_KEY")), 0, []
    for g in pool:
        o = match.get(id(g))
        if o and not fresh[id(g)]:
            st = dict(o)
            st.pop("stale", None)
            st.setdefault("first", nowiso)
            st.setdefault("upd", st["first"])
        else:
            ai_now = use_ai and ai_n < a.ai_stories and id(g) not in hid_ids
            ai_n += ai_now
            st = build_story(g, ai_now, None)
            st["first"] = o.get("first", nowiso) if o else nowiso
            st["upd"] = nowiso
            if o:
                st["id"] = o["id"]  # keep the same web address when a story is updated
        st["seen"], st["rank"] = nowiso, st["gs"]
        if id(g) in hid_ids:
            st["hidden"] = True
        else:
            st.pop("hidden", None)
        built.append((g, st))
    retained = []
    for k, o in enumerate(old):  # earlier stories no longer in the feeds stay for a while, fading down the ranking
        if k in used or not o.get("seen"):
            continue
        try:
            age = (now - datetime.datetime.fromisoformat(o["seen"])).total_seconds() / 3600
        except ValueError:
            continue
        if age < a.keep_hours:
            retained.append(dict(o, stale=True, rank=o["gs"] * 0.5 ** (age / 12)))
    new_n = sum(1 for g, _ in built if id(g) not in match)
    print(f"  update: {new_n} new, {sum(1 for g, _ in built if id(g) in match and fresh[id(g)])} changed, "
          f"{sum(1 for g, _ in built if id(g) in match and not fresh[id(g)])} unchanged, {len(retained)} kept from earlier")

    by_g = {id(g): st for g, st in built}
    for g, st in built:  # follow-ups: separate stories on the same topic reference each other
        base = st.setdefault("sum0", st["summary"])
        pairs = [(r, sim) for r, sim in sorted(rel.get(id(g), []), key=lambda x: -x[1]) if id(r) in by_g and by_g[id(r)]["id"] != st["id"]]
        st["related"] = [dict(id=by_g[id(r)]["id"], title=by_g[id(r)]["title"]) for r, _ in pairs][:3]
        later = [by_g[id(r)]["title"] for r, _ in pairs if by_g[id(r)]["outlets"] < st["outlets"]][:2]
        st["summary"] = (base + " Related development: " + "; ".join(later) + ".") if later else base
    stories = [st for _, st in built] + retained
    main = sorted([x for x in stories if not x.get("hidden")], key=lambda x: -x["rank"])[:120]
    for x in main:  # homepage = stories reported across many countries, weighted so no single country dominates
        x["home"] = x["countries"] >= 4 and x["majors"] >= 3
    short = a.min_home - sum(x["home"] for x in main)
    for x in main:
        if short > 0 and not x["home"]:
            x["home"], short = True, short - 1
    stories = main + [x for x in stories if x.get("hidden")][:80]

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
            for s in main if s["home"]][:10])

    os.makedirs(a.out, exist_ok=True)
    data = dict(v=3, updated=now.strftime("%d %b %Y, %H:%M"), cats=CATS, regions=list(REG) + ["Global"], stories=stories,
                roundup=roundup, outlets=[dict(name=o["name"], bucket=bucket(o), region=o["region"]) for o in OUTLETS], ai=use_ai)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"), encoding="utf-8").read()
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", blob))
    json.dump(roundup, open(os.path.join(a.out, "roundup.json"), "w"))
    json.dump(data, open(os.path.join(a.out, "data.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Built {len(stories)} stories -> {a.out}/index.html")


if __name__ == "__main__":
    main()
