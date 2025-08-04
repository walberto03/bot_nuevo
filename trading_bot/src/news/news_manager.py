import os, json
from datetime import datetime, timedelta
import requests
from trading_bot.config import cfg

class NewsManager:
    def __init__(self):
        self.cache_file = os.path.join(cfg["cache_dir"], "news.json")
        if os.path.exists(self.cache_file):
            self.news_cache = json.load(open(self.cache_file))
        else:
            self.news_cache = {}
    def save(self):
        json.dump(self.news_cache, open(self.cache_file, "w"), indent=2)
    def get_terms(self, symbol):
        base,quote=symbol[:3],symbol[3:]
        return [f"{base}/{quote}", base.lower(), quote.lower()]
    def fetch_chunk(self, term, start, end):
        url="https://newsapi.org/v2/everything"
        params={"q":term,"from":start,"to":end,"apiKey":cfg["news_api_key"],"pageSize":100}
        arts=[]; page=1
        while True:
            params["page"]=page
            r=requests.get(url,params=params).json().get("articles",[])
            if not r: break
            arts+=r; page+=1
        return arts
    def fetch_all_news(self, symbols, full_refresh=False):
        start = datetime.fromisoformat(cfg["date"]["start_cache"]) if full_refresh or not self.news_cache \
                else max(datetime.fromisoformat(k.split("_")[1]) for k in self.news_cache)+timedelta(days=1)
        end = datetime.utcnow()
        cur=start
        while cur<=end:
            end_block = min(cur+timedelta(days=6), end)
            for s in symbols:
                key=f"{s}_{cur.date()}_{end_block.date()}"
                if not full_refresh and key in self.news_cache: continue
                combined={}
                for term in self.get_terms(s):
                    arts=self.fetch_chunk(term,cur.strftime("%Y-%m-%d"),end_block.strftime("%Y-%m-%d"))
                    for a in arts:
                        d=a["publishedAt"][:10]
                        combined.setdefault(d,[]).append({
                            "title":a["title"],"desc":a.get("description",""),"url":a["url"],
                            "source":a["source"]["name"]
                        })
                self.news_cache[key]=combined
            cur=end_block+timedelta(days=1)
        self.save()
    def get_news_for_date(self,symbol,date):
        d=date.strftime("%Y-%m-%d"); out=[]
        for k,v in self.news_cache.items():
            if k.startswith(symbol+"_") and d in v:
                out+=v[d]
        return out
