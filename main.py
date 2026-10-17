from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from jmcomic import JmAlbumDetail, JmOption  # type: ignore

log = logging.getLogger(__name__)

# 作者查询模板 —— 按 jm 站点实际支持的语法调整
# 试过的三种常见写法：
#   'author:"{author}"'  ← 最希望它支持的
#   '"{author}"'         ← 短语匹配
#   '+{author}'          ← 当 tag 一样 AND
AUTHOR_QUERY_TEMPLATE = '"{author}"'


class ComicAdv:
    def __init__(self, max_workers: int = 8) -> None:
        self.file_path = Path(__file__).parent / "memo.json"
        self.fav_tag: Counter[str] = Counter()
        self.fav_author: Counter[str] = Counter()
        self.added: set[str] = set()
        self.tgl: list[str] = []     # 最近一次 tag 组合
        self.agl: list[str] = []     # 最近一次作者组合

        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._local = threading.local()

    # ---------- client ----------
    def _get_client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = JmOption.default().new_jm_client()
            self._local.client = client
        return client

    # ---------- 持久化 ----------
    def load(self) -> None:
        if not self.file_path.exists():
            return
        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.fav_tag = Counter(data.get("fav_tag", {}))
        self.fav_author = Counter(data.get("fav_author", {}))   # 兼容旧数据：缺失时为空
        self.added = set(data.get("added", []))

    def save(self) -> None:
        with self._lock:
            data = {
                "fav_tag": dict(self.fav_tag.most_common()),
                "fav_author": dict(self.fav_author.most_common()),
                "added": sorted(self.added),
            }
        tmp = self.file_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        tmp.replace(self.file_path)

    # ---------- 抓取 ----------
    def _extract_authors(self, album) -> list[str]:
        # 不同 jmcomic 版本属性名不同，同时兼容 authors / author
        for attr in ("authors", "author"):
            val = getattr(album, attr, None)
            if not val:
                continue
            if isinstance(val, str):
                return [val]
            return [str(x) for x in val]
        return []

    def _fetch_meta(self, album_id: str) -> tuple[list[str], list[str]] | None:
        try:
            album: JmAlbumDetail = self._get_client().get_album_detail(album_id)
            tags = list(album.tags or [])
            authors = self._extract_authors(album)
            return tags, authors
        except Exception as e:  # noqa: BLE001
            log.warning("fetch %s failed: %r", album_id, e)
            return None

    def add(self, inpt: Iterable[str]) -> None:
        pending = [i for i in dict.fromkeys(inpt) if i not in self.added]
        if not pending:
            return

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._fetch_meta, i): i for i in pending}
            for fut in as_completed(futures):
                album_id = futures[fut]
                meta = fut.result()
                if meta is None:
                    continue
                tags, authors = meta
                with self._lock:
                    self.added.add(album_id)
                    self.fav_tag.update(tags)
                    self.fav_author.update(authors)
                log.info("[ok] %s -> %d tags, %d authors", album_id, len(tags), len(authors))

    # ---------- 查询构造 ----------
    def _build_query(self, tags: Iterable[str] = (), authors: Iterable[str] = ()) -> str:
        parts: list[str] = []
        for a in authors:
            parts.append(AUTHOR_QUERY_TEMPLATE.format(author=a))
        for t in tags:
            parts.append(f"+{t}")
        return " ".join(parts)

    def _get_comic(
        self,
        tags: Iterable[str] = (),
        authors: Iterable[str] = (),
        page: int = 1,
    ) -> list[str]:
        tags = list(tags)
        authors = list(authors)
        if not tags and not authors:
            return []
        query = self._build_query(tags, authors)
        try:
            page_obj = self._get_client().search_site(search_query=query, page=page)
        except Exception as e:  # noqa: BLE001
            log.warning("search %r failed: %r", query, e)
            return []

        log.info(
            "query=%r total=%d page_size=%d page_count=%d",
            query, page_obj.total, page_obj.page_size, page_obj.page_count,
        )
        return [f"[{aid}]: {title}" for aid, title in page_obj]

    # ---------- 搜索方式 1：按 tag 渐进 ----------
    def ad(self) -> dict[str, list[str]]:
        """[t1] -> [t1,t2] -> [t1,t2,t3] ...，无结果即停。"""
        self.tgl = []
        results: dict[str, list[str]] = {}
        for tag, _ in self.fav_tag.most_common():
            trial = [*self.tgl, tag]
            comics = self._get_comic(tags=trial)
            if not comics:
                log.info("stop at tag=%r, combo=%s 无结果", tag, trial)
                break
            self.tgl = trial
            results[" ".join(trial)] = comics
            log.info("combo %s -> %d results", trial, len(comics))
        return results

    # ---------- 搜索方式 2：按作者渐进 ----------
    def ad_author(self) -> dict[str, list[str]]:
        """[a1] -> [a1,a2] -> [a1,a2,a3] ...，无结果即停。

        注意：jm 上大部分作品只有一个作者，多作者 AND 通常第一两轮就 0 结果，
        想让作者维度真正有用，更推荐 ad_author_tag()。
        """
        self.agl = []
        results: dict[str, list[str]] = {}
        for author, _ in self.fav_author.most_common():
            trial = [*self.agl, author]
            comics = self._get_comic(authors=trial)
            if not comics:
                log.info("stop at author=%r, combo=%s 无结果", author, trial)
                break
            self.agl = trial
            results[" | ".join(trial)] = comics
            log.info("combo %s -> %d results", trial, len(comics))
        return results

    # ---------- 搜索方式 3：先作者，再按 tag 渐进 ----------
    def ad_author_tag(self, top_authors: int = 5) -> dict[str, list[str]]:
        """对 top N 作者，锁定作者后在其作品范围内按 tag 热度渐进加 tag。

        key 形式：
          'author:<name>'                 ← 作者本身
          'author:<name> | tag1 tag2 ...' ← 作者 + 累积 tag
        """
        results: dict[str, list[str]] = {}
        for author, _ in self.fav_author.most_common(top_authors):
            base = self._get_comic(authors=[author])
            if not base:
                log.info("author=%r 无结果，跳过", author)
                continue
            results[f"author:{author}"] = base
            log.info("author=%s 基线 -> %d results", author, len(base))

            combo: list[str] = []
            for tag, _c in self.fav_tag.most_common():
                trial = [*combo, tag]
                comics = self._get_comic(tags=trial, authors=[author])
                if not comics:
                    log.info("author=%r 加 tag=%r 后无结果，停止该作者", author, tag)
                    break
                combo = trial
                results[f"author:{author} | {' '.join(combo)}"] = comics
                log.info("author=%s + %s -> %d results", author, combo, len(comics))
        return results

    # ---------- 备选：每个 tag 单独搜 ----------
    def ad_top(self, top_n: int = 20) -> dict[str, list[str]]:
        results: dict[str, list[str]] = {}
        for tag, _ in self.fav_tag.most_common(top_n):
            results[tag] = self._get_comic(tags=[tag])
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cad = ComicAdv()
    cad.load()

    cad.add([])
    cad.save()

    print("top tags:", cad.fav_tag.most_common(10))
    print("top authors:", cad.fav_author.most_common(10))

    if cad.fav_tag:
        print("\n=== 按 tag 渐进 ===")
        for combo, comics in cad.ad().items():
            print(f"{combo}: {len(comics)} results")
            for line in comics[:50]:
                print("   ", line)

    if cad.fav_author:
        print("\n=== 按作者渐进 ===")
        for combo, comics in cad.ad_author().items():
            print(f"{combo}: {len(comics)} results")
            for line in comics[:30]:
                print("   ", line)

    if cad.fav_author and cad.fav_tag:
        print("\n=== 作者限定 + tag 渐进 ===")
        for combo, comics in cad.ad_author_tag(top_authors=3).items():
            print(f"{combo}: {len(comics)} results")
            for line in comics[:20]:
                print("   ", line)