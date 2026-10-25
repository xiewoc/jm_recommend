from __future__ import annotations

import argparse
import html as html_mod
import json
import logging
import math
import os
import re
import threading
import unicodedata
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

from jmcomic import JmAlbumDetail, JmOption, create_option_by_file  # type: ignore
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.prompt import Prompt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_RETRYABLE_EXC: list[type[BaseException]] = [OSError, TimeoutError]
try:
    from jmcomic.exception_all import JmNetworkException as _JmNetExc  # type: ignore
    _RETRYABLE_EXC.append(_JmNetExc)
except Exception:
    pass
try:
    from jmcomic.exception_all import JmTimeoutException as _JmTmExc  # type: ignore
    _RETRYABLE_EXC.append(_JmTmExc)
except Exception:
    pass
try:
    import requests  # type: ignore
    for _n in ('ConnectionError', 'Timeout', 'ConnectTimeout', 'ReadTimeout',
               'ChunkedEncodingError'):
        _e = getattr(requests.exceptions, _n, None)
        if isinstance(_e, type) and issubclass(_e, BaseException):
            _RETRYABLE_EXC.append(_e)
except Exception:
    pass
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(dict.fromkeys(_RETRYABLE_EXC))

console = Console()
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
AUTHOR_QUERY_TEMPLATE = '"{author}"'
PAGE_SIZE = 80

DEFAULT_LANGUAGE_TAGS: set[str] = {
    "中文", "日文", "英文", "漢化", "汉化", "翻译", "翻譯",
    "無修正", "无修正", "全彩", "全彩色",
    "Chinese", "Japanese", "English", "CG",
}

_TAG_PATTERN = r"(?:\[[^\]]*\]|\([^)]*\)|【[^】]*】)"
_TAG_ONE_RE = re.compile(_TAG_PATTERN)
_TAG_PREFIX_RE = re.compile(rf"^(?:{_TAG_PATTERN}\s*)+")
_TAG_SUFFIX_RE = re.compile(rf"\s*(?:{_TAG_PATTERN})+$")

# ==================== 标题工具（纯函数 + LRU 缓存） ====================
@lru_cache(maxsize=8192)
def _split_title_cached(norm_title: str) -> tuple[str, str]:
    """输入已 normalize 的标题，返回 (base, suffix_key)。纯函数，可缓存。"""
    s = _TAG_PREFIX_RE.sub("", norm_title)

    suffixes: list[str] = []
    while True:
        m = _TAG_SUFFIX_RE.search(s)
        if not m:
            break
        for tag in _TAG_ONE_RE.findall(m.group(0)):
            suffixes.append(re.sub(r"\s+", "", tag).lower())
        new_s = s[: m.start()].rstrip()
        if new_s == s:
            break
        s = new_s

    base = re.sub(r"\s+", "", s).lower()
    return base, "|".join(sorted(suffixes))


# ==================== 配置 ====================
@dataclass
class SearchConfig:
    mode: str = "ranked"
    top_tags: int = 15
    top_authors: int = 10
    top_n: int = 100
    idf_power: float = 1.0
    max_pages: int = 1
    author_weight: float = 0.5
    exclude_language_tags: bool = True
    extra_language_tags: list[str] = field(default_factory=list)
    blacklist_tags: list[str] = field(default_factory=list)
    blacklist_penalty: float = 2.0
    enrich_candidates: bool = True
    enrich_extra: int = 50


@dataclass
class ExportConfig:
    format: str = ""


@dataclass
class FeedbackConfig:
    enabled: bool = True
    selected_bonus: float = 0.5
    skipped_penalty: float = 0.2


@dataclass
class ConcurrencyConfig:
    max_workers: int = 8
    # 注意：rate_limit 实际是"并发上限"（Semaphore），并非 QPS 限速。
    rate_limit: int = 4
    cache_size: int = 256


@dataclass
class StorageConfig:
    memo_path: str = "memo.toml"


@dataclass
class AppConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


_DEFAULT_CONFIG_TEXT = """\
# ComicAdv 配置文件（可手动编辑）

[search]
# "ranked"      每个 tag / 作者独立搜索，再按 IDF 加权偏好分排序（推荐）
# "progressive" 老的渐进求交逻辑
mode = "ranked"
top_tags = 15
top_authors = 10
top_n = 100
idf_power = 1.0
max_pages = 1
author_weight = 0.5
exclude_language_tags = true
extra_language_tags = []

# 黑名单：不再硬过滤，而是对命中条目降权（软惩罚）
blacklist_tags = []
blacklist_penalty = 2.0
enrich_candidates = true
enrich_extra = 50

[export]
format = ""          # "" / "md" / "html"

[feedback]
enabled = true
selected_bonus = 0.5
skipped_penalty = 0.2

[concurrency]
max_workers = 8
rate_limit = 4
cache_size = 256

[storage]
memo_path = "memo.toml"
"""


def _write_default_config(path: Path) -> None:
    path.write_text(_DEFAULT_CONFIG_TEXT, encoding="utf-8")


# ---- 配置容错辅助 ----
def _safe_int(d: dict, k: str, default: int) -> int:
    v = d.get(k, default)
    if isinstance(v, bool):  # 防止 True 被当成 1
        log.warning("配置项 %s=%r 类型非法，使用默认 %r", k, v, default)
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        log.warning("配置项 %s=%r 非法，使用默认 %r", k, v, default)
        return default


def _safe_float(d: dict, k: str, default: float) -> float:
    v = d.get(k, default)
    if isinstance(v, bool):
        log.warning("配置项 %s=%r 类型非法，使用默认 %r", k, v, default)
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        log.warning("配置项 %s=%r 非法，使用默认 %r", k, v, default)
        return default


def _safe_bool(d: dict, k: str, default: bool) -> bool:
    v = d.get(k, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in ("true", "1", "yes", "on"):
            return True
        if lv in ("false", "0", "no", "off"):
            return False
    log.warning("配置项 %s=%r 非法，使用默认 %r", k, v, default)
    return default


def _safe_str(d: dict, k: str, default: str) -> str:
    v = d.get(k, default)
    if isinstance(v, str):
        return v
    log.warning("配置项 %s=%r 应为字符串，使用默认 %r", k, v, default)
    return default


def _safe_str_list(d: dict, k: str) -> list[str]:
    v = d.get(k, [])
    if not isinstance(v, list):
        log.warning("配置项 %s=%r 应为列表，已忽略", k, v)
        return []
    return [str(x) for x in v]


def _load_config(path: Path) -> AppConfig:
    if not path.exists():
        log.warning("⚠️ 未找到 %s，已写入默认配置", path)
        _write_default_config(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        log.error("❌ config.toml 解析失败: %s，使用默认配置", e)
        return AppConfig()
    except OSError as e:
        log.error("❌ 读取 config.toml 失败: %s，使用默认配置", e)
        return AppConfig()

    s = data.get("search", {})
    e = data.get("export", {})
    fb = data.get("feedback", {})
    cc = data.get("concurrency", {})
    st = data.get("storage", {})
    if not isinstance(s, dict):  s = {}
    if not isinstance(e, dict):  e = {}
    if not isinstance(fb, dict): fb = {}
    if not isinstance(cc, dict): cc = {}
    if not isinstance(st, dict): st = {}

    return AppConfig(
        search=SearchConfig(
            mode=_safe_str(s, "mode", "ranked"),
            top_tags=_safe_int(s, "top_tags", 15),
            top_authors=_safe_int(s, "top_authors", 10),
            top_n=_safe_int(s, "top_n", 100),
            idf_power=_safe_float(s, "idf_power", 1.0),
            max_pages=_safe_int(s, "max_pages", 1),
            author_weight=_safe_float(s, "author_weight", 0.5),
            exclude_language_tags=_safe_bool(s, "exclude_language_tags", True),
            extra_language_tags=_safe_str_list(s, "extra_language_tags"),
            blacklist_tags=_safe_str_list(s, "blacklist_tags"),
            blacklist_penalty=_safe_float(s, "blacklist_penalty", 2.0),
            enrich_candidates=_safe_bool(s, "enrich_candidates", True),
            enrich_extra=_safe_int(s, "enrich_extra", 50),
        ),
        export=ExportConfig(format=_safe_str(e, "format", "")),
        feedback=FeedbackConfig(
            enabled=_safe_bool(fb, "enabled", True),
            selected_bonus=_safe_float(fb, "selected_bonus", 0.5),
            skipped_penalty=_safe_float(fb, "skipped_penalty", 0.2),
        ),
        concurrency=ConcurrencyConfig(
            max_workers=_safe_int(cc, "max_workers", 8),
            rate_limit=_safe_int(cc, "rate_limit", 4),
            cache_size=_safe_int(cc, "cache_size", 256),
        ),
        storage=StorageConfig(memo_path=_safe_str(st, "memo_path", "memo.toml")),
    )


# ==================== TOML 序列化辅助 ====================
_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(k: str) -> str:
    if _TOML_BARE_KEY_RE.match(k):
        return k
    # 用 JSON 字符串转义：\uXXXX / \n / \r / \t / \" / \\ 都是合法 TOML 基本字符串
    return json.dumps(k, ensure_ascii=False)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"不支持 TOML 类型: {type(v).__name__}")


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


# ---- memo 反序列化容错 ----
def _as_counter(x) -> Counter[str]:
    out: Counter[str] = Counter()
    if isinstance(x, dict):
        for k, v in x.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                log.warning("memo 中 %r 的计数 %r 非法，已忽略", k, v)
    return out


def _as_str_list(x) -> list[str]:
    if isinstance(x, list):
        return [str(i) for i in x]
    return []


def _as_str_set(x) -> set[str]:
    return set(_as_str_list(x))


def _as_title_versions(x) -> dict[str, set[str]]:
    if not isinstance(x, dict):
        return {}
    out: dict[str, set[str]] = {}
    for k, v in x.items():
        if isinstance(v, list):
            out[str(k)] = {str(s) for s in v}
    return out


# ==================== 主类 ====================
class ComicAdv:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.base_dir = BASE_DIR
        self.config_path = BASE_DIR / "config.toml"
        self.config = config or _load_config(self.config_path)

        self.memo_path = BASE_DIR / self.config.storage.memo_path
        self.option_path = BASE_DIR / "option.yml"

        self.fav_tag: Counter[str] = Counter()
        self.fav_author: Counter[str] = Counter()
        self.selected_tags: Counter[str] = Counter()
        self.skipped_tags: Counter[str] = Counter()
        self.added: set[str] = set()
        self.added_title_versions: dict[str, set[str]] = {}
        self.if_login: bool = False

        self._eff_tags_cache: list[tuple[str, int]] | None = None

        if self.config.search.exclude_language_tags:
            self.language_tags: set[str] = (
                set(DEFAULT_LANGUAGE_TAGS)
                | {t.strip() for t in self.config.search.extra_language_tags if t.strip()}
            )
        else:
            self.language_tags = set()

        self.blacklist_tags: set[str] = {
            t.strip() for t in self.config.search.blacklist_tags if t.strip()
        }
        if self.blacklist_tags and not self.config.search.enrich_candidates:
            log.warning(
                "⚠️ 已配置黑名单但 enrich_candidates=false，"
                "搜索结果不带 tags 时将无法计算黑名单惩罚"
            )

        if self.option_path.exists():
            self.option: JmOption = create_option_by_file(str(self.option_path))
            log.info("已加载 option.yml: %s", self.option_path)
        else:
            self.option = JmOption.default()
            log.warning("⚠️ 未找到 %s，使用默认 JmOption", self.option_path)

        self._lock = threading.Lock()
        self._max_workers = self.config.concurrency.max_workers
        # 拆成两个信号量，避免 meta 请求与 search 请求互相挤占
        self._meta_semaphore = threading.Semaphore(self.config.concurrency.rate_limit)
        self._search_semaphore = threading.Semaphore(self.config.concurrency.rate_limit)
        self._local = threading.local()

        self._query_cache: OrderedDict[tuple[str, int], list] = OrderedDict()
        self._query_cache_lock = threading.Lock()

        self._username: str = ""
        self._password: str = ""
        self._login_generation: int = 0

    # ---------- 内部缓存管理 ----------
    def _invalidate_eff_tags(self) -> None:
        self._eff_tags_cache = None

    # ---------- login ----------
    def login(self, username: str = "", password: str = "") -> bool:
        if not (username and password):
            raise ValueError("需要提供 username + password；cookies 请在 option.yml 配置")

        with self._lock:
            self._username = username
            self._password = password
            self._login_generation += 1

        try:
            client = self.option.new_jm_client()
            result = client.login(username, password)
            if not result:
                raise RuntimeError("login returned empty result")
        except Exception as e:
            # 失败时清空凭据，避免后续每个线程反复用错误密码重试
            with self._lock:
                self._username = ""
                self._password = ""
                self._login_generation += 1
            self.if_login = False
            log.error("❌ 登录失败: %r", e)
            return False

        self._local.client = client
        self._local.generation = self._login_generation
        with self._query_cache_lock:
            self._query_cache.clear()
        self.if_login = True
        log.info("✅ 登录成功")
        return True

    # ---------- client ----------
    def _get_client(self):
        gen = self._login_generation
        client = getattr(self._local, "client", None)
        client_gen = getattr(self._local, "generation", -1)

        if client is None or client_gen != gen:
            client = self.option.new_jm_client()
            if self._username and self._password:
                try:
                    result = client.login(self._username, self._password)
                    if not result:
                        log.warning("⚠️ 线程 client 登录返回空结果，以匿名状态继续")
                except Exception as e:
                    log.warning("⚠️ 线程 client 登录异常: %r，以匿名状态继续", e)
            self._local.client = client
            self._local.generation = gen
        return client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def _fetch_meta_safe(self, album_id: str) -> tuple[list[str], list[str], str]:
        with self._meta_semaphore:
            album: JmAlbumDetail = self._get_client().get_album_detail(album_id)
        tags = list(album.tags or [])
        authors = self._extract_authors(album)
        title = getattr(album, "title", "") or getattr(album, "name", "") or ""
        return tags, authors, str(title)

    # ---------- 标题工具 ----------
    @staticmethod
    def _norm_title(title: str) -> str:
        if not title:
            return ""
        s = unicodedata.normalize("NFKC", title)
        return " ".join(s.split())

    @classmethod
    def _split_title(cls, title: str) -> tuple[str, str]:
        # 先 normalize，再走模块级 LRU 缓存
        return _split_title_cached(cls._norm_title(title))

    def _is_title_dup(self, title: str) -> bool:
        base, suf = self._split_title(title)
        if not base:
            return False
        return suf in self.added_title_versions.get(base, set())

    def _register_title(self, title: str) -> bool:
        base, suf = self._split_title(title)
        if not base:
            return False
        dup = suf in self.added_title_versions.get(base, set())
        self.added_title_versions.setdefault(base, set()).add(suf)
        return dup

    @staticmethod
    def _extract_authors(album) -> list[str]:
        for attr in ("authors", "author"):
            val = getattr(album, attr, None)
            if not val:
                continue
            if isinstance(val, str):
                return [val]
            return [str(x) for x in val]
        return []

    @staticmethod
    def _clean_ids(inpt: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for raw in inpt:
            aid = raw.strip()
            if not aid:
                continue
            if not aid.isdigit():
                log.warning("⚠️ 跳过非法 ID: %r（应为纯数字）", raw)
                continue
            if aid in seen:
                continue
            seen.add(aid)
            result.append(aid)
        return result

    def _effective_fav_tags(self) -> list[tuple[str, int]]:
        if self._eff_tags_cache is not None:
            return self._eff_tags_cache
        eff = [
            (t, c) for t, c in self.fav_tag.most_common()
            if t not in self.language_tags
        ]
        eff.sort(key=lambda x: self._tag_weight(x[0]), reverse=True)
        self._eff_tags_cache = eff
        return eff

    # ---------- 持久化 ----------
    def load(self) -> None:
        if self.memo_path.exists():
            self._load_memo_toml()
            return
        old_json = self.base_dir / "memo.json"
        if old_json.exists():
            log.info("发现旧 memo.json，迁移到 %s", self.memo_path)
            self._migrate_from_json(old_json)
            self.save()
            return
        log.info("未找到 memo，从空开始")

    def _load_memo_toml(self) -> None:
        try:
            with self.memo_path.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError) as e:
            log.error("❌ memo.toml 读取/解析失败: %s，从空开始", e)
            return

        self.fav_tag = _as_counter(data.get("fav_tag"))
        self.fav_author = _as_counter(data.get("fav_author"))

        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        self.added = _as_str_set(meta.get("added"))
        self.added_title_versions = _as_title_versions(meta.get("title_versions"))

        fb = data.get("feedback", {})
        if not isinstance(fb, dict):
            fb = {}
        self.selected_tags = _as_counter(fb.get("selected_tags"))
        self.skipped_tags = _as_counter(fb.get("skipped_tags"))

        self._invalidate_eff_tags()
        n_versions = sum(len(v) for v in self.added_title_versions.values())
        log.info(
            "已加载 %d tags, %d authors, %d added, %d 主标题 / %d 版本, "
            "feedback %d selected / %d skipped",
            len(self.fav_tag), len(self.fav_author),
            len(self.added), len(self.added_title_versions), n_versions,
            len(self.selected_tags), len(self.skipped_tags),
        )

    def _migrate_from_json(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.error("❌ 读取 %s 失败: %s", path, e)
            return
        self.fav_tag = _as_counter(data.get("fav_tag"))
        self.fav_author = _as_counter(data.get("fav_author"))
        self.added = _as_str_set(data.get("added"))
        for t in data.get("added_titles", []):
            self._register_title(t)
        self._invalidate_eff_tags()
        log.info(
            "迁移完成：%d tags, %d authors, %d added, %d 主标题",
            len(self.fav_tag), len(self.fav_author),
            len(self.added), len(self.added_title_versions),
        )

    def save(self) -> None:
        with self._lock:
            lines: list[str] = [
                "# ComicAdv memo —— 程序自动生成，可手动编辑",
                "",
                "[meta]",
                f"added = {_toml_value(sorted(self.added))}",
                "",
            ]
            if self.added_title_versions:
                lines.append("[meta.title_versions]")
                for base, sufs in sorted(self.added_title_versions.items()):
                    if not base:
                        continue
                    lines.append(f"{_toml_key(base)} = {_toml_value(sorted(sufs))}")
                lines.append("")
            if self.fav_tag:
                lines.append("[fav_tag]")
                for tag, cnt in self.fav_tag.most_common():
                    lines.append(f"{_toml_key(tag)} = {cnt}")
                lines.append("")
            if self.fav_author:
                lines.append("[fav_author]")
                for author, cnt in self.fav_author.most_common():
                    lines.append(f"{_toml_key(author)} = {cnt}")
                lines.append("")
            if self.selected_tags:
                lines.append("[feedback.selected_tags]")
                for tag, cnt in self.selected_tags.most_common():
                    lines.append(f"{_toml_key(tag)} = {cnt}")
                lines.append("")
            if self.skipped_tags:
                lines.append("[feedback.skipped_tags]")
                for tag, cnt in self.skipped_tags.most_common():
                    lines.append(f"{_toml_key(tag)} = {cnt}")
                lines.append("")
            content = "\n".join(lines)

        tmp = self.memo_path.with_suffix(".toml.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.memo_path)
        log.info("💾 数据已保存到 %s", self.memo_path)

    # ---------- 抓取 ----------
    def add(
        self,
        inpt: Iterable[str],
        preset_meta: dict[str, tuple[list[str], list[str], str]] | None = None,
    ) -> None:
        pending = [i for i in self._clean_ids(inpt) if i not in self.added]
        if not pending:
            log.info("✅ 无新增条目需要处理")
            return

        preset_meta = preset_meta or {}

        def _get_meta(aid: str):
            if aid in preset_meta:
                return preset_meta[aid]
            return self._fetch_meta_safe(aid)

        log.info("🚀 开始抓取 %d 个新条目...", len(pending))
        success = fail = dup = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_get_meta, i): i for i in pending}
            for fut in as_completed(futures):
                aid = futures[fut]
                try:
                    tags, authors, title = fut.result()
                    with self._lock:
                        self.added.add(aid)
                        self.fav_tag.update(tags)
                        self.fav_author.update(authors)
                        if self._register_title(title):
                            dup += 1
                            log.warning("⚠️ 标题重复（仍计入 added）: %r", title)
                    success += 1
                except Exception as e:
                    fail += 1
                    log.warning("❌ fetch %s failed: %r", aid, e)

        self._invalidate_eff_tags()
        console.print(
            f"[bold green]抓取完成:[/] 成功 {success}, 失败 {fail}, "
            f"标题重复 {dup}"
        )

    # ---------- 查询 ----------
    def _build_query(
        self,
        tags: Iterable[str] = (),
        authors: Iterable[str] = (),
    ) -> str:
        """构建搜索 query。黑名单改为软惩罚，不发服务端。"""
        parts: list[str] = []
        for a in authors:
            # 转义作者名中的引号和反斜杠，避免破坏 query 语法
            a_esc = a.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(AUTHOR_QUERY_TEMPLATE.format(author=a_esc))
        for t in tags:
            parts.append(f"+{t}")
        return " ".join(parts)

    def _cached_search(self, query: str, page: int) -> list | None:
        key = (query, page)
        with self._query_cache_lock:
            if key in self._query_cache:
                self._query_cache.move_to_end(key)  # LRU: 命中移到末尾
                return self._query_cache[key]
        with self._search_semaphore:
            try:
                page_obj = self._get_client().search_site(search_query=query, page=page)
                batch = list(page_obj)
            except Exception as e:
                log.warning("search %r (page=%d) failed: %r", query, page, e)
                return None
        with self._query_cache_lock:
            self._query_cache[key] = batch
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self.config.concurrency.cache_size:
                self._query_cache.popitem(last=False)
        return batch

    def _get_comic(
        self,
        tags: Iterable[str] = (),
        authors: Iterable[str] = (),
        page: int = 1,
        exclude_added: bool = True,
        max_pages: int = 1,
    ) -> tuple[list[dict], int]:
        tags_list = list(tags)
        authors_list = list(authors)
        if not tags_list and not authors_list:
            return [], 0

        query = self._build_query(tags_list, authors_list)
        raw_items: list = []
        fetched_pages = 0
        stop_reason = "未开始"
        last_page_size = 0

        for p in range(page, page + max_pages):
            batch = self._cached_search(query, p)
            if batch is None:
                stop_reason = f"第 {p} 页异常"
                break
            fetched_pages += 1
            last_page_size = len(batch)
            if not batch:
                stop_reason = f"第 {p} 页返回空"
                break
            raw_items.extend(batch)
            if len(batch) < PAGE_SIZE:
                stop_reason = f"第 {p} 页仅 {len(batch)} 条（视为末页）"
                break
        else:
            stop_reason = f"达到 max_pages={max_pages}"

        log.debug(
            "🔍 query=%r -> 共 %d 条（翻 %d 页，末页 %d 条，%s）",
            query, len(raw_items), fetched_pages, last_page_size, stop_reason,
        )

        items: list[dict] = []
        for item in raw_items:
            if hasattr(item, "album_id"):
                sid = str(item.album_id)
                title = getattr(item, "title", "") or getattr(item, "name", "") or ""
            else:
                try:
                    sid_raw, title = item
                except Exception:
                    continue
                sid = str(sid_raw)
                title = str(title)
            is_added = sid in self.added or self._is_title_dup(title)
            if exclude_added and is_added:
                continue
            item_authors = (self._extract_authors(item)
                              if hasattr(item, "album_id") else [])
            items.append({
                "id": sid, "title": title, "added": is_added,
                "authors": item_authors,
            })

        return items, len(raw_items)

    # ---------- 打分 ----------
    def _tag_weight(self, tag: str) -> float:
        base = float(self.fav_tag.get(tag, 0))
        fb = self.config.feedback
        if not fb.enabled:
            return base
        bonus = self.selected_tags.get(tag, 0) * fb.selected_bonus
        penalty = self.skipped_tags.get(tag, 0) * fb.skipped_penalty
        return max(0.0, base + bonus - penalty)

    @staticmethod
    def _idf(count: int, total: int, power: float = 1.0) -> float:
        base = math.log((total + 1) / (count + 1)) + 1.0
        return base ** power

    def _score_hits(
        self,
        hit_tags: Iterable[str],
        hit_authors: Iterable[str],
        total_tag: int,
        total_author: int,
        idf_power: float,
        author_weight: float,
        all_tags: Iterable[str] = (),
    ) -> float:
        tag_score = sum(
            self._tag_weight(t)
            * self._idf(self.fav_tag.get(t, 0), total_tag, idf_power)
            for t in hit_tags if t in self.fav_tag
        )
        author_score = sum(
            self.fav_author.get(a, 0)
            * self._idf(self.fav_author.get(a, 0), total_author, 1.0)
            for a in hit_authors if a in self.fav_author
        )
        base = tag_score + author_weight * author_score
        if self.blacklist_tags:
            tag_set = all_tags if isinstance(all_tags, (set, frozenset)) else set(all_tags)
            bl_hit = tag_set & self.blacklist_tags
            if bl_hit:
                base -= len(bl_hit) * self.config.search.blacklist_penalty
        return base

    # ---------- enrich ----------
    def _enrich_candidates(self, candidates: list[dict]) -> None:
        to_fetch = [c["id"] for c in candidates if not c.get("all_tags")]
        if not to_fetch:
            return

        log.info("🔎 enrich %d 个候选的 tags（用于黑名单降权）...", len(to_fetch))
        by_id = {c["id"]: c for c in candidates}

        def _fetch(aid: str):
            try:
                tags, authors, _title = self._fetch_meta_safe(aid)
                return aid, tags, authors
            except Exception:
                log.debug("enrich %s failed", aid, exc_info=True)
                return aid, None, None

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for aid, tags, authors in pool.map(_fetch, to_fetch):
                if tags is None:
                    continue
                cand = by_id.get(aid)
                if not cand:
                    continue
                cand["all_tags"] = tags
                if authors:
                    for a in authors:
                        if a in self.fav_author:
                            cand["hit_authors"].add(a)

        log.info("✅ enrich 完成")

    # ---------- 共享的排名流程 ----------
    def _rank_candidates(
        self,
        candidates: dict[str, dict],
        total_tag: int,
        total_author: int,
        top_n: int,
        idf_power: float,
        author_weight: float,
    ) -> list[dict]:
        cfg = self.config.search
        for c in candidates.values():
            c["_pre_score"] = self._score_hits(
                c["hit_tags"], c["hit_authors"],
                total_tag, total_author, idf_power, author_weight,
            )

        if cfg.enrich_candidates and self.blacklist_tags and candidates:
            pre_sorted = sorted(
                candidates.values(), key=lambda c: c["_pre_score"], reverse=True
            )
            enrich_target = pre_sorted[: top_n + cfg.enrich_extra]
            self._enrich_candidates(enrich_target)

        for c in candidates.values():
            c["_score"] = self._score_hits(
                c["hit_tags"], c["hit_authors"],
                total_tag, total_author, idf_power, author_weight,
                all_tags=c.get("all_tags", []),
            )
            c["hit_tags"] = sorted(c["hit_tags"])
            c["hit_authors"] = sorted(c["hit_authors"])
            c["blacklist_hit"] = sorted(
                set(c.get("all_tags", [])) & self.blacklist_tags
            )
            c.pop("_pre_score", None)

        # 已收藏的沉底，避免挤占 top_n
        return sorted(
            candidates.values(),
            key=lambda c: (bool(c.get("added")), -c["_score"]),
        )[:top_n]

    # ---------- ranked ----------
    def ad_ranked(
        self,
        top_tags: int | None = None,
        top_n: int | None = None,
        max_pages: int | None = None,
        idf_power: float | None = None,
    ) -> dict[str, list[dict]]:
        cfg = self.config.search
        top_tags = top_tags if top_tags is not None else cfg.top_tags
        top_n = top_n if top_n is not None else cfg.top_n
        max_pages = max_pages if max_pages is not None else cfg.max_pages
        idf_power = idf_power if idf_power is not None else cfg.idf_power

        eff_tags = self._effective_fav_tags()[:top_tags]
        if not eff_tags:
            return {}

        total_tag = sum(self.fav_tag.values()) or 1
        total_author = sum(self.fav_author.values()) or 1
        candidates: dict[str, dict] = {}

        def _fetch(tag: str):
            comics, _ = self._get_comic(tags=[tag], max_pages=max_pages)
            return tag, comics

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for tag, comics in pool.map(_fetch, [t for t, _ in eff_tags]):
                for c in comics:
                    cand = candidates.setdefault(c["id"], {
                        "id": c["id"], "title": c["title"], "added": c["added"],
                        "hit_tags": set(), "hit_authors": set(),
                        "all_tags": [],
                    })
                    cand["hit_tags"].add(tag)
                    for _a in c.get("authors", []):
                        if _a in self.fav_author:
                            cand["hit_authors"].add(_a)

        ranked = self._rank_candidates(
            candidates, total_tag, total_author, top_n, idf_power, cfg.author_weight
        )
        if not ranked:
            return {}
        log.info("✅ [ranked tag] 候选 %d，Top %d", len(candidates), len(ranked))
        label = f"🏆 按 tag 打分（top_tags={len(eff_tags)}，top_n={len(ranked)}）"
        return {label: ranked}

    def ad_author_ranked(
        self,
        top_authors: int | None = None,
        top_n: int | None = None,
        max_pages: int | None = None,
    ) -> dict[str, list[dict]]:
        cfg = self.config.search
        top_authors = top_authors if top_authors is not None else cfg.top_authors
        top_n = top_n if top_n is not None else cfg.top_n
        max_pages = max_pages if max_pages is not None else cfg.max_pages

        if not self.fav_author:
            return {}

        total_tag = sum(self.fav_tag.values()) or 1
        total_author = sum(self.fav_author.values()) or 1
        candidates: dict[str, dict] = {}

        def _fetch(author: str):
            comics, _ = self._get_comic(
                authors=[author], exclude_added=False, max_pages=max_pages
            )
            return author, comics

        authors = [a for a, _ in self.fav_author.most_common(top_authors)]
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for author, comics in pool.map(_fetch, authors):
                for c in comics:
                    cand = candidates.setdefault(c["id"], {
                        "id": c["id"], "title": c["title"], "added": c["added"],
                        "hit_tags": set(), "hit_authors": set(),
                        "all_tags": [],
                    })
                    cand["hit_authors"].add(author)

        ranked = self._rank_candidates(
            candidates, total_tag, total_author, top_n, cfg.idf_power, cfg.author_weight
        )
        if not ranked:
            return {}
        log.info("✅ [ranked author] 候选 %d，Top %d", len(candidates), len(ranked))
        label = f"🏆 按作者打分（top_authors={len(authors)}，top_n={len(ranked)}）"
        return {label: ranked}

    # ---------- progressive ----------
    def ad(self, max_pages: int = 1) -> dict[str, list[dict]]:
        tgl: list[str] = []
        results: dict[str, list[dict]] = {}
        skipped: list[str] = []
        for tag, _ in self._effective_fav_tags():
            trial = [*tgl, tag]
            comics, raw = self._get_comic(tags=trial, max_pages=max_pages)
            if raw == 0:
                skipped.append(tag)
                continue
            tgl = trial
            results[" + ".join(trial)] = comics
        if skipped:
            log.warning("⚠️ 跳过 %d 个无效 tag", len(skipped))
        return results

    def ad_author(self, max_pages: int = 1) -> dict[str, list[dict]]:
        agl: list[str] = []
        results: dict[str, list[dict]] = {}
        skipped: list[str] = []
        for author, _ in self.fav_author.most_common():
            trial = [*agl, author]
            comics, raw = self._get_comic(authors=trial, max_pages=max_pages)
            if raw == 0:
                skipped.append(author)
                continue
            agl = trial
            results[" | ".join(trial)] = comics
        if skipped:
            log.warning("⚠️ 跳过 %d 个无效作者", len(skipped))
        return results

    def ad_author_tag(
        self, top_authors: int = 5, max_pages: int = 1
    ) -> dict[str, list[dict]]:
        results: dict[str, list[dict]] = {}
        for author, _ in self.fav_author.most_common(top_authors):
            base, base_total = self._get_comic(
                authors=[author], exclude_added=False, max_pages=max_pages
            )
            if base_total == 0:
                continue
            results[f"author:{author}"] = base
            combo: list[str] = []
            for tag, _ in self._effective_fav_tags():
                trial = [*combo, tag]
                comics, raw = self._get_comic(
                    tags=trial, authors=[author],
                    exclude_added=False, max_pages=max_pages,
                )
                if raw == 0:
                    continue
                combo = trial
                results[f"author:{author} | {' + '.join(combo)}"] = comics
        return results

    # ---------- 黑名单检查（软警告） ----------
    def _check_blacklist_meta(
        self, ids: list[str]
    ) -> tuple[list[str], dict[str, tuple[list[str], list[str], str]]]:
        if not ids or not self.blacklist_tags:
            return ids, {}

        meta_cache: dict[str, tuple[list[str], list[str], str]] = {}
        meta_lock = threading.Lock()

        def _check(aid: str):
            try:
                meta = self._fetch_meta_safe(aid)
            except Exception as e:
                log.warning("blacklist check failed for %s: %r", aid, e)
                return aid, set()
            with meta_lock:
                meta_cache[aid] = meta
            return aid, set(meta[0]) & self.blacklist_tags

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            results = list(pool.map(_check, ids))

        for aid, bad in results:
            if bad:
                log.warning(
                    "⚠️ %s 命中黑名单 %s（尊重用户选择，仍添加）",
                    aid, sorted(bad),
                )

        return ids, meta_cache

    # ---------- 收集 ----------
    def _collect_grouped(
        self, results: dict[str, list[dict]]
    ) -> list[tuple[str, list[dict]]]:
        grouped: list[tuple[str, list[dict]]] = []
        seen_ids: set[str] = set()
        seen_versions: set[tuple[str, str]] = set()

        for combo, comics in results.items():
            if not comics:
                continue
            uniq: list[dict] = []
            for item in comics:
                nid = item["id"]
                base, suf = self._split_title(item["title"])
                key = (base, suf) if base else None
                if nid in seen_ids or (key and key in seen_versions):
                    continue
                seen_ids.add(nid)
                if key:
                    seen_versions.add(key)
                uniq.append(item)
            if uniq:
                grouped.append((combo, uniq))
        return grouped

    # ---------- 前端视图 ----------
    def build_view(self, results: dict[str, list[dict]]) -> list[dict]:
        grouped = self._collect_grouped(results)
        out: list[dict] = []
        for combo, uniq in grouped:
            items = []
            for i, it in enumerate(uniq, 1):
                all_tags = list(it.get("all_tags", []))
                bl_hit = (
                    sorted(set(all_tags) & self.blacklist_tags)
                    if all_tags else list(it.get("blacklist_hit", []))
                )
                items.append({
                    "idx": i,
                    "id": it["id"],
                    "title": it["title"],
                    "added": bool(it.get("added")),
                    "hit_tags": list(it.get("hit_tags", [])),
                    "all_tags": all_tags,
                    "blacklist_hit": bl_hit,
                    "score": it.get("_score"),
                })
            out.append({"combo": combo, "items": items})
        return out

    # ---------- 导出 ----------
    def _default_export_path(self, fmt: str, label: str = "") -> Path:
        ext = "html" if fmt in ("html", "htm") else "md"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        name = f"results_{ts}"
        if label:
            safe = re.sub(r'[\\/:*?"<>|]', "_", label)
            name += f"_{safe}"
        return self.base_dir / f"{name}.{ext}"

    def _to_markdown(
        self, grouped: list[tuple[str, list[dict]]], title: str
    ) -> str:
        lines: list[str] = [f"# {title}", ""]
        lines.append(f"> 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        if self.blacklist_tags:
            lines.append(
                f"> 黑名单（软惩罚 × {self.config.search.blacklist_penalty}）："
                f"{', '.join(sorted(self.blacklist_tags))}"
            )
            lines.append("")
        if not grouped:
            lines.append("_未搜索到任何结果_")
            return "\n".join(lines) + "\n"

        total = 0
        for combo, items in grouped:
            lines.append(f"## {combo} ({len(items)} 条)")
            lines.append("")
            lines.append("| # | ID | 标题 | 状态 | 命中 | 黑名单 | 分数 |")
            lines.append("|---:|----|------|------|------|--------|-----:|")
            for idx, item in enumerate(items, 1):
                status = "已收藏" if item.get("added") else "新"
                t = _md_escape(item["title"])
                hit = _md_escape(", ".join(item.get("hit_tags", [])))
                bl = _md_escape(", ".join(item.get("blacklist_hit", [])))
                score = item.get("_score")
                score_s = (
                    f"{score:.2f}"
                    if isinstance(score, (int, float)) and not isinstance(score, bool)
                    else ""
                )
                lines.append(
                    f"| {idx} | {item['id']} | {t} | {status} | {hit} | {bl} | {score_s} |"
                )
            lines.append("")
            total += len(items)
        lines.append(f"共 {len(grouped)} 组，{total} 条结果。")
        return "\n".join(lines) + "\n"

    def _to_html(self, grouped: list[tuple[str, list[dict]]], title: str) -> str:
        esc = html_mod.escape
        parts: list[str] = [
            "<!DOCTYPE html>", '<html lang="zh-CN">', "<head>",
            '<meta charset="utf-8">',
            f"<title>{esc(title)}</title>",
            "<style>",
            "body{font-family:-apple-system,'Segoe UI',sans-serif;margin:2em;color:#222;}",
            "table{border-collapse:collapse;width:100%;margin-bottom:2em;}",
            "th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;}",
            "th{background:#f5f5f5;}",
            "tr:nth-child(even){background:#fafafa;}",
            ".added{color:#999;}",
            ".hit{color:#666;font-size:.9em;}",
            ".bl{color:#c0392b;font-size:.9em;}",
            ".score{color:#0a7;font-variant-numeric:tabular-nums;}",
            "h1{margin-top:0;}",
            "</style>", "</head>", "<body>",
            f"<h1>{esc(title)}</h1>",
            f'<p>导出时间：{esc(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>',
        ]
        if self.blacklist_tags:
            parts.append(
                f'<p>黑名单（软惩罚 × {self.config.search.blacklist_penalty}）：'
                f'{esc(", ".join(sorted(self.blacklist_tags)))}</p>'
            )
        if not grouped:
            parts.append("<p><em>未搜索到任何结果</em></p>")
        else:
            total = 0
            for combo, items in grouped:
                parts.append(f"<h2>{esc(combo)} ({len(items)} 条)</h2>")
                parts.append(
                    "<table><thead><tr>"
                    "<th>#</th><th>ID</th><th>标题</th><th>状态</th>"
                    "<th>命中</th><th>黑名单</th><th>分数</th>"
                    "</tr></thead><tbody>"
                )
                for idx, item in enumerate(items, 1):
                    status = (
                        '<span class="added">已收藏</span>'
                        if item.get("added") else "新"
                    )
                    hit = ", ".join(item.get("hit_tags", []))
                    bl = ", ".join(item.get("blacklist_hit", []))
                    score = item.get("_score")
                    score_s = (
                        f"{score:.2f}"
                        if isinstance(score, (int, float)) and not isinstance(score, bool)
                        else ""
                    )
                    parts.append(
                        f"<tr><td>{idx}</td>"
                        f"<td>{esc(item['id'])}</td>"
                        f"<td>{esc(item['title'])}</td>"
                        f"<td>{status}</td>"
                        f'<td class="hit">{esc(hit)}</td>'
                        f'<td class="bl">{esc(bl)}</td>'
                        f'<td class="score">{esc(score_s)}</td></tr>'
                    )
                parts.append("</tbody></table>")
                total += len(items)
            parts.append(f"<p>共 {len(grouped)} 组，{total} 条结果。</p>")
        parts.append("</body>")
        parts.append("</html>")
        return "\n".join(parts) + "\n"

    def export_results(
        self,
        results: dict[str, list[dict]],
        fmt: str = "md",
        path: Path | None = None,
        title: str = "ComicAdv 搜索结果",
        label: str = "",
        grouped: list[tuple[str, list[dict]]] | None = None,
    ) -> Path:
        fmt = fmt.lower().lstrip(".")
        if grouped is None:
            grouped = self._collect_grouped(results)
        if fmt in ("md", "markdown"):
            content = self._to_markdown(grouped, title)
        elif fmt in ("html", "htm"):
            content = self._to_html(grouped, title)
        else:
            raise ValueError(f"不支持的导出格式: {fmt!r}（支持 md / html）")
        if path is None:
            path = self._default_export_path(fmt, label)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        log.info("📤 已导出 %s: %s", fmt.upper(), path)
        return path

    # ---------- 展示 & 交互 ----------
    def display_results(
        self,
        results: dict[str, list[dict]],
        max_show: int = 30,
        export_format: str | None = None,
        export_path: Path | None = None,
        export_label: str = "",
    ) -> Path | None:
        # 提前算好 grouped，导出时复用，避免重复计算
        grouped = self._collect_grouped(results) if results else []

        def _do_export() -> Path | None:
            if not export_format:
                return None
            p = self.export_results(
                results, fmt=export_format, path=export_path,
                label=export_label, grouped=grouped,
            )
            console.print(f"[green]已导出 {export_format.upper()}:[/] {p}")
            return p

        if not results:
            console.print(
                "[yellow]未搜索到任何结果[/] "
                "（若每条 query 都返回 0，通常是未登录 / cookies 失效）"
            )
            return _do_export()

        if not grouped:
            console.print("[yellow]搜索结果均已收藏或已过滤，无新条目[/]")
            return _do_export()

        # 全局连续编号，避免多组时用户序号错位
        all_candidates: list[dict] = []
        global_idx = 0
        for combo, uniq in grouped:
            table = Table(title=f"📚 {combo} ({len(uniq)} 条结果)", show_lines=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("ID", style="cyan", width=8)
            table.add_column("标题", ratio=1)
            table.add_column("分数", style="green", width=7, justify="right")

            show_items = uniq[:max_show]
            for item in show_items:
                global_idx += 1
                title_text = item["title"]
                if item.get("added"):
                    title_text += " [dim](已收藏)[/dim]"
                if item.get("hit_tags"):
                    preview = ", ".join(item["hit_tags"][:3])
                    title_text += f" [dim](命中: {preview})[/dim]"
                bl_hit = item.get("blacklist_hit") or []
                if bl_hit:
                    title_text += f" [red](黑名单: {', '.join(bl_hit[:2])})[/red]"
                score = item.get("_score")
                score_s = f"{score:.2f}" if isinstance(score, float) else ""
                table.add_row(str(global_idx), item["id"], title_text, score_s)
                all_candidates.append(item)

            if len(uniq) > max_show:
                table.add_row(
                    "...", f"+{len(uniq) - max_show}",
                    f"更多结果未显示（仅前 {max_show} 条可选）", "",
                )
            console.print(table)

        if not all_candidates:
            return _do_export()

        choice = Prompt.ask(
            "\n输入全局序号添加到收藏（逗号/空格分隔，如 1,3,5-8），直接回车跳过",
            default="",
        )
        selected_ids = self._parse_selection(choice, len(all_candidates))

        if self.config.feedback.enabled:
            selected_set = set(selected_ids)
            with self._lock:
                for i, item in enumerate(all_candidates):
                    tags = item.get("hit_tags", [])
                    if not tags:
                        continue
                    if i in selected_set:
                        target = self.selected_tags
                    elif not item.get("added"):
                        target = self.skipped_tags
                    else:
                        continue
                    for tag in tags:
                        target[tag] += 1
            self._invalidate_eff_tags()
            log.info(
                "📝 反馈记录：选中 %d 条，跳过 %d 条（已收藏不记 skipped）",
                len(selected_ids),
                sum(1 for i, it in enumerate(all_candidates)
                    if i not in selected_set and not it.get("added")),
            )

        if selected_ids:
            ids = [all_candidates[i]["id"] for i in selected_ids]
            kept, meta = self._check_blacklist_meta(ids)
            console.print(f"[green]开始添加 {len(kept)} 条...[/]")
            self.add(kept, preset_meta=meta)
            self.save()

        return _do_export()

    @staticmethod
    def _parse_selection(raw: str, max_idx: int) -> list[int]:
        if not raw.strip():
            return []
        indices: set[int] = set()
        # 支持中文逗号、英文逗号、空白分隔
        for part in re.split(r"[,，\s]+", raw.strip()):
            if not part:
                continue
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    a, b = int(start), int(end)
                    if a > b:
                        a, b = b, a
                    for i in range(a, b + 1):
                        if 1 <= i <= max_idx:
                            indices.add(i - 1)
                        else:
                            log.warning("⚠️ 序号越界: %d（有效 1-%d）", i, max_idx)
                except ValueError:
                    log.warning("⚠️ 无法解析选择: %r", part)
            else:
                try:
                    i = int(part)
                    if 1 <= i <= max_idx:
                        indices.add(i - 1)
                    else:
                        log.warning("⚠️ 序号越界: %d（有效 1-%d）", i, max_idx)
                except ValueError:
                    log.warning("⚠️ 无法解析选择: %r", part)
        return sorted(indices)


# ==================== CLI ====================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,  # 允许重复调用（测试 / notebook）
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ComicAdv",
        description="基于 jmcomic 的偏好推荐 / 搜索工具",
    )
    p.add_argument("--add", nargs="*", default=[],
                   help="直接抓取并加入收藏的 album id 列表")
    p.add_argument("--user", default=os.getenv("JM_USER", ""),
                   help="登录用户名（默认读环境变量 JM_USER）")
    p.add_argument("--password", default=os.getenv("JM_PASSWORD", ""),
                   help="登录密码（默认读环境变量 JM_PASSWORD）")
    p.add_argument("--mode", choices=["ranked", "progressive"], default=None,
                   help="覆盖 config.toml 中的 search.mode")
    p.add_argument("--no-login", action="store_true",
                   help="跳过登录（使用匿名 / cookies 模式）")
    p.add_argument("--no-search", action="store_true",
                   help="只做 --add 抓取，不执行推荐搜索")
    return p


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = _build_arg_parser().parse_args(argv)

    cfg = _load_config(BASE_DIR / "config.toml")
    cad = ComicAdv(config=cfg)
    cad.load()

    # ---------- 登录 ----------
    if not args.no_login:
        if args.user and args.password:
            if not cad.login(args.user, args.password):
                console.print("[red]登录失败，将以匿名状态继续[/red]")
        elif args.user or args.password:
            log.warning("⚠️ 只提供了用户名或密码之一，跳过登录")
        else:
            log.debug("未提供登录凭据，使用匿名 / cookies 模式")

    # ---------- 批量添加 ----------
    if args.add:
        cad.add(args.add)
        cad.save()

    if args.no_search:
        return

    if cad.blacklist_tags:
        console.print(
            f"[bold]🚫 黑名单（软惩罚 × {cfg.search.blacklist_penalty}）:[/] "
            f"{sorted(cad.blacklist_tags)}"
        )
    if cad.language_tags:
        console.print(
            f"[bold]🌐 已忽略语言标签:[/] {sorted(cad.language_tags)[:6]}…"
        )
    console.print("\n[bold]🏷️ Top Tags:[/]", cad._effective_fav_tags()[:10])
    console.print("[bold]✍️ Top Authors:[/]", cad.fav_author.most_common(10))

    s = cfg.search
    mode = args.mode or s.mode
    export_fmt = cfg.export.format or None

    if mode == "ranked":
        def _tag():
            return cad.ad_ranked()

        def _author():
            return cad.ad_author_ranked()
    else:
        def _tag():
            return cad.ad(max_pages=s.max_pages)

        def _author():
            return cad.ad_author(max_pages=s.max_pages)

    if cad._effective_fav_tags():
        console.rule(f"[bold blue]按 Tag 搜索（{mode}）")
        cad.display_results(
            _tag(), export_format=export_fmt, export_label="tag",
        )

    if cad.fav_author:
        console.rule(f"[bold magenta]按作者搜索（{mode}）")
        cad.display_results(
            _author(), export_format=export_fmt, export_label="author",
        )

    if mode == "progressive" and cad.fav_author and cad._effective_fav_tags():
        console.rule("[bold yellow]作者限定 + Tag 渐进")
        cad.display_results(
            cad.ad_author_tag(top_authors=3, max_pages=s.max_pages),
            export_format=export_fmt, export_label="author_tag",
        )


if __name__ == "__main__":
    main()