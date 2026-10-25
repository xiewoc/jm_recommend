from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 按你的实际文件名改这一行
from main import BASE_DIR, ComicAdv, _load_config

app = FastAPI(title="ComicAdv Web")
log = logging.getLogger("web")

_cad: Optional[ComicAdv] = None
_lock = asyncio.Lock()


def get_cad() -> ComicAdv:
    global _cad
    if _cad is None:
        cfg = _load_config(BASE_DIR / "config.toml")
        _cad = ComicAdv(config=cfg)
        _cad.load()
    return _cad


@app.get("/api/status")
async def status():
    cad = get_cad()
    return {
        "fav_tag_top": cad.fav_tag.most_common(10),
        "fav_author_top": cad.fav_author.most_common(10),
        "effective_tags": cad._effective_fav_tags()[:10],
        "blacklist": sorted(cad.blacklist_tags),
        "language_tags": sorted(cad.language_tags)[:10],
        "added_count": len(cad.added),
        "selected_tags_top": cad.selected_tags.most_common(5),
        "skipped_tags_top": cad.skipped_tags.most_common(5),
        "mode": cad.config.search.mode,
    }


class SearchReq(BaseModel):
    kind: str = "tag"          # "tag" | "author"
    top_n: Optional[int] = None
    max_pages: Optional[int] = None


@app.post("/api/search")
async def search(req: SearchReq):
    cad = get_cad()
    async with _lock:
        if req.kind == "tag":
            fn = lambda: cad.ad_ranked(top_n=req.top_n, max_pages=req.max_pages)
        elif req.kind == "author":
            fn = lambda: cad.ad_author_ranked(top_n=req.top_n, max_pages=req.max_pages)
        else:
            raise HTTPException(400, "kind 必须是 tag 或 author")
        results = await asyncio.to_thread(fn)
    return {"groups": cad.build_view(results)}


class CommitItem(BaseModel):
    id: str
    hit_tags: list[str] = []
    added: bool = False


class CommitReq(BaseModel):
    selected_ids: list[str]
    all_items: list[CommitItem] = []


@app.post("/api/commit")
async def commit(req: CommitReq):
    cad = get_cad()
    async with _lock:
        def _work():
            selected_set = set(req.selected_ids)
            # 反馈（与 CLI 一致：已收藏且未选不记 skipped）
            if cad.config.feedback.enabled:
                with cad._lock:
                    for it in req.all_items:
                        if not it.hit_tags:
                            continue
                        if it.id in selected_set:
                            target = cad.selected_tags
                        elif not it.added:
                            target = cad.skipped_tags
                        else:
                            continue
                        for t in it.hit_tags:
                            target[t] += 1
            # 黑名单本地兜底
            if cad.blacklist_tags and req.selected_ids:
                kept, meta = cad._filter_blacklisted_with_meta(req.selected_ids)
            else:
                kept, meta = req.selected_ids, {}
            if kept:
                cad.add(kept, preset_meta=meta)
                cad.save()
            return {
                "requested": len(req.selected_ids),
                "added": len(kept),
                "blocked": len(req.selected_ids) - len(kept),
            }
        return await asyncio.to_thread(_work)


@app.get("/api/memo")
async def memo():
    cad = get_cad()
    return {
        "added": sorted(cad.added),
        "title_versions": {k: sorted(v) for k, v in cad.added_title_versions.items()},
        "fav_tag": dict(cad.fav_tag.most_common()),
        "fav_author": dict(cad.fav_author.most_common()),
        "selected_tags": dict(cad.selected_tags),
        "skipped_tags": dict(cad.skipped_tags),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ComicAdv</title>
<style>
  :root{
    --bg:#0f1115;--panel:#171a21;--panel2:#1f232c;--border:#2a2f3a;
    --text:#e6e8ee;--muted:#8b93a7;--accent:#4f8cff;--ok:#26c281;
  }
  *{box-sizing:border-box;}
  body{margin:0;font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       background:var(--bg);color:var(--text);}
  header{padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--border);
         display:flex;gap:16px;align-items:center;flex-wrap:wrap;}
  header h1{margin:0;font-size:16px;font-weight:600;}
  .badge{background:var(--panel2);padding:2px 8px;border-radius:10px;color:var(--muted);font-size:12px;}
  main{padding:16px 20px 100px;}
  .toolbar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;}
  button{background:var(--panel2);color:var(--text);border:1px solid var(--border);
         border-radius:6px;padding:6px 14px;cursor:pointer;font-size:14px;}
  button:hover{background:#262b35;}
  button.primary{background:var(--accent);border-color:var(--accent);}
  button.primary:hover{background:#3d7cf0;}
  button.success{background:var(--ok);border-color:var(--ok);color:#06281a;}
  button:disabled{opacity:.5;cursor:not-allowed;}
  .group{background:var(--panel);border:1px solid var(--border);border-radius:8px;
         margin-bottom:16px;overflow:hidden;}
  .group h2{margin:0;padding:10px 14px;font-size:14px;font-weight:600;
            background:var(--panel2);border-bottom:1px solid var(--border);
            display:flex;justify-content:space-between;}
  table{width:100%;border-collapse:collapse;}
  th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top;}
  th{background:var(--panel2);color:var(--muted);font-weight:500;font-size:12px;}
  tr:last-child td{border-bottom:none;}
  tr.added td{color:var(--muted);}
  .title{word-break:break-all;}
  .hit{color:var(--muted);font-size:12px;}
  .score{font-variant-numeric:tabular-nums;color:var(--ok);text-align:right;}
  input[type=checkbox]{width:16px;height:16px;accent-color:var(--accent);}
  .footer{position:fixed;bottom:0;left:0;right:0;background:var(--panel);
          border-top:1px solid var(--border);padding:10px 20px;display:flex;gap:12px;
          align-items:center;}
  .count{color:var(--muted);}
  .spacer{flex:1;}
  .toast{position:fixed;bottom:80px;left:50%;transform:translateX(-50%) translateY(20px);
         background:var(--panel2);border:1px solid var(--border);padding:8px 16px;
         border-radius:8px;opacity:0;transition:all .2s;pointer-events:none;max-width:80vw;}
  .toast.show{opacity:1;transform:translateX(-50%) translateY(0);}
  .spin{display:inline-block;width:14px;height:14px;border:2px solid var(--muted);
        border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;
        vertical-align:middle;margin-right:6px;}
  @keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>
<header>
  <h1>📚 ComicAdv</h1>
  <span class="badge" id="stat">加载中…</span>
</header>
<main>
  <div class="toolbar">
    <button class="primary" id="btnTag">🔍 按 Tag 搜索</button>
    <button class="primary" id="btnAuthor">✍️ 按作者搜索</button>
    <button id="btnReload">🔄 刷新状态</button>
  </div>
  <div id="results"></div>
</main>
<div class="footer">
  <label><input type="checkbox" id="selAll"> 全选</label>
  <span class="count" id="count">0 条已选择</span>
  <span class="spacer"></span>
  <button class="success" id="btnCommit" disabled>✅ 提交（添加选中）</button>
</div>
<div class="toast" id="toast"></div>
<script>
const $ = id => document.getElementById(id);
let currentGroups = [];
let selected = new Set();

function toast(msg, ms=2000) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove('show'), ms);
}
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
function escapeHTML(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function loadStatus() {
  try {
    const s = await fetchJSON('/api/status');
    const tags = s.effective_tags.slice(0,5).map(t => t[0]).join(', ');
    $('stat').textContent =
      `${s.added_count} 已收藏 · 模式 ${s.mode} · TopTags: ${tags || '—'} · 黑名单 ${s.blacklist.length}`;
  } catch (e) {
    $('stat').textContent = '加载失败: ' + e.message;
  }
}

function renderGroups(groups) {
  currentGroups = groups;
  selected.clear();
  updateCount();
  const root = $('results');
  if (!groups.length) {
    root.innerHTML = '<p style="color:var(--muted)">未找到结果</p>';
    return;
  }
  root.innerHTML = groups.map(g => `
    <div class="group">
      <h2><span>${escapeHTML(g.combo)}</span><span>${g.items.length} 条</span></h2>
      <table>
        <thead><tr>
          <th style="width:36px"></th>
          <th style="width:70px">ID</th>
          <th>标题</th>
          <th style="width:180px">命中</th>
          <th style="width:70px">分数</th>
        </tr></thead>
        <tbody>
          ${g.items.map(it => `
            <tr data-id="${escapeHTML(it.id)}" class="${it.added?'added':''}">
              <td><input type="checkbox" data-id="${escapeHTML(it.id)}" ${it.added?'disabled':''}></td>
              <td>${escapeHTML(it.id)}</td>
              <td class="title">${escapeHTML(it.title)}${it.added?' <span class="hit">(已收藏)</span>':''}</td>
              <td class="hit">${escapeHTML(it.hit_tags.slice(0,4).join(', '))}</td>
              <td class="score">${it.score != null ? it.score.toFixed(2) : ''}</td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`).join('');

  root.querySelectorAll('input[type=checkbox][data-id]').forEach(cb => {
    cb.addEventListener('change', e => {
      const id = e.target.dataset.id;
      if (e.target.checked) selected.add(id); else selected.delete(id);
      updateCount();
    });
  });
}

function updateCount() {
  $('count').textContent = `${selected.size} 条已选择`;
  $('btnCommit').disabled = selected.size === 0;
}

async function doSearch(kind) {
  const btn = kind === 'tag' ? $('btnTag') : $('btnAuthor');
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>搜索中…';
  try {
    const r = await fetchJSON('/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({kind}),
    });
    renderGroups(r.groups);
    const total = r.groups.reduce((a, g) => a + g.items.length, 0);
    toast(`搜索完成，共 ${total} 条`);
  } catch (e) {
    toast('搜索失败: ' + e.message, 4000);
  } finally {
    btn.disabled = false;
    btn.innerHTML = old;
  }
}

async function commit() {
  const addedIds = new Set(selected);
  const all_items = [];
  currentGroups.forEach(g => g.items.forEach(it => {
    all_items.push({id: it.id, hit_tags: it.hit_tags, added: it.added});
  }));
  const btn = $('btnCommit');
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = '提交中…';
  try {
    const r = await fetchJSON('/api/commit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({selected_ids: [...addedIds], all_items}),
    });
    toast(`请求 ${r.requested} 条，实际添加 ${r.added} 条`
      + (r.blocked ? `，黑名单拦截 ${r.blocked} 条` : ''));
    // 本地标记已添加项
    currentGroups.forEach(g => g.items.forEach(it => {
      if (addedIds.has(it.id)) it.added = true;
    }));
    renderGroups(currentGroups);
    await loadStatus();
  } catch (e) {
    toast('提交失败: ' + e.message, 4000);
  } finally {
    btn.textContent = old;
    updateCount();
  }
}

$('btnTag').addEventListener('click', () => doSearch('tag'));
$('btnAuthor').addEventListener('click', () => doSearch('author'));
$('btnReload').addEventListener('click', loadStatus);
$('btnCommit').addEventListener('click', commit);
$('selAll').addEventListener('change', e => {
  const on = e.target.checked;
  document.querySelectorAll('input[type=checkbox][data-id]:not(:disabled)').forEach(cb => {
    cb.checked = on;
    const id = cb.dataset.id;
    if (on) selected.add(id); else selected.delete(id);
  });
  updateCount();
});

loadStatus();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)