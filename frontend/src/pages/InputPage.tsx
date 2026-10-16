import { useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../api/client'
import { listGames, getGame, createGame } from '../api/games'
import { runAnalysis } from '../api/analysis'
import type { AnalysisResult, GameSummary } from '../api/types'
import { parseTable, toCsv } from '../input/csv'
import {
  SYSTEMS, SYSTEM_BY_KEY, blankRow, buildGame, checkTables, emptyTables, gameToTables, mapHeaders,
  type Issue, type Row, type SystemKey, type Tables,
} from '../input/schema'

const STORAGE_KEY = 'idleframework.input.v1'

interface ServerCheck { ok: boolean; errors: { loc: string; msg: string }[]; warnings: { loc: string; msg: string }[] }

function loadDraft() {
  try { const raw = localStorage.getItem(STORAGE_KEY); if (raw) return JSON.parse(raw) } catch { /* ignore */ }
  return null
}

export default function InputPage() {
  const draft = useMemo(loadDraft, [])
  const [name, setName] = useState<string>(draft?.name ?? '我的放置游戏')
  const [tables, setTables] = useState<Tables>(draft?.tables ?? emptyTables())
  const [extra, setExtra] = useState<{ nodes: unknown[]; edges: unknown[] }>(draft?.extra ?? { nodes: [], edges: [] })
  const [active, setActive] = useState<SystemKey>('resources')
  const [games, setGames] = useState<GameSummary[]>([])
  const [importText, setImportText] = useState('')
  const [importMode, setImportMode] = useState<'append' | 'replace'>('replace')
  const [importMsg, setImportMsg] = useState<string>('')
  const [server, setServer] = useState<ServerCheck | null>(null)
  const [simTime, setSimTime] = useState(1800)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [runErr, setRunErr] = useState('')

  useEffect(() => { listGames().then((r) => setGames(r.games)).catch(() => {}) }, [])
  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify({ name, tables, extra })) }, [name, tables, extra])

  const issues = useMemo(() => checkTables(tables), [tables])
  const errors = issues.filter((i) => i.level === 'error')
  const sys = SYSTEM_BY_KEY[active]
  const rows = tables[active]
  const cellIssue = (r: number, c: string) => issues.find((i) => i.system === active && i.row === r && i.col === c)

  const setRows = (k: SystemKey, rs: Row[]) => { setTables((t) => ({ ...t, [k]: rs })); setServer(null) }
  const updateCell = (r: number, key: string, v: string) => setRows(active, rows.map((row, i) => (i === r ? { ...row, [key]: v } : row)))

  function doImport(text: string) {
    const grid = parseTable(text)
    if (grid.length < 2) { setImportMsg('至少需要表头一行 + 数据一行'); return }
    const keys = mapHeaders(sys, grid[0])
    const unknown = grid[0].filter((_, i) => !keys[i])
    if (!keys.includes('id')) { setImportMsg(`表头里没找到 ID 列。可用表头：${sys.cols.map((c) => c.label).join('、')}`); return }
    const imported = grid.slice(1).map((cells) => {
      const row = blankRow(sys)
      keys.forEach((k, i) => { if (k) row[k] = (cells[i] ?? '').trim() })
      return row
    })
    setRows(active, importMode === 'append' ? [...rows, ...imported] : imported)
    setImportMsg(`已导入 ${imported.length} 行到「${sys.label}」` + (unknown.length ? `；忽略未识别列：${unknown.join('、')}` : ''))
    setImportText('')
  }

  async function onFile(f: File | undefined) {
    if (!f) return
    if (/\.(xlsx|xls)$/i.test(f.name)) { setImportMsg('暂不直接读取 .xlsx：请在 Excel 里选中表格区域复制，粘贴到下方文本框；或另存为 CSV(UTF-8)。'); return }
    const text = await f.text()
    if (/\.json$/i.test(f.name)) { loadGameJson(JSON.parse(text)); return }
    doImport(text)
  }

  function loadGameJson(g: unknown) {
    const r = gameToTables(g)
    setName(r.name); setTables(r.tables); setExtra(r.extra); setServer(null); setResult(null)
    setImportMsg(`已载入「${r.name}」` + (r.extra.nodes.length ? `；${r.extra.nodes.length} 个表格暂不支持的节点已原样保留（导出 JSON / 保存时一并带上）` : ''))
  }

  function exportCsv() {
    const data = [sys.cols.map((c) => c.label), ...rows.map((r) => sys.cols.map((c) => r[c.key] ?? ''))]
    download(`${active}.csv`, '\uFEFF' + toCsv(data), 'text/csv')
  }

  const game = () => buildGame(name, tables, extra)

  async function serverValidate(): Promise<ServerCheck> {
    const r = await apiFetch<ServerCheck>('/games/validate', { method: 'POST', body: JSON.stringify(game()) })
    setServer(r); return r
  }

  async function saveAndRun() {
    setBusy(true); setRunErr(''); setResult(null)
    try {
      const v = await serverValidate()
      if (!v.ok) { setRunErr('服务器校验未通过，见上方错误'); return }
      const saved = await createGame(game())
      const r = await runAnalysis({ game_id: saved.id, simulation_time: simTime, optimizer: 'greedy' })
      setResult(r)
      listGames().then((x) => setGames(x.games)).catch(() => {})
    } catch (e) {
      setRunErr(e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }

  return (
    <div className="space-y-4" data-testid="input-page">
      <div className="flex flex-wrap items-end gap-3 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm">
        <label className="text-sm">游戏名称<br />
          <input className="border rounded px-2 py-1 w-56 dark:bg-gray-700" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="text-sm">载入已有游戏<br />
          <select className="border rounded px-2 py-1 dark:bg-gray-700" value="" onChange={async (e) => { if (e.target.value) loadGameJson(await getGame(e.target.value)) }}>
            <option value="">— 选择 —</option>
            {games.map((g) => <option key={g.id} value={g.id}>{g.name}{g.bundled ? '（示例）' : ''}</option>)}
          </select>
        </label>
        <button className="btn" onClick={() => { if (confirm('清空所有表格？')) { setTables(emptyTables()); setExtra({ nodes: [], edges: [] }); setResult(null) } }}>新建空白</button>
        <button className="btn" onClick={() => download(`${name}.json`, JSON.stringify(game(), null, 2), 'application/json')}>导出 JSON</button>
        <span className="ml-auto text-sm">
          {errors.length ? <span className="text-red-600">❌ {errors.length} 个错误</span> : <span className="text-green-600">✓ 表格检查通过</span>}
          {issues.length - errors.length > 0 && <span className="text-amber-600 ml-2">⚠ {issues.length - errors.length} 个提醒</span>}
        </span>
      </div>

      <div className="flex gap-1 flex-wrap">
        {SYSTEMS.map((s) => {
          const n = issues.filter((i) => i.system === s.key && i.level === 'error').length
          return (
            <button key={s.key} onClick={() => setActive(s.key)}
              className={`px-3 py-1.5 rounded-t-md text-sm border-b-2 ${active === s.key ? 'border-blue-600 bg-white dark:bg-gray-800 font-semibold' : 'border-transparent text-gray-500'}`}>
              {s.label} <span className="text-xs text-gray-400">({tables[s.key].length})</span>{n > 0 && <span className="ml-1 text-red-600 text-xs">●{n}</span>}
            </button>
          )
        })}
      </div>

      <section className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm space-y-3">
        <p className="text-sm text-gray-600 dark:text-gray-300">{sys.desc}</p>
        <div className="overflow-x-auto">
          <table className="text-sm border-collapse min-w-full" data-testid="system-table">
            <thead>
              <tr>{sys.cols.map((c) => (
                <th key={c.key} title={c.hint} className="text-left px-2 py-1 bg-gray-100 dark:bg-gray-700 border whitespace-nowrap">
                  {c.label}{c.required && <span className="text-red-500">*</span>}
                  {c.hint && <div className="font-normal text-xs text-gray-400">{c.hint}</div>}
                </th>))}
                <th className="bg-gray-100 dark:bg-gray-700 border" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row, r) => (
                <tr key={r}>
                  {sys.cols.map((c) => {
                    const iss = cellIssue(r, c.key)
                    const cls = `w-full px-1 py-0.5 bg-transparent outline-none ${iss?.level === 'error' ? 'bg-red-50 dark:bg-red-900/30' : iss ? 'bg-amber-50 dark:bg-amber-900/30' : ''}`
                    return (
                      <td key={c.key} className="border px-1" title={iss?.msg}>
                        {c.kind === 'select'
                          ? <select className={cls} value={row[c.key] ?? ''} onChange={(e) => updateCell(r, c.key, e.target.value)}>
                              {c.options!.map((o) => <option key={o}>{o}</option>)}
                            </select>
                          : <input className={`${cls} ${c.kind === 'number' ? 'text-right font-mono' : ''}`} value={row[c.key] ?? ''} onChange={(e) => updateCell(r, c.key, e.target.value)} />}
                      </td>
                    )
                  })}
                  <td className="border px-1"><button className="text-red-500 text-xs" onClick={() => setRows(active, rows.filter((_, i) => i !== r))}>删除</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex gap-2">
          <button className="btn" onClick={() => setRows(active, [...rows, blankRow(sys)])}>+ 添加一行</button>
          <button className="btn" onClick={exportCsv}>导出本表 CSV</button>
        </div>

        <details open={rows.length === 0} className="border rounded p-3">
          <summary className="cursor-pointer text-sm font-semibold">从表格导入（Excel 复制粘贴 / CSV 文件）</summary>
          <p className="text-xs text-gray-500 my-2">第一行必须是表头。可识别表头：{sys.cols.map((c) => [c.label, ...(c.aliases ?? [])].join('/')).join('；')}</p>
          <textarea data-testid="import-text" className="w-full h-28 border rounded p-2 font-mono text-xs dark:bg-gray-700" placeholder={sys.cols.map((c) => c.label).join('\t')} value={importText} onChange={(e) => setImportText(e.target.value)} />
          <div className="flex items-center gap-3 mt-2 text-sm">
            <label><input type="radio" checked={importMode === 'replace'} onChange={() => setImportMode('replace')} /> 替换本表</label>
            <label><input type="radio" checked={importMode === 'append'} onChange={() => setImportMode('append')} /> 追加</label>
            <button className="btn" onClick={() => doImport(importText)}>导入粘贴内容</button>
            <label className="btn cursor-pointer">选择 CSV / JSON 文件<input type="file" accept=".csv,.tsv,.txt,.json,.xlsx" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} /></label>
          </div>
        </details>
        {importMsg && <p className="text-sm text-blue-700 dark:text-blue-300" data-testid="import-msg">{importMsg}</p>}
      </section>

      <IssueList issues={issues} server={server} onJump={(k) => setActive(k)} extraCount={extra.nodes.length} />

      <section className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm space-y-3">
        <div className="flex items-end gap-3">
          <label className="text-sm">模拟时长（秒）<br />
            <input type="number" className="border rounded px-2 py-1 w-32 dark:bg-gray-700" value={simTime} onChange={(e) => setSimTime(Number(e.target.value))} />
          </label>
          <button className="btn" onClick={() => serverValidate().catch((e) => setRunErr(String(e)))}>服务器校验</button>
          <button className="btn btn-primary" disabled={busy || errors.length > 0} onClick={saveAndRun}>{busy ? '运行中…' : '保存并模拟'}</button>
          {runErr && <span className="text-red-600 text-sm">{runErr}</span>}
        </div>
        {result && <ResultView r={result} simTime={simTime} />}
      </section>
    </div>
  )
}

function IssueList({ issues, server, onJump, extraCount }: { issues: Issue[]; server: ServerCheck | null; onJump: (k: SystemKey) => void; extraCount: number }) {
  if (!issues.length && !server && !extraCount) return null
  return (
    <section className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm text-sm space-y-1" data-testid="issues">
      <h3 className="font-semibold">校验结果</h3>
      {issues.map((i, k) => (
        <div key={k} className={i.level === 'error' ? 'text-red-600' : 'text-amber-600'}>
          {i.level === 'error' ? '❌' : '⚠'}{' '}
          {i.system && <button className="underline" onClick={() => onJump(i.system!)}>{SYSTEM_BY_KEY[i.system].label}</button>}
          {i.row !== undefined && ` 第${i.row + 1}行`}：{i.msg}
        </div>
      ))}
      {extraCount > 0 && <div className="text-gray-500">ℹ 高级：另有 {extraCount} 个表格未覆盖的节点（如增益、协同、成就）随 JSON 原样保存。</div>}
      {server && (server.ok
        ? <div className="text-green-600">✓ 服务器校验通过（与 Python 引擎同一套规则）</div>
        : server.errors.map((e, k) => <div key={k} className="text-red-600">❌ 服务器：{e.loc && <code>{e.loc}</code>} {e.msg}</div>))}
      {server?.warnings.map((w, k) => <div key={k} className="text-amber-600">⚠ 服务器：{w.msg}</div>)}
    </section>
  )
}

function fmt(n: number) { return Math.abs(n) >= 1e6 ? n.toExponential(2) : n.toLocaleString('zh-CN', { maximumFractionDigits: 2 }) }

function ResultView({ r, simTime }: { r: AnalysisResult; simTime: number }) {
  const opt = r.optimizer_result
  const firstBuy = new Map<string, number>()
  opt?.purchases.forEach((p) => { if (!firstBuy.has(p.node_id)) firstBuy.set(p.node_id, p.time) })
  const truncated = opt && opt.final_time < simTime - 1
  return (
    <div className="space-y-3 text-sm" data-testid="result">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="最终产量/秒" v={opt ? fmt(opt.final_production) : '-'} />
        <Stat label="购买次数" v={opt ? String(opt.purchases.length) : '-'} />
        <Stat label="模拟到" v={opt ? `${fmt(opt.final_time)} 秒` : '-'} />
        <Stat label="主导生产者" v={r.dominant_strategy?.dominant_gen ? `${r.dominant_strategy.dominant_gen} (${fmt(r.dominant_strategy.ratio)}x)` : '无'} />
      </div>
      {truncated && <div className="text-amber-600">⚠ 模拟在 {fmt(opt!.final_time)} 秒提前结束：引擎有 500 次购买上限（已知限制），结果只覆盖前半段。</div>}
      {r.dead_upgrades.length > 0 && <div><b>废升级：</b>{r.dead_upgrades.map((d) => `${d.upgrade_id}（${d.reason}）`).join('；')}</div>}
      {r.progression_walls.length > 0 && <div><b>进度墙（仅按成长率规则判断）：</b>{r.progression_walls.map((w) => w.reason).join('；')}</div>}
      <div>
        <b>各项首次购买时间（贪心玩家）</b>
        <table className="mt-1 border-collapse">
          <thead><tr><th className="border px-2 bg-gray-100 dark:bg-gray-700">节点</th><th className="border px-2 bg-gray-100 dark:bg-gray-700">首次购买(秒)</th></tr></thead>
          <tbody>{[...firstBuy.entries()].map(([id, t]) => <tr key={id}><td className="border px-2">{id}</td><td className="border px-2 text-right font-mono">{fmt(t)}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="text-xs text-gray-500">说明：当前优化器模拟的是“贪心玩家的购买顺序”，不会自动修改你的数值；转生不会被自动触发。</p>
    </div>
  )
}

function Stat({ label, v }: { label: string; v: string }) {
  return <div className="border rounded p-2"><div className="text-xs text-gray-500">{label}</div><div className="font-mono text-base">{v}</div></div>
}

function download(fname: string, content: string, mime: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([content], { type: mime }))
  a.download = fname
  a.click()
  URL.revokeObjectURL(a.href)
}
