/**
 * Table-based input model for the Chinese input editor.
 * Each "system" is a table; rows are converted into IdleFramework GameDefinition nodes/edges.
 */
export type ColKind = 'text' | 'number' | 'select' | 'list' | 'bool' | 'price' | 'reqs'

export interface ColDef {
  key: string
  label: string
  kind: ColKind
  required?: boolean
  options?: string[]
  aliases?: string[] // extra accepted header names for import
  hint?: string
  default?: string
}

export interface SystemDef {
  key: SystemKey
  label: string
  desc: string
  cols: ColDef[]
}

export type SystemKey = 'resources' | 'generators' | 'upgrades' | 'prestige' | 'autobuyers' | 'groups'
export type Row = Record<string, string>
export type Tables = Record<SystemKey, Row[]>

const ID: ColDef = { key: 'id', label: 'ID', kind: 'text', required: true, aliases: ['编号', 'key', '键'], hint: '英文/数字/下划线，全局唯一' }
const NAME: ColDef = { key: 'name', label: '名称', kind: 'text', aliases: ['name', '名字'] }

export const SYSTEMS: SystemDef[] = [
  {
    key: 'resources', label: '资源/货币', desc: '金币、钻石、转生货币等。第一行资源是默认货币。',
    cols: [ID, NAME,
      { key: 'initial_value', label: '初始值', kind: 'number', default: '0', aliases: ['初始', 'initial'] },
      { key: 'capacity', label: '上限', kind: 'number', aliases: ['cap', '容量'], hint: '留空=无上限' }],
  },
  {
    key: 'generators', label: '生产者', desc: '矿工、工厂等。价格填数字 = 用它产出的资源购买；填「货币:数量; 货币:数量」= 多货币价格（如 gold:400; diamond:5）。',
    cols: [ID, NAME,
      { key: 'produces', label: '产出资源', kind: 'text', required: true, aliases: ['产出', 'resource', 'target'], hint: '资源ID' },
      { key: 'base_production', label: '基础产量/周期', kind: 'number', required: true, aliases: ['基础产量', '产量', 'production'] },
      { key: 'cycle_time', label: '周期(秒)', kind: 'number', default: '1', aliases: ['周期', 'cycle'] },
      { key: 'cost_base', label: '基础价格', kind: 'price', required: true, aliases: ['价格', '初始价格', 'cost'], hint: '数字，或 货币:数量; 货币:数量（多货币，每项都按成长率涨价）' },
      { key: 'cost_growth_rate', label: '价格成长率', kind: 'number', required: true, default: '1.15', aliases: ['成长率', '增长率', 'growth'], hint: '第n个价格=基础价格×成长率^(n-1)' },
      { key: 'requires', label: '购买门槛', kind: 'reqs', aliases: ['门槛', '需求', 'requires'], hint: '货币>=数量; …  只检查不消耗' }],
  },
  {
    key: 'upgrades', label: '升级/倍率', desc: '一次性购买的倍率加成。target 填生产者ID，或 _all 表示全体。',
    cols: [ID, NAME,
      { key: 'target', label: '作用目标', kind: 'text', required: true, aliases: ['目标'], hint: '生产者ID 或 _all' },
      { key: 'upgrade_type', label: '类型', kind: 'select', required: true, options: ['multiplicative', 'additive', 'percentage'], default: 'multiplicative', aliases: ['类型'], hint: 'multiplicative=乘算' },
      { key: 'magnitude', label: '数值', kind: 'number', required: true, aliases: ['倍率', '加成'] },
      { key: 'cost', label: '价格', kind: 'price', required: true, aliases: ['价格'], hint: '数字=默认货币；或 gold:15000; diamond:10' },
      { key: 'requires', label: '购买门槛', kind: 'reqs', aliases: ['门槛', '需求', 'requires'], hint: '如 honor>=100（只检查不消耗）' },
      { key: 'stacking_group', label: '叠加组', kind: 'text', required: true, default: 'upgrades', aliases: ['叠加', 'group'] }],
  },
  {
    key: 'prestige', label: '转生层', desc: '重置并获得转生货币。公式可用变量：lifetime_earnings、lifetime_<资源ID>、owned_<生产者ID> 等。',
    cols: [ID, NAME,
      { key: 'formula_expr', label: '收益公式', kind: 'text', required: true, aliases: ['公式', 'formula'], hint: '例：150*sqrt(lifetime_earnings/1e15)' },
      { key: 'currency_id', label: '获得货币', kind: 'text', aliases: ['货币'], hint: '资源ID' },
      { key: 'layer_index', label: '层级', kind: 'number', default: '0', aliases: ['层'] },
      { key: 'reset_scope', label: '重置项', kind: 'list', required: true, aliases: ['重置'], hint: '逗号分隔ID' },
      { key: 'persistence_scope', label: '保留项', kind: 'list', aliases: ['保留'], hint: '逗号分隔ID' }],
  },
  {
    key: 'autobuyers', label: '自动购买', desc: '定时自动购买某个生产者/升级。',
    cols: [ID, NAME,
      { key: 'target', label: '购买目标', kind: 'text', required: true, aliases: ['目标'] },
      { key: 'interval', label: '间隔(秒)', kind: 'number', default: '1', aliases: ['间隔'] },
      { key: 'bulk_amount', label: '每次数量', kind: 'select', options: ['1', '10', 'max'], default: '1', aliases: ['数量'] },
      { key: 'condition', label: '条件公式', kind: 'text', aliases: ['条件'], hint: '可留空；>0 时执行' }],
  },
  {
    key: 'groups', label: '叠加组', desc: '同组内的升级如何合并：乘算/加算/百分比。',
    cols: [{ ...ID, label: '组ID' },
      { key: 'mode', label: '合并方式', kind: 'select', required: true, options: ['multiplicative', 'additive', 'percentage'], default: 'multiplicative', aliases: ['方式'] }],
  },
]

export const SYSTEM_BY_KEY = Object.fromEntries(SYSTEMS.map((s) => [s.key, s])) as Record<SystemKey, SystemDef>

export function emptyTables(): Tables {
  return { resources: [], generators: [], upgrades: [], prestige: [], autobuyers: [], groups: [] }
}

export function blankRow(sys: SystemDef): Row {
  return Object.fromEntries(sys.cols.map((c) => [c.key, c.default ?? '']))
}

export interface Issue { system?: SystemKey; row?: number; col?: string; msg: string; level: 'error' | 'warning' }

const ID_RE = /^[A-Za-z_][A-Za-z0-9_]*$/

/** Client-side checks with row/column positions (server validation runs too). */
export function checkTables(t: Tables): Issue[] {
  const issues: Issue[] = []
  const seen = new Map<string, string>()
  for (const sys of SYSTEMS) {
    t[sys.key].forEach((row, i) => {
      for (const c of sys.cols) {
        const v = (row[c.key] ?? '').trim()
        if (c.required && v === '') issues.push({ system: sys.key, row: i, col: c.key, msg: `「${c.label}」必填`, level: 'error' })
        if (c.kind === 'number' && v !== '' && !Number.isFinite(Number(v))) issues.push({ system: sys.key, row: i, col: c.key, msg: `「${c.label}」不是数字：${v}`, level: 'error' })
        if (c.kind === 'select' && v !== '' && c.options && !c.options.includes(v)) issues.push({ system: sys.key, row: i, col: c.key, msg: `「${c.label}」只能是 ${c.options.join(' / ')}`, level: 'error' })
        if (c.kind === 'price' || c.kind === 'reqs') {
          const p = c.kind === 'price' ? parsePrice(v) : parseReqs(v)
          if (p.error) issues.push({ system: sys.key, row: i, col: c.key, msg: `「${c.label}」${p.error}`, level: 'error' })
          for (const part of p.parts) if (!(t.resources ?? []).some((r) => r.id.trim() === part.currency)) issues.push({ system: sys.key, row: i, col: c.key, msg: `「${c.label}」里的货币「${part.currency}」不是资源`, level: 'error' })
        }
      }
      const id = (row.id ?? '').trim()
      if (id && !ID_RE.test(id)) issues.push({ system: sys.key, row: i, col: 'id', msg: `ID「${id}」只能用英文字母、数字、下划线，且不能以数字开头`, level: 'error' })
      if (id && sys.key !== 'groups') {
        if (seen.has(id)) issues.push({ system: sys.key, row: i, col: 'id', msg: `ID「${id}」与「${seen.get(id)}」重复`, level: 'error' })
        else seen.set(id, sys.label)
      }
    })
  }
  const resIds = new Set(t.resources.map((r) => r.id.trim()))
  const genIds = new Set(t.generators.map((r) => r.id.trim()))
  const allIds = new Set(seen.keys())
  const groupIds = new Set(t.groups.map((r) => r.id.trim()))
  if (t.resources.length === 0) issues.push({ system: 'resources', msg: '至少需要一个资源', level: 'error' })
  t.generators.forEach((r, i) => {
    if (r.produces && !resIds.has(r.produces.trim())) issues.push({ system: 'generators', row: i, col: 'produces', msg: `产出资源「${r.produces}」不存在`, level: 'error' })
    const g = Number(r.cost_growth_rate)
    if (Number.isFinite(g) && g <= 1) issues.push({ system: 'generators', row: i, col: 'cost_growth_rate', msg: '成长率 ≤ 1：价格不会上涨', level: 'warning' })
    if (Number.isFinite(g) && g >= 1.3) issues.push({ system: 'generators', row: i, col: 'cost_growth_rate', msg: '成长率 ≥ 1.3：分析器会判为进度墙', level: 'warning' })
  })
  t.upgrades.forEach((r, i) => {
    const tg = r.target.trim()
    if (tg && tg !== '_all' && !allIds.has(tg)) issues.push({ system: 'upgrades', row: i, col: 'target', msg: `作用目标「${tg}」不存在`, level: 'error' })
    else if (tg && tg !== '_all' && !genIds.has(tg)) issues.push({ system: 'upgrades', row: i, col: 'target', msg: `目标「${tg}」不是生产者，引擎可能不生效`, level: 'warning' })
    if (r.stacking_group && !groupIds.has(r.stacking_group.trim())) issues.push({ system: 'upgrades', row: i, col: 'stacking_group', msg: `叠加组「${r.stacking_group}」未在「叠加组」表定义（保存时会自动按乘算补上）`, level: 'warning' })
  })
  t.prestige.forEach((r, i) => {
    for (const key of ['reset_scope', 'persistence_scope'] as const)
      splitList(r[key]).forEach((x) => { if (!allIds.has(x)) issues.push({ system: 'prestige', row: i, col: key, msg: `「${x}」不存在`, level: 'error' }) })
    if (r.currency_id && !resIds.has(r.currency_id.trim())) issues.push({ system: 'prestige', row: i, col: 'currency_id', msg: `货币「${r.currency_id}」不是资源`, level: 'error' })
  })
  t.autobuyers.forEach((r, i) => {
    if (r.target && !allIds.has(r.target.trim())) issues.push({ system: 'autobuyers', row: i, col: 'target', msg: `目标「${r.target}」不存在`, level: 'error' })
  })
  return issues
}

export interface PricePart { currency: string; amount: number }

/** Parse a price cell: plain number → null (legacy single price); "gold:10; gem:2" → parts. */
export function parsePrice(v: string | undefined): { legacy: number | null; parts: PricePart[]; error?: string } {
  const txt = (v ?? '').trim()
  if (txt === '') return { legacy: null, parts: [] }
  if (Number.isFinite(Number(txt))) return { legacy: Number(txt), parts: [] }
  return parseParts(txt, /[:：=]/)
}

/** Parse a requirement cell: "honor>=100; gem:5". */
export function parseReqs(v: string | undefined) {
  return parseParts((v ?? '').trim(), /\s*(?:>=|≥|[:：=])\s*/)
}

function parseParts(txt: string, sep: RegExp): { legacy: null; parts: PricePart[]; error?: string } {
  const parts: PricePart[] = []
  if (txt === '') return { legacy: null, parts }
  for (const item of txt.split(/[;；+]+/).map((x) => x.trim()).filter(Boolean)) {
    const m = item.split(sep).map((x) => x.trim()).filter(Boolean)
    if (m.length !== 2 || !Number.isFinite(Number(m[1])) || Number(m[1]) < 0)
      return { legacy: null, parts, error: `无法识别「${item}」，应写成 货币ID:数量` }
    parts.push({ currency: m[0], amount: Number(m[1]) })
  }
  return { legacy: null, parts }
}

const fmtParts = (parts: { currency: string; amount: number }[], sep: string) => parts.map((p) => `${p.currency}${sep}${p.amount}`).join('; ')

export function splitList(v: string | undefined): string[] {
  return (v ?? '').split(/[,，;；\s]+/).map((s) => s.trim()).filter(Boolean)
}

const num = (v: string | undefined, d?: number) => (v === undefined || v.trim() === '' ? d : Number(v))

/** Convert tables → GameDefinition JSON. `extraNodes` keeps node types the tables don't cover. */
export function buildGame(name: string, t: Tables, extra: { nodes: unknown[]; edges: unknown[] } = { nodes: [], edges: [] }) {
  const nodes: Record<string, unknown>[] = []
  const edges: Record<string, unknown>[] = []
  const groups: Record<string, string> = {}
  t.groups.forEach((g) => { if (g.id.trim()) groups[g.id.trim()] = g.mode || 'multiplicative' })
  t.upgrades.forEach((u) => { const g = u.stacking_group?.trim(); if (g && !groups[g]) groups[g] = 'multiplicative' })
  for (const r of t.resources) nodes.push({ id: r.id.trim(), type: 'resource', name: r.name || r.id, initial_value: num(r.initial_value, 0), ...(r.capacity?.trim() ? { capacity: num(r.capacity) } : {}) })
  const priceFields = (key: 'cost_base' | 'cost', v: string | undefined) => {
    const p = parsePrice(v)
    return p.parts.length ? { cost: p.parts } : { [key]: p.legacy ?? undefined }
  }
  const reqFields = (v: string | undefined) => {
    const p = parseReqs(v)
    return p.parts.length ? { requires: p.parts } : {}
  }
  for (const r of t.generators) {
    nodes.push({ id: r.id.trim(), type: 'generator', name: r.name || r.id, base_production: num(r.base_production), ...priceFields('cost_base', r.cost_base), cost_growth_rate: num(r.cost_growth_rate), cycle_time: num(r.cycle_time, 1), ...reqFields(r.requires) })
    if (r.produces?.trim()) edges.push({ id: `e_${r.id.trim()}_${r.produces.trim()}`, source: r.id.trim(), target: r.produces.trim(), edge_type: 'production_target' })
  }
  for (const r of t.upgrades) nodes.push({ id: r.id.trim(), type: 'upgrade', name: r.name || r.id, upgrade_type: r.upgrade_type || 'multiplicative', magnitude: num(r.magnitude), ...priceFields('cost', r.cost), target: r.target.trim(), stacking_group: r.stacking_group.trim(), ...reqFields(r.requires) })
  for (const r of t.prestige) nodes.push({ id: r.id.trim(), type: 'prestige_layer', name: r.name || r.id, formula_expr: r.formula_expr, layer_index: num(r.layer_index, 0), reset_scope: splitList(r.reset_scope), persistence_scope: splitList(r.persistence_scope), ...(r.currency_id?.trim() ? { currency_id: r.currency_id.trim() } : {}) })
  for (const r of t.autobuyers) nodes.push({ id: r.id.trim(), type: 'autobuyer', name: r.name || r.id, target: r.target.trim(), interval: num(r.interval, 1), bulk_amount: r.bulk_amount || '1', ...(r.condition?.trim() ? { condition: r.condition.trim() } : {}) })
  return { schema_version: '1.0', name, nodes: [...nodes, ...extra.nodes], edges: [...edges, ...extra.edges], stacking_groups: groups }
}

/** GameDefinition JSON → tables (+ untouched leftovers). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- arbitrary game JSON
export function gameToTables(game: any): { name: string; tables: Tables; extra: { nodes: unknown[]; edges: unknown[] } } {
  const t = emptyTables()
  const extraNodes: unknown[] = []
  const prodEdge = new Map<string, string>()
  const extraEdges: unknown[] = []
  for (const e of game.edges ?? []) {
    if (e.edge_type === 'production_target' && !prodEdge.has(e.source)) prodEdge.set(e.source, e.target)
    else extraEdges.push(e)
  }
  const s = (v: unknown) => (v === undefined || v === null ? '' : String(v))
  for (const n of game.nodes ?? []) {
    switch (n.type) {
      case 'resource': t.resources.push({ id: n.id, name: s(n.name), initial_value: s(n.initial_value), capacity: s(n.capacity) }); break
      case 'generator': t.generators.push({ id: n.id, name: s(n.name), produces: s(prodEdge.get(n.id)), base_production: s(n.base_production), cycle_time: s(n.cycle_time), cost_base: n.costs?.length || Array.isArray(n.cost) ? fmtParts(n.costs ?? n.cost, ':') : s(n.cost_base), cost_growth_rate: s(n.cost_growth_rate), requires: fmtParts(n.requires ?? [], '>=') }); break
      case 'upgrade': t.upgrades.push({ id: n.id, name: s(n.name), target: s(n.target), upgrade_type: s(n.upgrade_type), magnitude: s(n.magnitude), cost: n.costs?.length ? fmtParts(n.costs, ':') : Array.isArray(n.cost) ? fmtParts(n.cost, ':') : s(n.cost), stacking_group: s(n.stacking_group), requires: fmtParts(n.requires ?? [], '>=') }); break
      case 'prestige_layer': t.prestige.push({ id: n.id, name: s(n.name), formula_expr: s(n.formula_expr), currency_id: s(n.currency_id), layer_index: s(n.layer_index), reset_scope: (n.reset_scope ?? []).join(', '), persistence_scope: (n.persistence_scope ?? []).join(', ') }); break
      case 'autobuyer': t.autobuyers.push({ id: n.id, name: s(n.name), target: s(n.target), interval: s(n.interval), bulk_amount: s(n.bulk_amount), condition: s(n.condition) }); break
      default: extraNodes.push(n)
    }
  }
  for (const [id, mode] of Object.entries(game.stacking_groups ?? {})) t.groups.push({ id, mode: String(mode) })
  return { name: game.name ?? '未命名', tables: t, extra: { nodes: extraNodes, edges: extraEdges } }
}

/** Map imported header row to column keys using label/key/aliases. */
export function mapHeaders(sys: SystemDef, header: string[]): (string | null)[] {
  return header.map((h) => {
    const k = h.trim().toLowerCase()
    const c = sys.cols.find((c) => c.key.toLowerCase() === k || c.label.toLowerCase() === k || (c.aliases ?? []).some((a) => a.toLowerCase() === k))
    return c ? c.key : null
  })
}
