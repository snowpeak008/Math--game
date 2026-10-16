import { describe, expect, it } from 'vitest'
import { buildGame, checkTables, emptyTables, gameToTables, parsePrice, parseReqs } from '../schema'

describe('multi-currency price cells', () => {
  it('parses legacy numbers and currency lists', () => {
    expect(parsePrice('15')).toEqual({ legacy: 15, parts: [] })
    expect(parsePrice('gold:15000; diamond:10').parts).toEqual([
      { currency: 'gold', amount: 15000 }, { currency: 'diamond', amount: 10 }])
    expect(parsePrice('gold：5 + gem=2').parts).toHaveLength(2)
    expect(parsePrice('gold 5').error).toBeTruthy()
    expect(parseReqs('honor>=100; gem≥3').parts).toEqual([
      { currency: 'honor', amount: 100 }, { currency: 'gem', amount: 3 }])
  })

  it('builds cost lists / requires and round-trips', () => {
    const t = emptyTables()
    t.resources.push({ id: 'gold', name: '金币', initial_value: '10', capacity: '' }, { id: 'honor', name: '荣誉值', initial_value: '0', capacity: '' })
    t.generators.push({ id: 'arena', name: '竞技场', produces: 'honor', base_production: '1', cycle_time: '1', cost_base: 'gold:800', cost_growth_rate: '1.2', requires: '' })
    t.upgrades.push({ id: 'h1', name: '荣誉', target: '_all', upgrade_type: 'multiplicative', magnitude: '2', cost: '2000', stacking_group: 'upgrades', requires: 'honor>=100' })
    expect(checkTables(t).filter((i) => i.level === 'error')).toEqual([])
    const g = buildGame('x', t)
    const arena = g.nodes.find((n) => (n as { id: string }).id === 'arena') as Record<string, unknown>
    expect(arena.cost).toEqual([{ currency: 'gold', amount: 800 }])
    expect(arena.cost_base).toBeUndefined()
    const h1 = g.nodes.find((n) => (n as { id: string }).id === 'h1') as Record<string, unknown>
    expect(h1.cost).toBe(2000)
    expect(h1.requires).toEqual([{ currency: 'honor', amount: 100 }])
    const back = gameToTables(g).tables
    expect(back.generators[0].cost_base).toBe('gold:800')
    expect(back.upgrades[0].requires).toBe('honor>=100')
  })

  it('flags unknown currencies', () => {
    const t = emptyTables()
    t.resources.push({ id: 'gold', name: '', initial_value: '0', capacity: '' })
    t.upgrades.push({ id: 'u', name: '', target: '_all', upgrade_type: 'multiplicative', magnitude: '2', cost: 'gold:1; gem:2', stacking_group: 'upgrades', requires: 'honor>=1' })
    const errs = checkTables(t).filter((i) => i.level === 'error').map((i) => i.msg)
    expect(errs.some((m) => m.includes('gem'))).toBe(true)
    expect(errs.some((m) => m.includes('honor'))).toBe(true)
  })
})
