interface PortfolioState {
  cash: number
  positions: Array<{
    ticker: string
    quantity: number
    entry_price: number
    entry_time: string
    cost_basis: number
  }>
  trades: Array<{
    id: number
    date: string
    ticker: string
    action: string
    quantity: number
    price: number
    exec_price: number
    total_cost: number
    reason: string
    timestamp: string
  }>
  start_date: string
  initial_capital: number
  daily_stats: {
    [date: string]: {
      pnl: number
      trades: number
      wins: number
      losses: number
    }
  }
}

const STORAGE_KEY = "day_trader_portfolio"
const INITIAL_CAPITAL = 10000

function loadPortfolio(): PortfolioState {
  if (typeof window === "undefined") {
    return createNewPortfolio()
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return createNewPortfolio()

    const portfolio = JSON.parse(stored) as PortfolioState

    // Reset daily if new day
    const today = new Date().toISOString().split("T")[0]
    if (portfolio.start_date !== today) {
      console.log("[v0] New trading day detected, resetting portfolio")
      return createNewPortfolio()
    }

    return portfolio
  } catch (error) {
    console.error("[v0] Error loading portfolio:", error)
    return createNewPortfolio()
  }
}

function createNewPortfolio(): PortfolioState {
  const today = new Date().toISOString().split("T")[0]
  return {
    cash: INITIAL_CAPITAL,
    positions: [],
    trades: [],
    start_date: today,
    initial_capital: INITIAL_CAPITAL,
    daily_stats: {},
  }
}

function savePortfolio(portfolio: PortfolioState) {
  if (typeof window === "undefined") return

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(portfolio))
    console.log("[v0] Portfolio saved:", {
      positions: portfolio.positions.length,
      cash: portfolio.cash,
      trades: portfolio.trades.length,
    })
  } catch (error) {
    console.error("[v0] Error saving portfolio:", error)
  }
}

export function getPortfolioState(): PortfolioState {
  return loadPortfolio()
}

export function addPosition(ticker: string, quantity: number, entry_price: number, reason: string): boolean {
  const portfolio = loadPortfolio()
  const cost = quantity * entry_price * 1.0005 // Include slippage
  const commission = 1.0

  if (portfolio.cash < cost + commission) {
    console.log("[v0] Insufficient cash for trade")
    return false
  }

  portfolio.cash -= cost + commission

  portfolio.positions.push({
    ticker,
    quantity,
    entry_price,
    entry_time: new Date().toISOString(),
    cost_basis: cost + commission,
  })

  const exec_price = entry_price * 1.0005
  portfolio.trades.push({
    id: portfolio.trades.length + 1,
    date: new Date().toISOString().split("T")[0],
    ticker,
    action: "BUY",
    quantity,
    price: entry_price,
    exec_price,
    total_cost: cost + commission,
    reason,
    timestamp: new Date().toISOString(),
  })

  savePortfolio(portfolio)
  console.log(`[v0] Position added: ${quantity} ${ticker} @ $${entry_price.toFixed(2)}`)
  return true
}

export function closePosition(ticker: string, current_price: number, reason: string): boolean {
  const portfolio = loadPortfolio()
  const posIndex = portfolio.positions.findIndex((p) => p.ticker === ticker)

  if (posIndex === -1) {
    console.log("[v0] Position not found:", ticker)
    return false
  }

  const position = portfolio.positions[posIndex]
  const exec_price = current_price * 0.9995 // Slippage on exit
  const proceeds = position.quantity * exec_price - 1.0 // Subtract commission

  portfolio.cash += proceeds
  portfolio.positions.splice(posIndex, 1)

  portfolio.trades.push({
    id: portfolio.trades.length + 1,
    date: new Date().toISOString().split("T")[0],
    ticker,
    action: "SELL",
    quantity: position.quantity,
    price: current_price,
    exec_price,
    total_cost: proceeds,
    reason,
    timestamp: new Date().toISOString(),
  })

  savePortfolio(portfolio)
  console.log(`[v0] Position closed: ${position.quantity} ${ticker} @ $${current_price.toFixed(2)}`)
  return true
}

export function getTodaysTrades(): Array<any> {
  const portfolio = loadPortfolio()
  const today = new Date().toISOString().split("T")[0]
  return portfolio.trades.filter((t) => t.date === today)
}

export function getAllTrades(): Array<any> {
  const portfolio = loadPortfolio()
  return [...portfolio.trades].reverse()
}

export function resetPortfolio() {
  if (typeof window !== "undefined") {
    localStorage.removeItem(STORAGE_KEY)
    console.log("[v0] Portfolio reset")
  }
}
