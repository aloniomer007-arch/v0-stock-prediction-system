let tradeHistory: Array<{
  id: number
  date: string
  ticker: string
  action: string
  quantity: number
  price: number
  exec_price: number
  slippage: number
  commission: number
  total_cost: number
  reason: string
  timestamp: string
}> = []

let tradeIdCounter = 1

export function logTrade(trade: {
  ticker: string
  action: string
  quantity: number
  price: number
  reason: string
}) {
  const slippage = trade.price * 0.0005 // 5 basis points
  const exec_price = trade.action === "BUY" ? trade.price + slippage : trade.price - slippage
  const commission = 1.0
  const total_cost = trade.quantity * exec_price + (trade.action === "BUY" ? commission : -commission)

  const newTrade = {
    id: tradeIdCounter++,
    date: new Date().toISOString().split("T")[0],
    ticker: trade.ticker,
    action: trade.action,
    quantity: trade.quantity,
    price: trade.price,
    exec_price,
    slippage,
    commission,
    total_cost,
    reason: trade.reason,
    timestamp: new Date().toISOString(),
  }

  tradeHistory.push(newTrade)

  // Keep only last 100 trades to prevent memory issues
  if (tradeHistory.length > 100) {
    tradeHistory = tradeHistory.slice(-100)
  }

  console.log("[v0] Trade logged:", {
    ticker: newTrade.ticker,
    action: newTrade.action,
    quantity: newTrade.quantity,
    price: newTrade.price,
  })

  return newTrade
}

export function getTrades() {
  return [...tradeHistory].reverse() // Most recent first
}

export function clearTrades() {
  tradeHistory = []
  tradeIdCounter = 1
}
