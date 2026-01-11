import { NextResponse } from "next/server"
import { fetchStockPrice, fetchHistoricalData } from "@/lib/multi-source-data"
import {
  DAY_TRADER_CONFIG,
  isLiquidTradingTime,
  calculatePositionSize,
  calculateVolatility,
} from "@/lib/day-trader-config"
import { getPortfolioState, addPosition, closePosition, getTodaysTrades } from "@/lib/persistent-portfolio"
import { getPrediction } from "@/lib/prediction" // Import getPrediction from shared utility instead of defining inline

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function logActivity(type: string, ticker: string | undefined, message: string, details?: any) {
  try {
    const activity = {
      timestamp: Date.now(),
      type,
      ticker,
      message,
      details,
    }

    const stored = localStorage.getItem("portfolio_activity_log")
    const logs = stored ? JSON.parse(stored) : []
    logs.push(activity)

    // Keep only last 100 activities
    const trimmed = logs.slice(-100)
    localStorage.setItem("portfolio_activity_log", JSON.stringify(trimmed))

    console.log(`[v0] Activity: ${type} - ${message}`)
  } catch (error) {
    console.error("[v0] Failed to log activity:", error)
  }
}

async function fetchRealTimePrice(ticker: string): Promise<number | null> {
  try {
    const result = await fetchStockPrice(ticker)
    return result.price
  } catch (error) {
    console.error(`[v0] Error fetching price for ${ticker}:`, error)
    return null
  }
}

export async function GET() {
  try {
    const portfolioState = getPortfolioState()
    const todaysTrades = getTodaysTrades()

    const isLiquidHours = isLiquidTradingTime()
    console.log("[v0] Trading hours check:", { isLiquidHours, time: new Date().toLocaleString() })

    logActivity(
      "scan",
      undefined,
      `Portfolio scan initiated. Liquid hours: ${isLiquidHours}. Positions: ${portfolioState.positions.length}`,
    )

    const today = new Date().toISOString().split("T")[0]

    const dailyPnL = todaysTrades.reduce((sum, t) => {
      return sum + (t.action === "SELL" ? t.total_cost : -t.total_cost)
    }, 0)
    const dailyLossPct = dailyPnL / portfolioState.initial_capital

    if (dailyLossPct < -DAY_TRADER_CONFIG.MAX_DAILY_LOSS_PCT) {
      console.log("[v0] Daily loss limit reached. Trading halted.", { dailyLossPct })
      logActivity("halt", undefined, `Trading halted: Daily loss limit exceeded (${(dailyLossPct * 100).toFixed(2)}%)`)
    }

    if (todaysTrades.length >= DAY_TRADER_CONFIG.MAX_TRADES_PER_DAY) {
      console.log("[v0] Max trades per day reached. No new positions.")
      logActivity("halt", undefined, `Max trades per day reached (${todaysTrades.length}/20)`)
    }

    const updatedPositions = []
    let positionsValue = 0

    for (const pos of portfolioState.positions) {
      try {
        const result = await fetchStockPrice(pos.ticker)
        const current_price = result.price
        const value = pos.quantity * current_price
        positionsValue += value
        const pnl = value - pos.cost_basis
        const pnl_pct = (pnl / pos.cost_basis) * 100

        const should_exit = pnl_pct <= -2 || pnl_pct >= 3 // Stop loss or take profit

        if (should_exit && isLiquidHours) {
          const reason = pnl_pct >= 3 ? "take_profit_3pct" : "stop_loss_2pct"
          closePosition(pos.ticker, current_price, reason)
          console.log(`[v0] Position exited: ${pos.ticker} - ${reason} (${pnl_pct.toFixed(2)}%)`)
          logActivity(
            "sell",
            pos.ticker,
            `Position exited: ${reason} at $${current_price.toFixed(2)} (${pnl_pct >= 0 ? "+" : ""}${pnl_pct.toFixed(2)}%)`,
            { reason, price: current_price, pnl_pct },
          )
          continue
        }

        updatedPositions.push({
          ticker: pos.ticker,
          quantity: pos.quantity,
          entry_price: pos.entry_price,
          current_price,
          value,
          pnl,
          pnl_pct,
        })
      } catch (error) {
        console.error(`[v0] Error updating position ${pos.ticker}:`, error)
      }
    }

    if (
      isLiquidHours &&
      updatedPositions.length < DAY_TRADER_CONFIG.MAX_SIMULTANEOUS_POSITIONS &&
      todaysTrades.length < DAY_TRADER_CONFIG.MAX_TRADES_PER_DAY
    ) {
      const sp500Pool = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK.B",
        "UNH",
        "JNJ",
        "V",
        "XOM",
        "WMT",
        "JPM",
        "PG",
        "MA",
        "MRK",
        "ABBV",
        "KO",
        "COST",
      ]

      console.log(`[v0] Scanning for new opportunities...`)
      logActivity("scan", undefined, `Scanning ${sp500Pool.length} S&P 500 stocks for opportunities...`)

      for (const ticker of sp500Pool) {
        if (updatedPositions.some((p) => p.ticker === ticker)) continue // Skip if already holding

        const symbolTradesToday = todaysTrades.filter((t) => t.ticker === ticker).length
        if (symbolTradesToday >= DAY_TRADER_CONFIG.MAX_TRADES_PER_SYMBOL) continue

        try {
          const result = await fetchStockPrice(ticker)
          const price = result.price

          const pred = await getPrediction(ticker)
          if (!pred) {
            logActivity("skip", ticker, `Insufficient data for prediction`)
            continue
          }

          if (pred.confidence > DAY_TRADER_CONFIG.MIN_CONFIDENCE_THRESHOLD) {
            const expectedGain = ((pred.prediction - price) / price) * 100

            if (expectedGain > DAY_TRADER_CONFIG.MIN_EDGE_PER_TRADE * 100) {
              const historicalResult = await fetchHistoricalData(ticker, "1y")
              const closes = historicalResult.indicators.quote[0].close.filter((c: number) => c !== null)
              const volatility = calculateVolatility(closes)

              const { size: targetAllocation } = calculatePositionSize(pred.confidence, volatility, portfolioState.cash)
              const quantity = Math.floor(targetAllocation / price)

              if (quantity > 0) {
                const success = addPosition(ticker, quantity, price, `ml_signal_buy_conf_${pred.confidence.toFixed(0)}`)
                if (success) {
                  console.log(`[v0] NEW TRADE EXECUTED: BUY ${quantity} ${ticker} @ $${price.toFixed(2)}`)
                  logActivity(
                    "buy",
                    ticker,
                    `BUY ${quantity} shares @ $${price.toFixed(2)} (Confidence: ${pred.confidence.toFixed(0)}%, Expected gain: ${expectedGain.toFixed(1)}%)`,
                    { quantity, price, confidence: pred.confidence, expectedGain },
                  )

                  updatedPositions.push({
                    ticker,
                    quantity,
                    entry_price: price,
                    current_price: price,
                    value: quantity * price,
                    pnl: 0,
                    pnl_pct: 0,
                  })

                  positionsValue += quantity * price
                }
              } else {
                logActivity("skip", ticker, `Position size too small (${quantity} shares)`)
              }
            } else {
              logActivity(
                "skip",
                ticker,
                `Expected gain too low (${expectedGain.toFixed(1)}%, need ${(DAY_TRADER_CONFIG.MIN_EDGE_PER_TRADE * 100).toFixed(1)}%+)`,
              )
            }
          } else {
            logActivity(
              "skip",
              ticker,
              `Confidence too low (${pred.confidence.toFixed(0)}%, need ${DAY_TRADER_CONFIG.MIN_CONFIDENCE_THRESHOLD}%+)`,
            )
          }
        } catch (error) {
          console.log(`[v0] Skipping ${ticker}:`, error)
          logActivity("skip", ticker, `Error during scan: ${error instanceof Error ? error.message : "Unknown"}`)
        }

        if (updatedPositions.length >= DAY_TRADER_CONFIG.MAX_SIMULTANEOUS_POSITIONS) break
      }
    }

    const totalValue = portfolioState.cash + positionsValue
    const totalPnL = totalValue - portfolioState.initial_capital
    const totalPnLPct = (totalPnL / portfolioState.initial_capital) * 100

    // Calculate metrics
    const winTrades = todaysTrades.filter((t) => t.action === "SELL" && t.total_cost > 0)
    const lossTrades = todaysTrades.filter((t) => t.action === "SELL" && t.total_cost < 0)
    const hitRate = todaysTrades.length > 0 ? winTrades.length / todaysTrades.length : 0
    const avgWin = winTrades.length > 0 ? winTrades.reduce((sum, t) => sum + t.total_cost, 0) / winTrades.length : 0
    const avgLoss =
      lossTrades.length > 0 ? Math.abs(lossTrades.reduce((sum, t) => sum + t.total_cost, 0) / lossTrades.length) : 0
    const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0

    return NextResponse.json({
      cash: portfolioState.cash,
      positions: updatedPositions,
      total_value: totalValue,
      total_pnl: totalPnL,
      total_pnl_pct: totalPnLPct,
      daily_pnl: dailyPnL,
      daily_pnl_pct: dailyLossPct * 100,
      metrics: {
        trades_today: todaysTrades.length,
        hit_rate: hitRate,
        profit_factor: profitFactor,
        avg_win: avgWin,
        avg_loss: avgLoss,
      },
      trading_active: isLiquidHours,
      last_updated: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] Portfolio error:", error)
    logActivity("halt", undefined, `Critical portfolio error: ${error instanceof Error ? error.message : "Unknown"}`)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
