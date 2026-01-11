"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, Activity, AlertCircle, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Position {
  ticker: string
  quantity: number
  current_price: number
  value: number
  pnl: number
  pnl_pct: number
  entry_price?: number
  expected_target?: number
  confidence?: number
}

interface Portfolio {
  cash: number
  positions: Position[]
  total_value: number
  total_pnl: number
  total_pnl_pct: number
  last_updated?: string
  initial_capital?: number
  start_date?: string
  is_liquid_hours?: boolean
  trading_halted?: boolean
  halt_reason?: string
  day_trader_metrics?: {
    trades_today: number
    hit_rate: number
    avg_gain: number
    opportunities_scanned: number
    profit_factor?: number
    avg_win?: number
    avg_loss?: number
    expectancy?: number
    daily_pnl?: number
    max_drawdown_pct?: number
    position_concentration?: number
  }
}

interface ActivityLog {
  timestamp: number
  type: "scan" | "buy" | "sell" | "skip" | "halt"
  ticker?: string
  message: string
  details?: any
}

export default function PortfolioView() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [activityLog, setActivityLog] = useState<ActivityLog[]>([])

  useEffect(() => {
    fetchPortfolio()
    loadActivityLog()

    const interval = setInterval(() => {
      fetchPortfolio()
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const loadActivityLog = () => {
    try {
      const stored = localStorage.getItem("portfolio_activity_log")
      if (stored) {
        const logs: ActivityLog[] = JSON.parse(stored)
        // Keep only last 50 activities
        setActivityLog(logs.slice(-50))
      }
    } catch (error) {
      console.error("[v0] Failed to load activity log:", error)
    }
  }

  const fetchPortfolio = async () => {
    try {
      const response = await fetch("/api/portfolio")
      const data = await response.json()
      setPortfolio(data)
      setLastUpdate(new Date())
      console.log("[v0] Portfolio updated:", data)
    } catch (error) {
      console.error("[v0] Error fetching portfolio:", error)
      addActivity({
        timestamp: Date.now(),
        type: "halt",
        message: "Failed to fetch portfolio data",
        details: { error: error instanceof Error ? error.message : "Unknown error" },
      })
    }
  }

  const addActivity = (activity: ActivityLog) => {
    setActivityLog((prev) => {
      const updated = [...prev, activity].slice(-50)
      try {
        localStorage.setItem("portfolio_activity_log", JSON.stringify(updated))
      } catch (error) {
        console.error("[v0] Failed to save activity log:", error)
      }
      return updated
    })
  }

  if (!portfolio) {
    return (
      <Card>
        <CardContent className="p-12">
          <div className="text-center text-muted-foreground">Loading portfolio...</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card className="border-blue-500/50 bg-blue-500/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" />
            Day Trading Mode - Live Simulation
            {portfolio.trading_halted && (
              <span className="ml-2 px-2 py-1 bg-red-500/20 text-red-500 text-xs rounded">HALTED</span>
            )}
            {!portfolio.is_liquid_hours && !portfolio.trading_halted && (
              <span className="ml-2 px-2 py-1 bg-yellow-500/20 text-yellow-500 text-xs rounded">OFF HOURS</span>
            )}
            {portfolio.is_liquid_hours && !portfolio.trading_halted && (
              <span className="ml-2 px-2 py-1 bg-green-500/20 text-green-500 text-xs rounded">LIVE</span>
            )}
          </CardTitle>
          <CardDescription>
            {portfolio.trading_halted
              ? `Trading halted: ${portfolio.halt_reason}`
              : `Started today with $${portfolio.initial_capital?.toLocaleString() || "10,000"} • Updates every 30 seconds • Trading during liquid hours (9:30-11:00 AM, 3:00-4:00 PM ET)`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Trades Today</p>
              <p className="text-2xl font-bold">{portfolio.day_trader_metrics?.trades_today || 0}/20</p>
            </div>
            <div>
              <p className="text-muted-foreground">Hit Rate</p>
              <p className="text-2xl font-bold">{portfolio.day_trader_metrics?.hit_rate.toFixed(0) || 0}%</p>
              <p className="text-xs text-muted-foreground">Target: 52%+</p>
            </div>
            <div>
              <p className="text-muted-foreground">Profit Factor</p>
              <p
                className="text-2xl font-bold"
                style={{ color: (portfolio.day_trader_metrics?.profit_factor || 0) >= 1.2 ? "#22c55e" : "#ef4444" }}
              >
                {portfolio.day_trader_metrics?.profit_factor?.toFixed(2) || "0.00"}
              </p>
              <p className="text-xs text-muted-foreground">Target: 1.2+</p>
            </div>
            <div>
              <p className="text-muted-foreground">Expectancy</p>
              <p
                className="text-2xl font-bold"
                style={{ color: (portfolio.day_trader_metrics?.expectancy || 0) >= 0 ? "#22c55e" : "#ef4444" }}
              >
                ${portfolio.day_trader_metrics?.expectancy?.toFixed(2) || "0.00"}
              </p>
              <p className="text-xs text-muted-foreground">Per trade</p>
            </div>
            <div>
              <p className="text-muted-foreground">Daily P&L</p>
              <p
                className="text-2xl font-bold"
                style={{ color: (portfolio.day_trader_metrics?.daily_pnl || 0) >= 0 ? "#22c55e" : "#ef4444" }}
              >
                {(portfolio.day_trader_metrics?.daily_pnl || 0) >= 0 ? "+" : ""}$
                {portfolio.day_trader_metrics?.daily_pnl?.toFixed(2) || "0.00"}
              </p>
              <p className="text-xs text-muted-foreground">Max loss: -$300</p>
            </div>
            <div>
              <p className="text-muted-foreground">Stocks Scanned</p>
              <p className="text-2xl font-bold">{portfolio.day_trader_metrics?.opportunities_scanned || 0}</p>
              <p className="text-xs text-muted-foreground">S&P 500</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Total Value</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">${portfolio.total_value.toFixed(2)}</div>
            <p className="text-sm text-muted-foreground mt-1">Cash: ${portfolio.cash.toFixed(2)}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Total P&L</CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className="text-3xl font-bold flex items-center gap-2"
              style={{ color: portfolio.total_pnl >= 0 ? "#22c55e" : "#ef4444" }}
            >
              {portfolio.total_pnl >= 0 ? <TrendingUp className="h-6 w-6" /> : <TrendingDown className="h-6 w-6" />}$
              {Math.abs(portfolio.total_pnl).toFixed(2)}
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              {portfolio.total_pnl >= 0 ? "+" : ""}
              {portfolio.total_pnl_pct.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{portfolio.positions.length}</div>
            <p className="text-sm text-muted-foreground mt-1">Active holdings</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle>Active Positions</CardTitle>
              <CardDescription>
                Current day trades • Max 5 simultaneous • Max 3 per symbol • Volatility-based sizing
                <br />
                Last updated: {lastUpdate.toLocaleTimeString()}
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={fetchPortfolio}>
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="grid grid-cols-7 gap-4 pb-2 border-b font-medium text-sm">
              <div>Ticker</div>
              <div className="text-right">Qty</div>
              <div className="text-right">Entry</div>
              <div className="text-right">Current</div>
              <div className="text-right">Target</div>
              <div className="text-right">P&L</div>
              <div className="text-right">Confidence</div>
            </div>

            {portfolio.positions.map((position) => (
              <div key={position.ticker} className="grid grid-cols-7 gap-4 py-2 border-b text-sm">
                <div className="font-medium">{position.ticker}</div>
                <div className="text-right text-muted-foreground">{position.quantity}</div>
                <div className="text-right text-xs">${position.entry_price?.toFixed(2) || "-"}</div>
                <div className="text-right">${position.current_price.toFixed(2)}</div>
                <div className="text-right text-xs text-green-500">${position.expected_target?.toFixed(2) || "-"}</div>
                <div className="text-right font-medium" style={{ color: position.pnl >= 0 ? "#22c55e" : "#ef4444" }}>
                  {position.pnl >= 0 ? "+" : ""}
                  {position.pnl_pct.toFixed(1)}%
                </div>
                <div className="text-right text-xs">{position.confidence?.toFixed(0) || "-"}%</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Real-Time Activity Feed</CardTitle>
          <CardDescription>Live scanner decisions and trade execution log</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {activityLog.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                No activity yet. Scanner runs every 30 seconds during market hours.
              </div>
            ) : (
              activityLog
                .slice()
                .reverse()
                .map((activity, index) => (
                  <div key={index} className="flex items-start gap-3 p-2 border-b text-sm">
                    <div className="mt-1">
                      {activity.type === "buy" && <CheckCircle className="h-4 w-4 text-green-500" />}
                      {activity.type === "sell" && <TrendingDown className="h-4 w-4 text-orange-500" />}
                      {activity.type === "scan" && <Activity className="h-4 w-4 text-blue-500" />}
                      {activity.type === "skip" && <AlertCircle className="h-4 w-4 text-yellow-500" />}
                      {activity.type === "halt" && <AlertCircle className="h-4 w-4 text-red-500" />}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">
                          {activity.ticker && <span className="text-blue-500">{activity.ticker}</span>}
                          {!activity.ticker && <span className="capitalize">{activity.type}</span>}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {new Date(activity.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-muted-foreground">{activity.message}</p>
                    </div>
                  </div>
                ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
