"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Activity } from "lucide-react"
import PredictionPanel from "@/components/prediction-panel"
import PortfolioView from "@/components/portfolio-view"
import TradeHistory from "@/components/trade-history"
import MetricsDashboard from "@/components/metrics-dashboard"

import { getCachedValidation, setCachedValidation, clearValidationCache } from "@/lib/validation-cache"

export default function Home() {
  const [ticker, setTicker] = useState("")
  const [horizon, setHorizon] = useState("1d")
  const [activeTicker, setActiveTicker] = useState("")
  const [activeHorizon, setActiveHorizon] = useState("")
  const [error, setError] = useState("")
  const [demoTickers, setDemoTickers] = useState<string[]>([])
  const [showValidation, setShowValidation] = useState(true)
  const [validationResults, setValidationResults] = useState<any>(null)
  const [isValidating, setIsValidating] = useState(false)

  useEffect(() => {
    const sp500Tickers = [
      "AAPL",
      "MSFT",
      "GOOGL",
      "AMZN",
      "NVDA",
      "META",
      "TSLA",
      "BRK-B",
      "UNH",
      "JNJ",
      "V",
      "XOM",
      "WMT",
      "JPM",
      "PG",
      "MA",
      "HD",
      "CVX",
      "MRK",
      "ABBV",
      "KO",
      "PEP",
      "COST",
      "AVGO",
      "TMO",
      "LLY",
      "MCD",
      "CSCO",
      "ACN",
      "ABT",
      "ADBE",
      "NKE",
      "DHR",
      "TXN",
      "NEE",
      "PM",
      "VZ",
      "CMCSA",
      "UPS",
      "RTX",
      "HON",
      "ORCL",
      "NFLX",
      "INTC",
      "CRM",
      "WFC",
      "BMY",
      "QCOM",
      "AMD",
      "INTU",
    ]

    const updateDemo = () => {
      const shuffled = [...sp500Tickers].sort(() => Math.random() - 0.5)
      setDemoTickers(shuffled.slice(0, 10))
    }

    updateDemo()
    const interval = setInterval(updateDemo, 30000)
    return () => clearInterval(interval)
  }, [])

  const runValidation = async () => {
    const cached = getCachedValidation()
    if (cached) {
      console.log("[v0] Using cached validation results")
      setValidationResults(cached)
      return
    }

    setIsValidating(true)
    try {
      const res = await fetch("/api/validate")
      const data = await res.json()
      setValidationResults(data)
      setCachedValidation(data)
    } catch (err) {
      console.error("[v0] Validation failed:", err)
    } finally {
      setIsValidating(false)
    }
  }

  useEffect(() => {
    runValidation()
  }, [])

  const handleGetPrediction = () => {
    const trimmedTicker = ticker.trim().toUpperCase()

    if (!trimmedTicker) {
      setError("Please enter a ticker symbol")
      setActiveTicker("")
      setActiveHorizon("")
      return
    }

    if (!/^[A-Z]{1,5}$/.test(trimmedTicker)) {
      setError("Ticker must be 1-5 uppercase letters (e.g., AAPL)")
      setActiveTicker("")
      setActiveHorizon("")
      return
    }

    setError("")
    setActiveTicker(trimmedTicker)
    setActiveHorizon(horizon)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleGetPrediction()
    }
  }

  const handleTickerChange = (value: string) => {
    setTicker(value.toUpperCase())
    setError("")

    if (!value.trim()) {
      setActiveTicker("")
      setActiveHorizon("")
    }
  }

  const handleRefreshValidation = () => {
    clearValidationCache()
    runValidation()
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto p-6 space-y-8">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Stock Prediction System</h1>
          <p className="text-muted-foreground text-lg">
            Real-time ML predictions using technical analysis on live market data
          </p>
        </div>

        {showValidation && validationResults && (
          <Card className="border-green-500/50 bg-green-500/5">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-green-600">Model Validation Results</CardTitle>
                  <CardDescription>
                    Tested on {validationResults.test_samples} predictions across {validationResults.tickers_tested} S&P
                    500 stocks over {validationResults.test_period_years || 5} years
                  </CardDescription>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setShowValidation(false)}>
                  Dismiss
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Direction Accuracy</p>
                  <p className="text-2xl font-bold text-green-600">
                    {(validationResults.direction_accuracy * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {validationResults.direction_accuracy > 0.55
                      ? "Excellent"
                      : validationResults.direction_accuracy > 0.53
                        ? "Good"
                        : validationResults.direction_accuracy > 0.5
                          ? "Above Random"
                          : "Baseline"}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">RMSE</p>
                  <p className="text-2xl font-bold">{(validationResults.rmse * 100).toFixed(2)}%</p>
                  <p className="text-xs text-muted-foreground">Prediction error</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Sharpe Ratio</p>
                  <p className="text-2xl font-bold">{validationResults.sharpe_ratio.toFixed(2)}</p>
                  <p className="text-xs text-muted-foreground">
                    {validationResults.sharpe_ratio > 1.5
                      ? "Excellent"
                      : validationResults.sharpe_ratio > 1.0
                        ? "Good"
                        : "Acceptable"}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Avg Return</p>
                  <p className="text-2xl font-bold text-green-600">
                    +{(validationResults.avg_return * 100).toFixed(2)}%
                  </p>
                  <p className="text-xs text-muted-foreground">Per prediction</p>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t">
                <Button variant="outline" size="sm" onClick={handleRefreshValidation} disabled={isValidating}>
                  {isValidating ? "Validating..." : "Re-run Validation"}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {showValidation && !validationResults && isValidating && (
          <Card className="border-blue-500/50 bg-blue-500/5">
            <CardContent className="py-8">
              <div className="text-center space-y-2">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto" />
                <p className="text-sm text-muted-foreground">Running backtesting validation on historical data...</p>
              </div>
            </CardContent>
          </Card>
        )}

        <Card className="border-blue-500/50 bg-blue-500/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-blue-500" />
              Live S&P 500 Demo (Auto-Updating Every 30s)
            </CardTitle>
            <CardDescription>Watching 10 random S&P 500 companies with real-time predictions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {demoTickers.map((ticker) => (
                <PredictionPanel key={`${ticker}-demo`} ticker={ticker} horizon="7d" compact={true} />
              ))}
            </div>
          </CardContent>
        </Card>

        <Tabs defaultValue="predict" className="space-y-4">
          <TabsList>
            <TabsTrigger value="predict">Predictions</TabsTrigger>
            <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
            <TabsTrigger value="trades">Trade History</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
          </TabsList>

          <TabsContent value="predict" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Get Stock Prediction</CardTitle>
                <CardDescription>
                  Enter a valid ticker symbol (e.g., AAPL, MSFT, GOOGL) to get ML-powered predictions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="ticker">Ticker Symbol</Label>
                    <Input
                      id="ticker"
                      value={ticker}
                      onChange={(e) => handleTickerChange(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Enter ticker (e.g., AAPL)"
                      className="uppercase"
                      maxLength={5}
                    />
                    {error && <p className="text-sm text-red-500">{error}</p>}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="horizon">Time Horizon</Label>
                    <Select value={horizon} onValueChange={setHorizon}>
                      <SelectTrigger id="horizon">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1d">1 Day</SelectItem>
                        <SelectItem value="7d">1 Week (5 days)</SelectItem>
                        <SelectItem value="30d">1 Month (21 days)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>&nbsp;</Label>
                    <Button className="w-full" onClick={handleGetPrediction} disabled={!ticker.trim()}>
                      Get Prediction
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {activeTicker && activeHorizon && <PredictionPanel ticker={activeTicker} horizon={activeHorizon} />}
          </TabsContent>

          <TabsContent value="portfolio">
            <PortfolioView />
          </TabsContent>

          <TabsContent value="trades">
            <TradeHistory />
          </TabsContent>

          <TabsContent value="metrics">
            <MetricsDashboard />
          </TabsContent>
        </Tabs>

        <Card>
          <CardHeader>
            <CardTitle>System Architecture</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h3 className="font-semibold mb-2">Data Pipeline</h3>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• 20 years historical data (OHLCV + fundamentals)</li>
                  <li>• Technical indicators: SMA, EMA, RSI, MACD, ATR</li>
                  <li>• Macro indicators: S&P 500, VIX</li>
                  <li>• Feature engineering with exact formulas</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold mb-2">Models</h3>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• LightGBM gradient boosted trees</li>
                  <li>• Ensemble with 5 models for uncertainty</li>
                  <li>• Quantile regression for confidence intervals</li>
                  <li>• Walk-forward cross-validation</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold mb-2">Trading Simulator</h3>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• $10,000 initial capital</li>
                  <li>• Realistic slippage (5 bps) and commissions</li>
                  <li>• Position sizing with risk limits (20% max)</li>
                  <li>• Full trade log with execution details</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold mb-2">Monitoring & Auto-Retrain</h3>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• RMSE, MAE, direction accuracy tracking</li>
                  <li>• Financial metrics: Sharpe, CAGR, drawdown</li>
                  <li>• Data drift detection (PSI)</li>
                  <li>• Automated retraining on degradation</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
