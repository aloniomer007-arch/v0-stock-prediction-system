"use client"
import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, AlertCircle, XCircle } from "lucide-react"
import { Progress } from "@/components/ui/progress"

interface Prediction {
  ticker: string
  horizon: string
  predicted_return?: number
  predicted_price?: number
  current_price?: number
  lower_bound?: number
  upper_bound?: number
  confidence: number
  model_version: string
  features_used?: any
  timestamp: string
  error?: string
}

export default function PredictionPanel({
  ticker,
  horizon,
  compact = false,
}: {
  ticker: string
  horizon: string
  compact?: boolean
}) {
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    if (ticker && ticker.trim()) {
      fetchPrediction()
    }
  }, [ticker, horizon])

  const fetchPrediction = async () => {
    setLoading(true)
    setError("")
    setPrediction(null)

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.trim(), horizon }),
      })

      const data = await response.json()

      if (!response.ok || data.error) {
        setError(data.error || "Failed to fetch prediction")
        setPrediction(null)
      } else {
        setPrediction(data)
        setError("")
      }
    } catch (err) {
      console.error("[v0] Error fetching prediction:", err)
      setError("Network error. Please try again.")
      setPrediction(null)
    }
    setLoading(false)
  }

  if (compact) {
    if (loading) {
      return (
        <div className="flex items-center gap-3 py-2 px-3 bg-muted/50 rounded-lg">
          <div className="text-sm text-muted-foreground">Loading {ticker}...</div>
        </div>
      )
    }

    if (error || !prediction) {
      return (
        <div className="flex items-center gap-3 py-2 px-3 bg-red-500/5 rounded-lg border border-red-500/20">
          <XCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
          <span className="text-sm font-medium">{ticker}</span>
          <span className="text-xs text-muted-foreground">Error loading data</span>
        </div>
      )
    }

    const predictedReturn = prediction.predicted_return ?? prediction.pred ?? 0
    const isPositive = predictedReturn > 0
    const predPercent = (predictedReturn * 100).toFixed(2)

    return (
      <div className="flex items-center justify-between gap-4 py-2 px-4 bg-card rounded-lg border hover:bg-accent/50 transition-colors">
        <div className="flex items-center gap-3 min-w-0">
          {isPositive ? (
            <TrendingUp className="h-4 w-4 text-green-500 flex-shrink-0" />
          ) : (
            <TrendingDown className="h-4 w-4 text-red-500 flex-shrink-0" />
          )}
          <span className="font-semibold text-sm">{prediction.ticker}</span>
          {prediction.current_price && (
            <span className="text-xs text-muted-foreground">${prediction.current_price.toFixed(2)}</span>
          )}
        </div>

        <div className="flex items-center gap-4">
          {prediction.predicted_price && (
            <div className="text-sm">
              <span className="text-muted-foreground text-xs mr-1">→</span>
              <span className="font-medium">${prediction.predicted_price.toFixed(2)}</span>
            </div>
          )}

          <div className="flex items-baseline gap-1">
            <span className="text-lg font-bold tabular-nums" style={{ color: isPositive ? "#22c55e" : "#ef4444" }}>
              {isPositive ? "+" : ""}
              {predPercent}%
            </span>
          </div>

          <div className="text-xs text-muted-foreground tabular-nums min-w-[3rem] text-right">
            {(prediction.confidence * 100).toFixed(0)}% conf
          </div>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="p-12">
          <div className="text-center text-muted-foreground">Loading prediction for {ticker}...</div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-12">
          <div className="flex flex-col items-center gap-4 text-center">
            <XCircle className="h-12 w-12 text-red-500" />
            <div>
              <h3 className="font-semibold text-lg mb-2">Error Loading Prediction</h3>
              <p className="text-muted-foreground">{error}</p>
              <p className="text-sm text-muted-foreground mt-4">
                Make sure the ticker exists and the model is trained.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!prediction) return null

  const predictedReturn = prediction.predicted_return ?? prediction.pred ?? 0
  const lowerBound = prediction.lower_bound ?? prediction.lower ?? 0
  const upperBound = prediction.upper_bound ?? prediction.upper ?? 0

  const isPositive = predictedReturn > 0
  const predPercent = (predictedReturn * 100).toFixed(2)
  const lowerPercent = (lowerBound * 100).toFixed(2)
  const upperPercent = (upperBound * 100).toFixed(2)

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {isPositive ? (
              <TrendingUp className="h-5 w-5 text-green-500" />
            ) : (
              <TrendingDown className="h-5 w-5 text-red-500" />
            )}
            {prediction.ticker} - {prediction.horizon.toUpperCase()} Prediction
          </CardTitle>
          <CardDescription>Model: {prediction.model_version}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {prediction.current_price && prediction.predicted_price && (
            <div className="space-y-2 pb-4 border-b">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Current Price</span>
                <span className="font-medium">${prediction.current_price.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Predicted Price</span>
                <span className="font-medium" style={{ color: isPositive ? "#22c55e" : "#ef4444" }}>
                  ${prediction.predicted_price.toFixed(2)}
                </span>
              </div>
            </div>
          )}

          <div>
            <div className="flex items-baseline gap-2 mb-2">
              <span className="text-4xl font-bold" style={{ color: isPositive ? "#22c55e" : "#ef4444" }}>
                {predPercent}%
              </span>
              <span className="text-muted-foreground">expected return</span>
            </div>

            <div className="space-y-2 mt-4">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">95% Confidence Interval</span>
                <span className="font-medium">
                  [{lowerPercent}%, {upperPercent}%]
                </span>
              </div>
              <Progress value={prediction.confidence * 100} className="h-2" />
              <div className="text-xs text-muted-foreground text-right">
                {(prediction.confidence * 100).toFixed(0)}% confidence
              </div>
            </div>
          </div>

          <div className="pt-4 border-t">
            <div className="flex items-start gap-2 text-sm text-muted-foreground">
              <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <p className="text-balance">
                This prediction is for informational and backtesting purposes only. Not financial advice.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Technical Indicators</CardTitle>
          <CardDescription>Features used for this prediction</CardDescription>
        </CardHeader>
        <CardContent>
          {prediction.features_used && (
            <div className="space-y-3">
              {Object.entries(prediction.features_used).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="font-medium text-muted-foreground">{key}</span>
                  <span className="font-mono">{typeof value === "number" ? value.toFixed(4) : String(value)}</span>
                </div>
              ))}
            </div>
          )}

          <div className="mt-6 pt-4 border-t text-xs text-muted-foreground">
            <p>
              Prediction generated using LightGBM model trained on 20 years of historical data with technical indicators
              and fundamental features.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
