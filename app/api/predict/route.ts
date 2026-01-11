import { type NextRequest, NextResponse } from "next/server"
import { fetchStockPrice, fetchHistoricalData } from "@/lib/multi-source-data"

async function validateTickerOnline(
  ticker: string,
): Promise<{ valid: boolean; error?: string; currentPrice?: number }> {
  try {
    const result = await fetchStockPrice(ticker)
    return {
      valid: true,
      currentPrice: result.price,
    }
  } catch (error) {
    console.error("[v0] Ticker validation error:", error)
    return { valid: false, error: "Could not validate ticker" }
  }
}

async function fetchRealData(ticker: string) {
  try {
    const result = await fetchHistoricalData(ticker, "1y")

    const timestamps = result.timestamp
    const quotes = result.indicators.quote[0]
    const closes = quotes.close
    const highs = quotes.high
    const lows = quotes.low
    const volumes = quotes.volume

    // Filter out null values
    const validData = timestamps
      .map((ts: number, i: number) => ({
        timestamp: ts,
        close: closes[i],
        high: highs[i],
        low: lows[i],
        volume: volumes[i],
      }))
      .filter((d: any) => d.close !== null && d.high !== null && d.low !== null)

    return validData
  } catch (error) {
    console.error("[v0] Error fetching historical data:", error)
    return null
  }
}

function calculateIndicators(data: any[]) {
  if (data.length < 50) return null

  const closes = data.map((d) => d.close)
  const highs = data.map((d) => d.high)
  const lows = data.map((d) => d.low)
  const volumes = data.map((d) => d.volume)

  // SMA
  const sma20 = closes.slice(-20).reduce((a, b) => a + b, 0) / 20
  const sma50 = closes.slice(-50).reduce((a, b) => a + b, 0) / 50

  // EMA
  const ema12 = calculateEMA(closes, 12)
  const ema26 = calculateEMA(closes, 26)

  // RSI
  const rsi = calculateRSI(closes, 14)

  // MACD
  const macd = ema12 - ema26

  // ATR
  const atr = calculateATR(highs, lows, closes, 14)

  // Volume average
  const avgVolume = volumes.slice(-20).reduce((a: number, b: number) => a + b, 0) / 20

  const currentPrice = closes[closes.length - 1]

  return {
    sma_20: sma20,
    sma_50: sma50,
    ema_12: ema12,
    ema_26: ema26,
    rsi_14: rsi,
    macd: macd,
    atr_14: atr,
    volume_20d_avg: avgVolume,
    current_price: currentPrice,
  }
}

function calculateEMA(prices: number[], period: number): number {
  const k = 2 / (period + 1)
  let ema = prices[0]

  for (let i = 1; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k)
  }

  return ema
}

function calculateRSI(prices: number[], period: number): number {
  let gains = 0
  let losses = 0

  for (let i = prices.length - period; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1]
    if (change > 0) gains += change
    else losses += Math.abs(change)
  }

  const avgGain = gains / period
  const avgLoss = losses / period

  if (avgLoss === 0) return 100
  const rs = avgGain / avgLoss
  return 100 - 100 / (1 + rs)
}

function calculateATR(highs: number[], lows: number[], closes: number[], period: number): number {
  const trs = []

  for (let i = 1; i < highs.length; i++) {
    const tr = Math.max(highs[i] - lows[i], Math.abs(highs[i] - closes[i - 1]), Math.abs(lows[i] - closes[i - 1]))
    trs.push(tr)
  }

  return trs.slice(-period).reduce((a, b) => a + b, 0) / period
}

function makePrediction(indicators: any, horizon: string) {
  const { sma_20, sma_50, rsi_14, macd, current_price } = indicators

  // Simple momentum-based prediction
  let signal = 0

  // Trend following
  if (current_price > sma_20) signal += 0.3
  if (current_price > sma_50) signal += 0.2
  if (sma_20 > sma_50) signal += 0.3

  // RSI
  if (rsi_14 < 30)
    signal += 0.4 // Oversold
  else if (rsi_14 > 70)
    signal -= 0.4 // Overbought
  else signal += (50 - rsi_14) * 0.01 // Neutral zone

  // MACD
  if (macd > 0) signal += 0.2
  else signal -= 0.2

  // Normalize signal to expected return
  const baseReturn = signal * 0.02 // Scale to reasonable returns

  // Adjust for horizon
  const horizonMultiplier = horizon === "1d" ? 1 : horizon === "7d" ? 3 : 10

  const predictedReturn = baseReturn * horizonMultiplier
  const predictedPrice = current_price * (1 + predictedReturn)

  // Confidence based on signal strength
  const confidence = Math.min(0.95, 0.5 + Math.abs(signal) * 0.1)

  // Confidence interval (wider for longer horizons)
  const uncertainty = 0.02 * horizonMultiplier
  const lowerBound = predictedReturn - uncertainty
  const upperBound = predictedReturn + uncertainty

  return {
    ticker: indicators.ticker,
    horizon,
    current_price,
    predicted_price: predictedPrice,
    predicted_return: predictedReturn,
    lower_bound: lowerBound,
    upper_bound: upperBound,
    confidence,
    model_version: "Technical Analysis v1.0",
    features_used: {
      RSI_14: rsi_14.toFixed(2),
      MACD: macd.toFixed(4),
      SMA_20: sma_20.toFixed(2),
      SMA_50: sma_50.toFixed(2),
      Signal: signal.toFixed(3),
    },
    timestamp: new Date().toISOString(),
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { ticker, horizon } = body

    console.log("[v0] Prediction request:", { ticker, horizon })

    if (!ticker || typeof ticker !== "string" || ticker.trim() === "") {
      return NextResponse.json({ error: "Ticker is required and must be a non-empty string" }, { status: 400 })
    }

    if (!horizon || !["1d", "7d", "30d"].includes(horizon)) {
      return NextResponse.json({ error: "Invalid horizon. Must be 1d, 7d, or 30d" }, { status: 400 })
    }

    const trimmedTicker = ticker.trim().toUpperCase()

    console.log("[v0] Validating ticker online:", trimmedTicker)
    const validation = await validateTickerOnline(trimmedTicker)

    if (!validation.valid) {
      console.log("[v0] Ticker validation failed:", validation.error)
      return NextResponse.json(
        { error: `Invalid ticker "${trimmedTicker}": ${validation.error || "Ticker does not exist"}` },
        { status: 404 },
      )
    }

    console.log("[v0] Ticker validated, fetching historical data...")

    const historicalData = await fetchRealData(trimmedTicker)

    if (!historicalData || historicalData.length < 50) {
      return NextResponse.json({ error: "Insufficient historical data for prediction" }, { status: 500 })
    }

    console.log("[v0] Calculating technical indicators...")

    const indicators = calculateIndicators(historicalData)

    if (!indicators) {
      return NextResponse.json({ error: "Could not calculate technical indicators" }, { status: 500 })
    }

    indicators.ticker = trimmedTicker

    console.log("[v0] Making prediction with indicators:", indicators)

    const prediction = makePrediction(indicators, horizon)

    console.log("[v0] Prediction generated:", prediction)

    return NextResponse.json(prediction)
  } catch (error) {
    console.error("[v0] Prediction error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
