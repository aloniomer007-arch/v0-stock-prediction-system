import { fetchHistoricalData } from "@/lib/multi-source-data"

export async function getPrediction(ticker: string): Promise<{ prediction: number; confidence: number } | null> {
  try {
    const result = await fetchHistoricalData(ticker, "1y")

    const timestamps = result.timestamp
    const quotes = result.indicators.quote[0]
    const closes = quotes.close.filter((c: number) => c !== null)
    const highs = quotes.high.filter((h: number) => h !== null)
    const lows = quotes.low.filter((l: number) => l !== null)

    if (closes.length < 50) return null

    // Calculate SMA indicators
    const sma20 = closes.slice(-20).reduce((a: number, b: number) => a + b, 0) / 20
    const sma50 = closes.slice(-50).reduce((a: number, b: number) => a + b, 0) / 50

    // Calculate EMA12
    const k = 2 / 13
    let ema12 = closes[0]
    for (let i = 1; i < closes.length; i++) {
      ema12 = closes[i] * k + ema12 * (1 - k)
    }

    // Calculate EMA26
    const k26 = 2 / 27
    let ema26 = closes[0]
    for (let i = 1; i < closes.length; i++) {
      ema26 = closes[i] * k26 + ema26 * (1 - k26)
    }

    // Calculate RSI
    let gains = 0
    let losses = 0
    for (let i = closes.length - 14; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1]
      if (change > 0) gains += change
      else losses += Math.abs(change)
    }
    const rsi = 100 - 100 / (1 + gains / losses)

    // Calculate MACD
    const macd = ema12 - ema26
    const currentPrice = closes[closes.length - 1]

    // Generate trading signal
    let signal = 0
    if (currentPrice > sma20) signal += 0.3
    if (currentPrice > sma50) signal += 0.2
    if (sma20 > sma50) signal += 0.3
    if (rsi < 30) signal += 0.4
    else if (rsi > 70) signal -= 0.4
    else signal += (50 - rsi) * 0.01
    if (macd > 0) signal += 0.2
    else signal -= 0.2

    // Calculate prediction
    const baseReturn = signal * 0.02
    const predictedPrice = currentPrice * (1 + baseReturn)
    const confidence = Math.min(95, 50 + Math.abs(signal) * 10)

    return {
      prediction: predictedPrice,
      confidence: confidence,
    }
  } catch (error) {
    console.error(`[v0] Prediction error for ${ticker}:`, error)
    return null
  }
}
