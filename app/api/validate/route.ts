import { NextResponse } from "next/server"

const SP500_TICKERS = [
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
  "IBM",
  "GE",
  "BA",
  "CAT",
  "MMM",
  "AXP",
  "GS",
  "MS",
  "C",
  "BAC",
  "BLK",
  "SCHW",
  "CB",
  "MMC",
  "PGR",
  "TRV",
  "AIG",
  "AFL",
  "MET",
  "PRU",
  "ALL",
  "TFC",
  "USB",
  "PNC",
  "BK",
  "STT",
  "FITB",
  "HBAN",
  "RF",
  "KEY",
  "CFG",
  "CMA",
  "MTB",
  "ZION",
  "WBS",
  "ALLY",
  "WTFC",
  "FHN",
  "ONB",
  "LOW",
  "TGT",
  "TJX",
  "DG",
  "DLTR",
  "ROST",
  "BBY",
  "EBAY",
  "ETSY",
  "W",
  "BKNG",
  "MAR",
  "HLT",
  "MGM",
  "WYNN",
  "LVS",
  "DIS",
  "CHTR",
  "T",
  "TMUS",
  "VOD",
  "TEF",
  "AMX",
  "SKM",
  "CHL",
  "CHT",
  "BCE",
  "FDX",
  "XPO",
  "JBHT",
  "KNX",
  "EXPD",
  "CHRW",
  "LSTR",
  "HUBG",
  "WERN",
  "DE",
  "CAT",
  "CMI",
  "EMR",
  "ETN",
  "ITW",
  "PH",
  "ROP",
  "ROK",
  "XYL",
  "LMT",
  "NOC",
  "GD",
  "LHX",
  "HII",
  "TXT",
  "HWM",
  "KTOS",
  "PFE",
  "CVS",
  "CI",
  "HUM",
  "ANTM",
  "MOH",
  "CNC",
  "WCG",
  "GILD",
  "REGN",
  "VRTX",
  "BIIB",
  "ALXN",
  "ILMN",
  "IDXX",
  "IQV",
  "A",
  "AMGN",
  "SYK",
  "BSX",
  "MDT",
  "ISRG",
  "EW",
  "BAX",
  "BDX",
  "ZBH",
  "RMD",
  "COO",
  "ALGN",
  "CL",
  "KMB",
  "CHD",
  "CLX",
  "EL",
  "NWL",
  "SJM",
  "CPB",
  "GIS",
]

function calculateTechnicalIndicators(prices: number[], highs: number[], lows: number[], volumes: number[]) {
  const n = prices.length

  // SMA
  const sma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20
  const sma50 = prices.slice(-50).reduce((a, b) => a + b, 0) / 50
  const sma200 = n >= 200 ? prices.slice(-200).reduce((a, b) => a + b, 0) / 200 : sma50

  // EMA (proper exponential calculation)
  const calculateEMA = (data: number[], period: number) => {
    const k = 2 / (period + 1)
    let ema = data[0]
    for (let i = 1; i < data.length; i++) {
      ema = data[i] * k + ema * (1 - k)
    }
    return ema
  }

  const ema12 = calculateEMA(prices, 12)
  const ema26 = calculateEMA(prices, 26)

  // RSI (Relative Strength Index)
  const calculateRSI = (prices: number[], period: number) => {
    let gains = 0,
      losses = 0
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

  const rsi14 = calculateRSI(prices, 14)

  // MACD
  const macd = ema12 - ema26
  const macdSignal = calculateEMA(
    prices.slice(-26).map((_, i) => {
      const e12 = calculateEMA(prices.slice(0, prices.length - 26 + i + 1), 12)
      const e26 = calculateEMA(prices.slice(0, prices.length - 26 + i + 1), 26)
      return e12 - e26
    }),
    9,
  )

  // ATR (Average True Range)
  const trs = []
  for (let i = 1; i < n; i++) {
    const tr = Math.max(highs[i] - lows[i], Math.abs(highs[i] - prices[i - 1]), Math.abs(lows[i] - prices[i - 1]))
    trs.push(tr)
  }
  const atr14 = trs.slice(-14).reduce((a, b) => a + b, 0) / 14

  // Bollinger Bands
  const std20 = Math.sqrt(prices.slice(-20).reduce((sum, p) => sum + Math.pow(p - sma20, 2), 0) / 20)
  const bbUpper = sma20 + 2 * std20
  const bbLower = sma20 - 2 * std20

  // Volume indicators
  const avgVolume20 = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20

  return {
    sma20,
    sma50,
    sma200,
    ema12,
    ema26,
    rsi14,
    macd,
    macdSignal,
    atr14,
    bbUpper,
    bbLower,
    avgVolume20,
    currentPrice: prices[n - 1],
  }
}

function generatePrediction(indicators: any, horizon: number) {
  const { sma20, sma50, sma200, rsi14, macd, macdSignal, currentPrice, bbUpper, bbLower } = indicators

  // Ensemble of signals (combining trend, momentum, mean reversion)
  let trendSignal = 0
  let momentumSignal = 0
  let meanReversionSignal = 0

  // 1. Trend Following Signals
  if (currentPrice > sma20) trendSignal += 0.25
  if (currentPrice > sma50) trendSignal += 0.25
  if (currentPrice > sma200) trendSignal += 0.25
  if (sma20 > sma50) trendSignal += 0.25

  // 2. Momentum Signals
  if (macd > macdSignal) momentumSignal += 0.33
  if (rsi14 < 30)
    momentumSignal += 0.33 // Oversold
  else if (rsi14 > 70) momentumSignal -= 0.33 // Overbought
  if (rsi14 > 50) momentumSignal += 0.17

  // 3. Mean Reversion Signals
  const bbPosition = (currentPrice - bbLower) / (bbUpper - bbLower)
  if (bbPosition < 0.2)
    meanReversionSignal += 0.5 // Near lower band
  else if (bbPosition > 0.8) meanReversionSignal -= 0.5 // Near upper band

  // Weighted ensemble (60% trend, 30% momentum, 10% mean reversion)
  const signal = trendSignal * 0.6 + momentumSignal * 0.3 + meanReversionSignal * 0.1

  // Scale to expected return based on horizon
  const baseReturn = signal * 0.015 // 1.5% base
  const horizonMultiplier = Math.sqrt(horizon) // Scale by sqrt of days (volatility scaling)
  const predictedReturn = baseReturn * horizonMultiplier

  return {
    predictedReturn,
    signal,
    confidence: 0.5 + Math.abs(signal) * 0.15,
  }
}

export async function GET() {
  try {
    console.log("[v0] Starting comprehensive validation on 200 S&P 500 stocks over 10 years...")

    const metrics = {
      // Regression metrics (for price/return prediction)
      totalSamples: 0,
      squaredErrors: [] as number[],
      absoluteErrors: [] as number[],

      // Classification metrics (for direction prediction)
      truePositives: 0, // Predicted up, actual up
      trueNegatives: 0, // Predicted down, actual down
      falsePositives: 0, // Predicted up, actual down
      falseNegatives: 0, // Predicted down, actual up

      // Financial performance metrics
      returns: [] as number[],
      strategyReturns: [] as number[], // Returns when following predictions
      drawdowns: [] as number[],

      // Per-horizon tracking
      horizonMetrics: {
        "1d": { samples: 0, correct: 0, returns: [] as number[] },
        "7d": { samples: 0, correct: 0, returns: [] as number[] },
        "30d": { samples: 0, correct: 0, returns: [] as number[] },
      },
    }

    let tickersTested = 0
    const horizons = [1, 7, 30]

    const startTime = Date.now()

    for (const ticker of SP500_TICKERS) {
      try {
        if (Date.now() - startTime > 5 * 60 * 1000) {
          console.log("[v0] Validation timeout reached, completing with current samples")
          break
        }

        const end = new Date()
        const start = new Date(end.getTime() - 10 * 365 * 24 * 60 * 60 * 1000)

        let response
        try {
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 10000)

          response = await fetch(
            `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?period1=${Math.floor(start.getTime() / 1000)}&period2=${Math.floor(end.getTime() / 1000)}&interval=1d`,
            { signal: controller.signal },
          )

          clearTimeout(timeoutId)

          if (!response.ok) {
            if (response.status === 404) {
              console.log(`[v0] Skipping ${ticker}: Likely delisted or invalid`)
            } else {
              console.log(`[v0] Skipping ${ticker}: HTTP ${response.status}`)
            }
            continue
          }
        } catch (fetchError) {
          console.log(`[v0] Skipping ${ticker}: Network error or timeout`)
          continue
        }

        const data = await response.json()

        if (!data.chart?.result?.[0]?.indicators?.quote?.[0]) {
          console.log(`[v0] Skipping ${ticker}: No valid data available`)
          continue
        }

        const quotes = data.chart.result[0]
        const timestamps = quotes.timestamp
        const prices = quotes.indicators.quote[0].close.filter((p: any) => p !== null)
        const highs = quotes.indicators.quote[0].high.filter((p: any) => p !== null)
        const lows = quotes.indicators.quote[0].low.filter((p: any) => p !== null)
        const volumes = quotes.indicators.quote[0].volume.filter((p: any) => p !== null)

        if (prices.length < 250) continue // Need enough data

        // Train on past data, test on future (prevents look-ahead bias)
        for (let i = 200; i < prices.length - 30; i += 7) {
          // Step by 7 days
          for (const horizon of horizons) {
            if (i + horizon >= prices.length) continue

            const historicalPrices = prices.slice(0, i)
            const historicalHighs = highs.slice(0, i)
            const historicalLows = lows.slice(0, i)
            const historicalVolumes = volumes.slice(0, i)

            // Calculate indicators using only past data
            const indicators = calculateTechnicalIndicators(
              historicalPrices,
              historicalHighs,
              historicalLows,
              historicalVolumes,
            )

            // Generate prediction
            const prediction = generatePrediction(indicators, horizon)

            // Get actual outcome
            const currentPrice = prices[i]
            const futurePrice = prices[i + horizon]
            const actualReturn = (futurePrice - currentPrice) / currentPrice

            // Regression metrics
            const error = prediction.predictedReturn - actualReturn
            metrics.squaredErrors.push(error * error)
            metrics.absoluteErrors.push(Math.abs(error))

            // Classification metrics (direction)
            const predictedDirection = prediction.predictedReturn > 0
            const actualDirection = actualReturn > 0

            if (predictedDirection && actualDirection) metrics.truePositives++
            else if (!predictedDirection && !actualDirection) metrics.trueNegatives++
            else if (predictedDirection && !actualDirection) metrics.falsePositives++
            else metrics.falseNegatives++

            // Track returns
            metrics.returns.push(actualReturn)

            // Strategy return (only trade when confident)
            if (prediction.confidence > 0.6) {
              const strategyReturn = predictedDirection ? actualReturn : -actualReturn
              metrics.strategyReturns.push(strategyReturn)
            }

            // Per-horizon metrics
            const horizonKey = `${horizon}d` as "1d" | "7d" | "30d"
            metrics.horizonMetrics[horizonKey].samples++
            if (predictedDirection === actualDirection) {
              metrics.horizonMetrics[horizonKey].correct++
            }
            metrics.horizonMetrics[horizonKey].returns.push(actualReturn)

            metrics.totalSamples++
          }
        }

        tickersTested++

        if (metrics.totalSamples % 10000 === 0) {
          console.log(`[v0] Progress: ${metrics.totalSamples} predictions tested...`)
        }

        // Limit to prevent timeout (aim for ~100k samples)
        if (metrics.totalSamples >= 100000) break
      } catch (err) {
        continue
      }
    }

    if (metrics.totalSamples === 0) {
      return NextResponse.json({ error: "No validation data" }, { status: 500 })
    }

    // 1. Regression Metrics (RMSE, MAE, MAPE)
    const rmse = Math.sqrt(metrics.squaredErrors.reduce((a, b) => a + b, 0) / metrics.totalSamples)
    const mae = metrics.absoluteErrors.reduce((a, b) => a + b, 0) / metrics.totalSamples
    const mape =
      (metrics.absoluteErrors.reduce((sum, err, i) => sum + Math.abs(err / (metrics.returns[i] || 0.001)), 0) /
        metrics.totalSamples) *
      100

    // 2. Classification Metrics (Accuracy, Precision, Recall, F1)
    const accuracy = (metrics.truePositives + metrics.trueNegatives) / metrics.totalSamples
    const precision = metrics.truePositives / (metrics.truePositives + metrics.falsePositives)
    const recall = metrics.truePositives / (metrics.truePositives + metrics.falseNegatives)
    const f1Score = (2 * (precision * recall)) / (precision + recall)

    // 3. Financial Performance Metrics
    const avgReturn = metrics.returns.reduce((a, b) => a + b, 0) / metrics.returns.length
    const stdReturn = Math.sqrt(
      metrics.returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / metrics.returns.length,
    )

    // Sharpe Ratio (annualized, risk-free rate ~3%)
    const riskFreeRate = 0.03 / 252 // Daily
    const sharpeRatio = ((avgReturn - riskFreeRate) / stdReturn) * Math.sqrt(252)

    // Sortino Ratio (only downside deviation)
    const downsideReturns = metrics.returns.filter((r) => r < 0)
    const downsideStd =
      downsideReturns.length > 0
        ? Math.sqrt(downsideReturns.reduce((sum, r) => sum + r * r, 0) / downsideReturns.length)
        : stdReturn
    const sortinoRatio = ((avgReturn - riskFreeRate) / downsideStd) * Math.sqrt(252)

    // Strategy performance
    const strategyAvgReturn =
      metrics.strategyReturns.length > 0
        ? metrics.strategyReturns.reduce((a, b) => a + b, 0) / metrics.strategyReturns.length
        : 0
    const strategyStd =
      metrics.strategyReturns.length > 0
        ? Math.sqrt(
            metrics.strategyReturns.reduce((sum, r) => sum + Math.pow(r - strategyAvgReturn, 2), 0) /
              metrics.strategyReturns.length,
          )
        : 0
    const strategySharpe = strategyStd > 0 ? ((strategyAvgReturn - riskFreeRate) / strategyStd) * Math.sqrt(252) : 0

    // Calculate cumulative returns and max drawdown
    let cumReturn = 0
    let peak = 0
    let maxDrawdown = 0

    for (const ret of metrics.strategyReturns) {
      cumReturn = (1 + cumReturn) * (1 + ret) - 1
      if (cumReturn > peak) peak = cumReturn
      const drawdown = (peak - cumReturn) / (1 + peak)
      if (drawdown > maxDrawdown) maxDrawdown = drawdown
    }

    const roi = cumReturn
    const calmarRatio = maxDrawdown > 0 ? roi / maxDrawdown : 0

    // Per-horizon results
    const horizonResults = Object.entries(metrics.horizonMetrics).map(([horizon, data]) => ({
      horizon,
      samples: data.samples,
      accuracy: data.samples > 0 ? data.correct / data.samples : 0,
      avg_return: data.returns.length > 0 ? data.returns.reduce((a, b) => a + b, 0) / data.returns.length : 0,
    }))

    const results = {
      // Summary
      total_predictions: metrics.totalSamples,
      tickers_tested: tickersTested,
      test_period_years: 10,

      // Regression Metrics
      rmse,
      mae,
      mape,

      // Classification Metrics
      direction_accuracy: accuracy,
      precision,
      recall,
      f1Score,

      // Directional breakdown
      hit_rate: accuracy, // Same as direction accuracy
      true_positives: metrics.truePositives,
      true_negatives: metrics.trueNegatives,
      false_positives: metrics.falsePositives,
      false_negatives: metrics.falseNegatives,

      // Financial Performance
      sharpe_ratio: sharpeRatio,
      sortino_ratio: sortinoRatio,
      strategy_sharpe_ratio: strategySharpe,
      roi: roi,
      max_drawdown: maxDrawdown,
      calmar_ratio: calmarRatio,
      avg_return: avgReturn,
      std_return: stdReturn,
      strategy_avg_return: strategyAvgReturn,

      // Per-horizon breakdown
      horizon_results: horizonResults,

      // Model info
      model_version: "Enhanced Technical Analysis v2.0 (Ensemble)",
      validation_method: "Walk-Forward Time-Series Validation",
      timestamp: new Date().toISOString(),

      // Interpretation guide
      interpretation: {
        accuracy_benchmark: "53-58% is good for stock direction, >55% is excellent",
        sharpe_benchmark: ">1.0 is good, >2.0 is excellent",
        comparison: "Random guess = 50% accuracy, Buy-and-hold SPY Sharpe ~0.8",
      },
    }

    console.log("[v0] Validation complete:", {
      samples: results.total_predictions,
      accuracy: (results.direction_accuracy * 100).toFixed(2) + "%",
      sharpe: results.sharpe_ratio.toFixed(2),
      rmse: (results.rmse * 100).toFixed(2) + "%",
    })

    return NextResponse.json(results)
  } catch (error) {
    console.error("[v0] Validation error:", error)
    return NextResponse.json({ error: "Validation failed" }, { status: 500 })
  }
}
