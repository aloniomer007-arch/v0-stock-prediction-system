const cache = new Map<string, { data: any; timestamp: number }>()
const CACHE_TTL = 300000 // 5 minutes cache (increased from 1 minute)

// Global rate limiter for Yahoo Finance
let yahooLastRequest = 0
const YAHOO_MIN_INTERVAL = 5000 // 5 seconds between ANY Yahoo request

const circuitBreakers = new Map<string, { failures: number; lockedUntil: number }>()
const MAX_FAILURES = 3
const CIRCUIT_BREAKER_DURATION = 30 * 60 * 1000 // 30 minutes

function checkCircuitBreaker(ticker: string): boolean {
  const breaker = circuitBreakers.get(ticker)
  if (!breaker) return true

  if (Date.now() > breaker.lockedUntil) {
    circuitBreakers.delete(ticker)
    return true
  }

  if (breaker.failures >= MAX_FAILURES) {
    console.log(
      `[v0] Circuit breaker active for ${ticker}, locked until ${new Date(breaker.lockedUntil).toLocaleTimeString()}`,
    )
    return false
  }

  return true
}

function recordFailure(ticker: string) {
  const breaker = circuitBreakers.get(ticker) || { failures: 0, lockedUntil: 0 }
  breaker.failures += 1

  if (breaker.failures >= MAX_FAILURES) {
    breaker.lockedUntil = Date.now() + CIRCUIT_BREAKER_DURATION
    console.log(`[v0] Circuit breaker triggered for ${ticker} after ${breaker.failures} failures`)
  }

  circuitBreakers.set(ticker, breaker)
}

function recordSuccess(ticker: string) {
  circuitBreakers.delete(ticker)
}

async function waitForYahooRateLimit() {
  const now = Date.now()
  const timeSinceLastRequest = now - yahooLastRequest

  if (timeSinceLastRequest < YAHOO_MIN_INTERVAL) {
    const waitTime = YAHOO_MIN_INTERVAL - timeSinceLastRequest
    await new Promise((resolve) => setTimeout(resolve, waitTime))
  }

  yahooLastRequest = Date.now()
}

async function fetchFromYahoo(ticker: string): Promise<any> {
  await waitForYahooRateLimit()

  const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=1d`, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      Accept: "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`${response.status}`)
  }

  const data = await response.json()
  if (!data?.chart?.result?.[0]) throw new Error("Invalid Yahoo data")

  return {
    price: data.chart.result[0].meta.regularMarketPrice,
    source: "yahoo",
  }
}

export async function fetchStockPrice(
  ticker: string,
): Promise<{ price: number; source: string; stale?: boolean; ageMinutes?: number }> {
  if (!checkCircuitBreaker(ticker)) {
    throw new Error(`Circuit breaker active for ${ticker}`)
  }

  const cacheKey = `price_${ticker}`
  const now = Date.now()

  // Check cache
  const cached = cache.get(cacheKey)
  if (cached && now - cached.timestamp < CACHE_TTL) {
    console.log(`[v0] Using cached price for ${ticker}: $${cached.data.price}`)
    return cached.data
  }

  try {
    const result = await fetchFromYahoo(ticker)
    cache.set(cacheKey, { data: result, timestamp: now })
    recordSuccess(ticker)
    console.log(`[v0] Fetched fresh price for ${ticker}: $${result.price}`)
    return result
  } catch (error) {
    recordFailure(ticker)
    if (cached) {
      const ageMinutes = Math.floor((now - cached.timestamp) / 60000)
      console.log(`[v0] Using stale cache for ${ticker} (${ageMinutes} minutes old)`)
      return { ...cached.data, stale: true, ageMinutes }
    }
    throw new Error(`Failed to fetch ${ticker}: ${(error as Error).message}`)
  }
}

export async function fetchHistoricalData(ticker: string, period: "1y" | "5y" = "1y"): Promise<any> {
  const cacheKey = `history_${ticker}_${period}`
  const now = Date.now()

  // Check cache
  const cached = cache.get(cacheKey)
  if (cached && now - cached.timestamp < CACHE_TTL) {
    return cached.data
  }

  await waitForYahooRateLimit()

  const endDate = Math.floor(Date.now() / 1000)
  const years = period === "1y" ? 1 : 5
  const startDate = endDate - years * 365 * 24 * 60 * 60

  const response = await fetch(
    `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?period1=${startDate}&period2=${endDate}&interval=1d`,
    {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        Accept: "application/json",
      },
    },
  )

  if (!response.ok) throw new Error(`Yahoo Finance historical error: ${response.status}`)

  const data = await response.json()
  const result = data?.chart?.result?.[0]

  if (!result || !result.indicators?.quote?.[0]) throw new Error("Invalid historical data")

  cache.set(cacheKey, { data: result, timestamp: now })
  return result
}
