// Professional day trader configuration and constraints
export const DAY_TRADER_CONFIG = {
  // Trading frequency
  MAX_TRADES_PER_DAY: 20,
  MAX_TRADES_PER_SYMBOL: 3,
  TARGET_TRADES_PER_DAY: { min: 5, max: 20 },

  // Risk management
  MAX_POSITION_SIZE_PCT: 0.2, // Max 20% of capital per position
  MAX_DAILY_LOSS_PCT: 0.03, // Stop trading if down 3% for the day
  MAX_TOTAL_EXPOSURE_PCT: 0.95, // Max 95% of capital deployed
  VOLATILITY_POSITION_SIZING: true,

  // Performance targets
  MIN_PROFIT_FACTOR: 1.2,
  MIN_WIN_RATE: 0.52, // 52%
  MIN_EDGE_PER_TRADE: 0.015, // 1.5% minimum expected gain

  // Market interaction
  LIQUID_HOURS: {
    morning: { start: 9.5, end: 16 }, // 9:30 AM - 4:00 PM ET (full trading day)
    afternoon: { start: 9.5, end: 16 }, // Same range for consistency
  },
  USE_LIMIT_ORDERS: true,
  MAX_SLIPPAGE_BPS: 10, // 10 basis points

  // Constraints
  NO_OVERNIGHT_POSITIONS: true,
  MAX_HOLDING_TIME_MINUTES: 240, // 4 hours max
  MAX_SIMULTANEOUS_POSITIONS: 5,
  MIN_CONFIDENCE_THRESHOLD: 65,

  // Monitoring
  CALCULATE_POSITION_CORRELATION: true,
  TRACK_LATENCY: true,
  LOG_SLIPPAGE: true,
}

export function isLiquidTradingTime(): boolean {
  const now = new Date()
  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }))
  const hour = et.getHours() + et.getMinutes() / 60

  const isWeekday = et.getDay() >= 1 && et.getDay() <= 5
  const isMarketHours = hour >= 9.5 && hour <= 16

  return isWeekday && isMarketHours
}

export function calculatePositionSize(
  confidence: number,
  volatility: number,
  accountValue: number,
): { size: number; reasoning: string } {
  // Kelly Criterion with half-Kelly for safety
  const winProb = confidence / 100
  const lossProb = 1 - winProb
  const winLossRatio = 2.0 // Assume 2:1 reward/risk

  let kellyFraction = (winProb * winLossRatio - lossProb) / winLossRatio
  kellyFraction = Math.max(0, Math.min(kellyFraction * 0.5, DAY_TRADER_CONFIG.MAX_POSITION_SIZE_PCT))

  // Adjust for volatility (higher volatility = smaller position)
  const volatilityAdjustment = Math.max(0.5, 1 - volatility / 0.5)
  const adjustedSize = kellyFraction * volatilityAdjustment

  const positionSize = accountValue * adjustedSize

  return {
    size: positionSize,
    reasoning: `Kelly: ${(kellyFraction * 100).toFixed(1)}%, Vol adj: ${(volatilityAdjustment * 100).toFixed(1)}%`,
  }
}

export function calculateVolatility(prices: number[]): number {
  if (prices.length < 2) return 0

  const returns = []
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1])
  }

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length
  const stdDev = Math.sqrt(variance)

  return stdDev * Math.sqrt(252) // Annualized volatility
}
