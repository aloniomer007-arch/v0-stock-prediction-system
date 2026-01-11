import { NextResponse } from "next/server"

export async function GET() {
  try {
    // In production, this would query actual monitoring metrics
    // For now, return mock metrics
    const mockMetrics = {
      model_performance: {
        rmse_30d: 0.0245,
        rmse_90d: 0.0268,
        rmse_252d: 0.0312,
        hit_rate_30d: 0.58,
        hit_rate_90d: 0.56,
        hit_rate_252d: 0.54,
      },
      financial_performance: {
        sharpe_30d: 1.42,
        sharpe_90d: 1.28,
        sharpe_252d: 1.15,
        cagr: 0.187,
        max_drawdown: -0.089,
      },
      data_quality: {
        psi_features: 0.12,
        missing_data_pct: 0.02,
        last_update: new Date().toISOString(),
      },
      alerts: [
        {
          level: "info",
          message: "Model performance stable",
          timestamp: new Date(Date.now() - 3600000).toISOString(),
        },
      ],
    }

    return NextResponse.json(mockMetrics)
  } catch (error) {
    console.error("[v0] Metrics error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
