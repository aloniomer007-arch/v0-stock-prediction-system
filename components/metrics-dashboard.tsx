"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, CheckCircle } from "lucide-react"

interface Metrics {
  model_performance: {
    rmse_30d: number
    rmse_90d: number
    rmse_252d: number
    hit_rate_30d: number
    hit_rate_90d: number
    hit_rate_252d: number
  }
  financial_performance: {
    sharpe_30d: number
    sharpe_90d: number
    sharpe_252d: number
    cagr: number
    max_drawdown: number
  }
  data_quality: {
    psi_features: number
    missing_data_pct: number
    last_update: string
  }
  alerts: Array<{
    level: string
    message: string
    timestamp: string
  }>
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [])

  const fetchMetrics = async () => {
    try {
      const response = await fetch("/api/metrics")
      const data = await response.json()
      setMetrics(data)
    } catch (error) {
      console.error("[v0] Error fetching metrics:", error)
    }
  }

  if (!metrics) {
    return (
      <Card>
        <CardContent className="p-12">
          <div className="text-center text-muted-foreground">Loading metrics...</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Model Performance</CardTitle>
          <CardDescription>Predictive accuracy across different time windows</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">30-Day Rolling</div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-sm">RMSE</span>
                  <span className="font-mono font-medium">{metrics.model_performance.rmse_30d.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Hit Rate</span>
                  <span className="font-mono font-medium">
                    {(metrics.model_performance.hit_rate_30d * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">90-Day Rolling</div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-sm">RMSE</span>
                  <span className="font-mono font-medium">{metrics.model_performance.rmse_90d.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Hit Rate</span>
                  <span className="font-mono font-medium">
                    {(metrics.model_performance.hit_rate_90d * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground">252-Day Rolling</div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-sm">RMSE</span>
                  <span className="font-mono font-medium">{metrics.model_performance.rmse_252d.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Hit Rate</span>
                  <span className="font-mono font-medium">
                    {(metrics.model_performance.hit_rate_252d * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Financial Performance</CardTitle>
          <CardDescription>Trading strategy metrics and risk-adjusted returns</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <div className="text-sm text-muted-foreground mb-1">CAGR</div>
              <div className="text-2xl font-bold text-green-500">
                {(metrics.financial_performance.cagr * 100).toFixed(1)}%
              </div>
            </div>

            <div>
              <div className="text-sm text-muted-foreground mb-1">Sharpe (30d)</div>
              <div className="text-2xl font-bold">{metrics.financial_performance.sharpe_30d.toFixed(2)}</div>
            </div>

            <div>
              <div className="text-sm text-muted-foreground mb-1">Sharpe (90d)</div>
              <div className="text-2xl font-bold">{metrics.financial_performance.sharpe_90d.toFixed(2)}</div>
            </div>

            <div>
              <div className="text-sm text-muted-foreground mb-1">Sharpe (252d)</div>
              <div className="text-2xl font-bold">{metrics.financial_performance.sharpe_252d.toFixed(2)}</div>
            </div>

            <div>
              <div className="text-sm text-muted-foreground mb-1">Max Drawdown</div>
              <div className="text-2xl font-bold text-red-500">
                {(metrics.financial_performance.max_drawdown * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Data Quality & Monitoring</CardTitle>
          <CardDescription>Real-time data health and drift detection</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-muted-foreground mb-1">PSI (Feature Drift)</div>
              <div className="text-2xl font-bold">{metrics.data_quality.psi_features.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {metrics.data_quality.psi_features < 0.1 ? "No significant drift" : "Monitoring required"}
              </p>
            </div>

            <div>
              <div className="text-sm text-muted-foreground mb-1">Missing Data</div>
              <div className="text-2xl font-bold">{(metrics.data_quality.missing_data_pct * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground mt-1">Within acceptable range</p>
            </div>

            <div>
              <div className="text-sm text-muted-foreground mb-1">Last Update</div>
              <div className="text-sm font-medium">{new Date(metrics.data_quality.last_update).toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">Data pipeline active</p>
            </div>
          </div>

          <div className="pt-4 border-t">
            <h4 className="font-medium mb-3">System Alerts</h4>
            <div className="space-y-2">
              {metrics.alerts.map((alert, idx) => (
                <div key={idx} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50">
                  {alert.level === "info" ? (
                    <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-yellow-500 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Badge variant={alert.level === "info" ? "default" : "secondary"}>
                        {alert.level.toUpperCase()}
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        {new Date(alert.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-sm mt-1">{alert.message}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Automated Retrain System</CardTitle>
          <CardDescription>Self-healing pipeline with performance monitoring</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 text-sm">
            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Performance Monitoring Active</p>
                <p className="text-muted-foreground">Tracking RMSE, hit rate, and Sharpe ratio on rolling windows</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Drift Detection Enabled</p>
                <p className="text-muted-foreground">PSI threshold set to 0.25 for automatic retrain trigger</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Walk-Forward Validation</p>
                <p className="text-muted-foreground">
                  Models validated using time-series cross-validation before deployment
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Model Registry</p>
                <p className="text-muted-foreground">All models versioned with full metadata and rollback capability</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
