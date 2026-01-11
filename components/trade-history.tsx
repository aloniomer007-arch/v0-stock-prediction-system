"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface Trade {
  id: number
  date: string
  ticker: string
  action: string
  quantity: number
  price: number
  exec_price: number
  slippage: number
  commission: number
  total_cost: number
  reason: string
}

export default function TradeHistory() {
  const [trades, setTrades] = useState<Trade[]>([])

  useEffect(() => {
    fetchTrades()
  }, [])

  const fetchTrades = async () => {
    try {
      const response = await fetch("/api/trades")
      const data = await response.json()
      setTrades(data.trades)
    } catch (error) {
      console.error("[v0] Error fetching trades:", error)
    }
  }

  if (trades.length === 0) {
    return (
      <Card>
        <CardContent className="p-12">
          <div className="text-center text-muted-foreground">No trades yet</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Trade History</CardTitle>
        <CardDescription>Complete log of all executed trades in the simulator</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="grid grid-cols-9 gap-4 pb-2 border-b font-medium text-sm">
            <div>Date</div>
            <div>Ticker</div>
            <div>Action</div>
            <div className="text-right">Qty</div>
            <div className="text-right">Price</div>
            <div className="text-right">Exec Price</div>
            <div className="text-right">Slippage</div>
            <div className="text-right">Total</div>
            <div>Reason</div>
          </div>

          {trades.map((trade) => (
            <div key={trade.id} className="grid grid-cols-9 gap-4 py-2 border-b text-sm">
              <div className="text-muted-foreground">{trade.date}</div>
              <div className="font-medium">{trade.ticker}</div>
              <div>
                <Badge variant={trade.action === "BUY" ? "default" : "secondary"}>{trade.action}</Badge>
              </div>
              <div className="text-right">{trade.quantity}</div>
              <div className="text-right font-mono">${trade.price.toFixed(2)}</div>
              <div className="text-right font-mono">${trade.exec_price.toFixed(2)}</div>
              <div className="text-right text-muted-foreground">${trade.slippage.toFixed(2)}</div>
              <div className="text-right font-medium">${trade.total_cost.toFixed(2)}</div>
              <div className="text-muted-foreground text-xs">{trade.reason}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
