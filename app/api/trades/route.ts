import { NextResponse } from "next/server"
import { getAllTrades } from "@/lib/persistent-portfolio"

export async function GET() {
  try {
    const trades = getAllTrades()

    console.log("[v0] Fetching trade history:", { count: trades.length })

    return NextResponse.json({ trades })
  } catch (error) {
    console.error("[v0] Trades error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
