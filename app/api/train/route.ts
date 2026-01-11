import { NextResponse } from "next/server"

export async function POST() {
  try {
    // In a real system, this would read from data/results.json
    // For now, return mock results after training completes
    const mockResults = {
      direction_accuracy: 53.8,
      rmse: 0.023456,
      mae: 0.018234,
      baseline_direction: 50.2,
      improvement: 3.6,
      model_type: "LightGBM",
      test_samples: 4523,
      tickers: 30,
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json({
      success: true,
      results: mockResults,
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: "Training results not found",
      needsTraining: true,
    })
  }
}

export async function GET() {
  try {
    const mockResults = {
      direction_accuracy: 53.8,
      rmse: 0.023456,
      mae: 0.018234,
      baseline_direction: 50.2,
      improvement: 3.6,
      model_type: "LightGBM",
      test_samples: 4523,
      tickers: 30,
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json({
      success: true,
      results: mockResults,
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: "Model not trained yet",
      needsTraining: true,
    })
  }
}
