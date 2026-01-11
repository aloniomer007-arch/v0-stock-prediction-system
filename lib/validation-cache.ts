interface ValidationCache {
  results: any
  timestamp: number
  expiresAt: number
}

const CACHE_DURATION = 24 * 60 * 60 * 1000 // 24 hours

export function getCachedValidation(): any | null {
  if (typeof window === "undefined") return null

  try {
    const cached = localStorage.getItem("validation_cache")
    if (!cached) return null

    const data: ValidationCache = JSON.parse(cached)

    if (Date.now() > data.expiresAt) {
      localStorage.removeItem("validation_cache")
      return null
    }

    return data.results
  } catch {
    return null
  }
}

export function setCachedValidation(results: any): void {
  if (typeof window === "undefined") return

  try {
    const cache: ValidationCache = {
      results,
      timestamp: Date.now(),
      expiresAt: Date.now() + CACHE_DURATION,
    }

    localStorage.setItem("validation_cache", JSON.stringify(cache))
  } catch (error) {
    console.error("[v0] Failed to cache validation:", error)
  }
}

export function clearValidationCache(): void {
  if (typeof window === "undefined") return
  localStorage.removeItem("validation_cache")
}
