import { describe, expect, it } from "vitest"

import {
  AUTO_REFRESH_INTERVAL_MS,
  canRunAutoRefresh,
  canReuseInFlightRefresh,
  formatRefreshWarningMessage,
  formatRefreshMode
} from "./refresh"

describe("dashboard refresh helpers", () => {
  it("keeps auto refresh opt-in and pauses while hidden", () => {
    expect(AUTO_REFRESH_INTERVAL_MS).toBeGreaterThanOrEqual(10000)
    expect(canRunAutoRefresh(false, "visible")).toBe(false)
    expect(canRunAutoRefresh(true, "hidden")).toBe(false)
    expect(canRunAutoRefresh(true, "visible")).toBe(true)
  })

  it("renders concise refresh mode labels", () => {
    expect(formatRefreshMode(false, "visible")).toBe("manual refresh")
    expect(formatRefreshMode(true, "hidden")).toBe("auto paused")
    expect(formatRefreshMode(true, "visible")).toBe("auto 10s")
  })

  it("does not reuse stale in-flight refreshes after mutations", () => {
    const inFlightRefresh = Promise.resolve()

    expect(canReuseInFlightRefresh(null, false)).toBe(false)
    expect(canReuseInFlightRefresh(inFlightRefresh, false)).toBe(true)
    expect(canReuseInFlightRefresh(inFlightRefresh, true)).toBe(false)
  })

  it("formats refresh failures from non-Error rejections", () => {
    expect(formatRefreshWarningMessage(new Error("service unavailable"))).toBe(
      "Refresh warning: service unavailable"
    )
    expect(formatRefreshWarningMessage("network offline")).toBe(
      "Refresh warning: network offline"
    )
    expect(formatRefreshWarningMessage(null)).toBe("Refresh warning: unknown error")
  })
})
