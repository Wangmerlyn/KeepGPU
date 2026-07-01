import { describe, expect, it } from "vitest"

import {
  AUTO_REFRESH_INTERVAL_MS,
  canRunAutoRefresh,
  canReuseInFlightRefresh,
  fetchDashboardPayloads,
  formatRefreshWarningMessage,
  formatRefreshMode,
  nextRefreshMessage
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

  it("keeps session payloads when telemetry refresh fails", async () => {
    const calls = []
    const requestJson = async (method, path) => {
      calls.push([method, path])
      if (path === "/api/gpus") {
        throw new Error("telemetry unavailable")
      }
      return { active_jobs: [{ job_id: "job-a" }] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      gpus: null,
      sessions: [{ job_id: "job-a" }],
      warning: "Refresh warning: telemetry unavailable"
    })
    expect(calls).toEqual([
      ["GET", "/api/gpus"],
      ["GET", "/api/sessions"]
    ])
  })

  it("keeps telemetry payloads when session refresh fails", async () => {
    const requestJson = async (_method, path) => {
      if (path === "/api/sessions") {
        throw new Error("sessions unavailable")
      }
      return { gpus: [{ id: 0, utilization: 12 }] }
    }

    await expect(fetchDashboardPayloads(requestJson)).resolves.toEqual({
      gpus: [{ id: 0, utilization: 12 }],
      sessions: null,
      warning: "Refresh warning: sessions unavailable"
    })
  })

  it("preserves mutation result messages when follow-up refresh warns", () => {
    expect(nextRefreshMessage({ userInitiated: true })).toBe("Dashboard refreshed.")
    expect(nextRefreshMessage({ warning: "Refresh warning: telemetry unavailable" })).toBe(
      "Refresh warning: telemetry unavailable"
    )
    expect(
      nextRefreshMessage({
        afterMutation: true,
        previousMessage: "Released session: job-a.",
        warning: "Refresh warning: telemetry unavailable"
      })
    ).toBe("Released session: job-a.")
  })
})
