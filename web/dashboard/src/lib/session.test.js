import { describe, expect, it } from "vitest"

import {
  buildSessionPayload,
  formatSessionState,
  formatSessionStateDetail,
  formatStopResultMessage,
  hasReleasableSessions,
  isSessionStopping,
  parseBusyThreshold,
  parseGpuIds,
  parsePositiveInt
} from "./session"

describe("parseGpuIds", () => {
  it("returns null for empty input", () => {
    expect(parseGpuIds("   ")).toBeNull()
  })

  it("parses comma-separated integers", () => {
    expect(parseGpuIds("0,1,7")).toEqual([0, 1, 7])
  })

  it("throws on invalid tokens", () => {
    expect(() => parseGpuIds("0,,2")).toThrow()
    expect(() => parseGpuIds("1,")).toThrow()
    expect(() => parseGpuIds("0,a")).toThrow()
    expect(() => parseGpuIds("-1")).toThrow()
  })
})

describe("numeric parsing", () => {
  it("validates interval", () => {
    expect(parsePositiveInt("5", "Interval")).toBe(5)
    expect(() => parsePositiveInt("0", "Interval")).toThrow()
  })

  it("validates busy threshold", () => {
    expect(parseBusyThreshold("25")).toBe(25)
    expect(parseBusyThreshold("100")).toBe(100)
    expect(parseBusyThreshold("-1")).toBe(-1)
    expect(() => parseBusyThreshold("-2")).toThrow()
    expect(() => parseBusyThreshold("101")).toThrow()
    expect(() => parseBusyThreshold("")).toThrow()
    expect(() => parseBusyThreshold("   ")).toThrow()
    expect(() => parseBusyThreshold(true)).toThrow()
    expect(() => parseBusyThreshold(false)).toThrow()
  })
})

describe("buildSessionPayload", () => {
  it("builds a normalized payload", () => {
    expect(
      buildSessionPayload({
        gpuIds: "0,1",
        vram: " 1GiB ",
        interval: "120",
        busyThreshold: "15"
      })
    ).toEqual({
      gpu_ids: [0, 1],
      vram: "1GiB",
      interval: 120,
      busy_threshold: 15
    })
  })
})

describe("isSessionStopping", () => {
  it("only disables affected session unless stop-all is active", () => {
    const stoppingIds = new Set(["job-a"])
    expect(isSessionStopping("job-a", stoppingIds, false)).toBe(true)
    expect(isSessionStopping("job-b", stoppingIds, false)).toBe(false)
    expect(isSessionStopping("job-b", stoppingIds, true)).toBe(true)
  })

  it("treats backend stopping sessions as stopping after refresh", () => {
    expect(
      isSessionStopping({ job_id: "job-a", state: "stopping" }, new Set(), false)
    ).toBe(true)
  })
})

describe("hasReleasableSessions", () => {
  it("disables stop-all when every tracked session is already stopping", () => {
    expect(
      hasReleasableSessions(
        [
          { job_id: "job-a", state: "stopping" },
          { job_id: "job-b", state: "stopping" }
        ],
        new Set(),
        false
      )
    ).toBe(false)
  })

  it("allows stop-all when a retained failed session can be retried", () => {
    expect(
      hasReleasableSessions(
        [
          { job_id: "job-a", state: "stopping" },
          { job_id: "job-b", state: "stop_failed" }
        ],
        new Set(),
        false
      )
    ).toBe(true)
  })
})

describe("session state formatting", () => {
  it("labels backend lifecycle states for display", () => {
    expect(formatSessionState({ state: "active" })).toBe("Active")
    expect(formatSessionState({ state: "stopping" })).toBe("Releasing")
    expect(formatSessionState({ state: "stop_failed" })).toBe("Release failed")
  })

  it("surfaces retained release error details", () => {
    expect(
      formatSessionStateDetail({
        state: "stop_failed",
        last_error: "release exploded"
      })
    ).toBe("release exploded")
  })
})

describe("formatStopResultMessage", () => {
  it("reports released sessions when every stop succeeded", () => {
    expect(formatStopResultMessage({ stopped: ["job-a", "job-b"] })).toBe(
      "Released sessions: job-a, job-b."
    )
  })

  it("reports timed-out sessions instead of claiming full success", () => {
    expect(
      formatStopResultMessage({ stopped: ["job-a"], timed_out: ["job-b"] })
    ).toBe("Timed out stopping session: job-b. Released session: job-a.")
  })

  it("reports failed sessions with backend errors", () => {
    expect(
      formatStopResultMessage({
        stopped: [],
        failed: ["job-a"],
        errors: ["job-a: release raised RuntimeError"]
      })
    ).toBe(
      "Failed to release session: job-a. Errors: job-a: release raised RuntimeError."
    )
  })

  it("reports when no sessions were released", () => {
    expect(formatStopResultMessage({ stopped: [], timed_out: [], failed: [] })).toBe(
      "No sessions were released."
    )
  })
})
