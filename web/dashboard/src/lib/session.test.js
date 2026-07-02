import { describe, expect, it } from "vitest"

import {
  buildSessionPayload,
  formatBusyThresholdLabel,
  formatUtilizationLabel,
  formatUtilizationWidth,
  formatGpuIdentity,
  formatSessionState,
  formatSessionStateDetail,
  formatStopResultMessage,
  getRenderableGpus,
  hasReleasableSessions,
  isSessionStopping,
  parseBusyThreshold,
  parseGpuIds,
  parsePositiveNumber,
  summarizeDashboardStats
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

  it("names visible ordinals in validation errors", () => {
    expect(() => parseGpuIds("0,a")).toThrow(
      "GPU IDs must be comma-separated visible ordinals"
    )
  })
})

describe("numeric parsing", () => {
  it("validates interval", () => {
    expect(parsePositiveNumber("5", "Interval")).toBe(5)
    expect(parsePositiveNumber("0.5", "Interval")).toBe(0.5)
    expect(parsePositiveNumber("1e+3", "Interval")).toBe(1000)
    expect(() => parsePositiveNumber("0", "Interval")).toThrow()
    expect(() => parsePositiveNumber("+1", "Interval")).toThrow()
    expect(() => parsePositiveNumber("１２３", "Interval")).toThrow()
    expect(() => parsePositiveNumber("1_000", "Interval")).toThrow()
    expect(() => parsePositiveNumber("NaN", "Interval")).toThrow()
    expect(() => parsePositiveNumber("Infinity", "Interval")).toThrow()
  })

  it("validates busy threshold", () => {
    expect(parseBusyThreshold("25")).toBe(25)
    expect(parseBusyThreshold("100")).toBe(100)
    expect(parseBusyThreshold("-1")).toBe(-1)
    expect(() => parseBusyThreshold("+25")).toThrow()
    expect(() => parseBusyThreshold("-0")).toThrow()
    expect(() => parseBusyThreshold("１２")).toThrow()
    expect(() => parseBusyThreshold("1_0")).toThrow()
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

  it("keeps fractional interval seconds in the payload", () => {
    expect(
      buildSessionPayload({
        gpuIds: "",
        vram: "1GiB",
        interval: "0.5",
        busyThreshold: "25"
      })
    ).toMatchObject({
      interval: 0.5
    })
  })
})

describe("formatGpuIdentity", () => {
  it("labels the visible GPU ordinal first and keeps physical metadata secondary", () => {
    expect(formatGpuIdentity({ id: 0, visible_id: 0, physical_id: 2 })).toBe(
      "GPU 0 (physical 2)"
    )
  })

  it("falls back to the public id when no physical metadata is present", () => {
    expect(formatGpuIdentity({ id: 1 })).toBe("GPU 1")
  })
})

describe("dashboard telemetry summary", () => {
  it("uses empty defaults when telemetry inputs are omitted", () => {
    expect(summarizeDashboardStats()).toEqual({
      gpuCount: 0,
      trackedCount: 0,
      averageUtilization: null
    })
  })

  it("does not turn fully unknown utilization into idle utilization", () => {
    expect(
      summarizeDashboardStats(
        [
          { id: 0, utilization: null },
          { id: 1 },
          { id: 2, utilization: Number.NaN }
        ],
        [{ job_id: "job-a" }]
      )
    ).toEqual({
      gpuCount: 3,
      trackedCount: 1,
      averageUtilization: null
    })
  })

  it("averages only known finite utilization readings", () => {
    expect(
      summarizeDashboardStats(
        [
          { id: 0, utilization: 10 },
          { id: 1, utilization: null },
          { id: 2, utilization: 30 }
        ],
        [{ job_id: "job-a" }, { job_id: "job-b" }]
      )
    ).toEqual({
      gpuCount: 3,
      trackedCount: 2,
      averageUtilization: 20
    })
  })
})

describe("getRenderableGpus", () => {
  it("filters nullish telemetry records before rendering", () => {
    const gpu = { id: 0, utilization: 42 }

    expect(getRenderableGpus([null, gpu, undefined])).toEqual([gpu])
  })
})

describe("formatUtilizationLabel", () => {
  it("labels unavailable utilization without appending a percent sign", () => {
    expect(formatUtilizationLabel(null)).toBe("n/a")
    expect(formatUtilizationLabel(undefined)).toBe("n/a")
    expect(formatUtilizationLabel(Number.NaN)).toBe("n/a")
  })

  it("labels known numeric utilization as a percentage", () => {
    expect(formatUtilizationLabel(0)).toBe("0%")
    expect(formatUtilizationLabel(42)).toBe("42%")
  })
})

describe("formatUtilizationWidth", () => {
  it("does not render unavailable telemetry as an idle-width bar", () => {
    expect(formatUtilizationWidth(null)).toBeNull()
    expect(formatUtilizationWidth(undefined)).toBeNull()
    expect(formatUtilizationWidth(Number.NaN)).toBeNull()
  })

  it("clamps known numeric utilization to a percentage width", () => {
    expect(formatUtilizationWidth(-5)).toBe("0%")
    expect(formatUtilizationWidth(42)).toBe("42%")
    expect(formatUtilizationWidth(125)).toBe("100%")
  })
})

describe("formatBusyThresholdLabel", () => {
  it("labels unconditional keepalive mode semantically", () => {
    expect(formatBusyThresholdLabel(-1)).toBe("unconditional")
  })

  it("labels normal utilization thresholds as percentages", () => {
    expect(formatBusyThresholdLabel(0)).toBe("0%")
    expect(formatBusyThresholdLabel(25)).toBe("25%")
    expect(formatBusyThresholdLabel(100)).toBe("100%")
  })

  it("labels unavailable thresholds as n/a", () => {
    expect(formatBusyThresholdLabel(null)).toBe("n/a")
    expect(formatBusyThresholdLabel(undefined)).toBe("n/a")
    expect(formatBusyThresholdLabel(NaN)).toBe("n/a")
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
    expect(formatSessionState({ state: "starting" })).toBe("Starting")
    expect(formatSessionState({ state: "stopping" })).toBe("Releasing")
    expect(formatSessionState({ state: "runtime_failed" })).toBe("Runtime failed")
    expect(formatSessionState({ state: "stop_failed" })).toBe("Release failed")
  })

  it("surfaces starting session details", () => {
    expect(formatSessionStateDetail({ state: "starting" })).toBe(
      "Controller startup is still in progress."
    )
  })

  it("surfaces retained release error details", () => {
    expect(
      formatSessionStateDetail({
        state: "stop_failed",
        last_error: "release exploded"
      })
    ).toBe("release exploded")
  })

  it("surfaces retained runtime failure details", () => {
    expect(
      formatSessionStateDetail({
        state: "runtime_failed",
        last_error: "allocation retries exhausted"
      })
    ).toBe("allocation retries exhausted")
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

  it("reports backend stop messages when no result lists are populated", () => {
    expect(formatStopResultMessage({ message: "job_id not found" })).toBe(
      "job_id not found"
    )
  })
})
