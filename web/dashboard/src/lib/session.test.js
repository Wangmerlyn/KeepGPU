import { describe, expect, it } from "vitest"

import {
  buildSessionPayload,
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
    expect(parseBusyThreshold("-1")).toBe(-1)
    expect(() => parseBusyThreshold("-2")).toThrow()
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
})
