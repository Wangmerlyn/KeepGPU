import { afterEach, describe, expect, it, vi } from "vitest"

import { formatApiErrorMessage, requestJson } from "./api"

describe("formatApiErrorMessage", () => {
  it("uses structured REST error messages instead of raw JSON", () => {
    expect(
      formatApiErrorMessage(
        '{"error":{"message":"Bad request: interval must be an integer >= 1"}}',
        400
      )
    ).toBe("Bad request: interval must be an integer >= 1")
  })

  it("supports compact error strings and top-level messages", () => {
    expect(formatApiErrorMessage('{"error":"Service is warming up"}', 503)).toBe(
      "Service is warming up"
    )
    expect(formatApiErrorMessage('{"message":"Dashboard route unavailable"}', 404)).toBe(
      "Dashboard route unavailable"
    )
  })

  it("falls back without hiding non-JSON bodies or empty responses", () => {
    expect(formatApiErrorMessage("upstream gateway failed", 502)).toBe(
      "upstream gateway failed"
    )
    expect(formatApiErrorMessage('{"detail":"route not found"}', 404)).toBe(
      '{"detail":"route not found"}'
    )
    expect(formatApiErrorMessage("", 500)).toBe("Request failed (500)")
    expect(formatApiErrorMessage('{"error":{"message":"   "}}', 400)).toBe(
      "Request failed (400)"
    )
  })

  it("throws only the human message from structured REST failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 400,
        text: async () =>
          '{"error":{"message":"Bad request: eco mode unavailable"}}'
      }))
    )

    await expect(requestJson("POST", "/api/sessions", { interval: 0 })).rejects.toThrow(
      "Bad request: eco mode unavailable"
    )

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ interval: 0 })
      })
    )
  })

  it("returns parsed JSON for successful REST responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ active_jobs: [] })
      }))
    )

    await expect(requestJson("GET", "/api/sessions")).resolves.toEqual({
      active_jobs: []
    })

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions",
      expect.objectContaining({ body: undefined })
    )
  })

  it("serializes provided falsy request bodies", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ ok: true })
      }))
    )

    for (const value of [0, false, "", null]) {
      await requestJson("POST", "/api/sessions", value)
    }

    expect(fetch).toHaveBeenNthCalledWith(
      1,
      "/api/sessions",
      expect.objectContaining({ body: "0" })
    )
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      "/api/sessions",
      expect.objectContaining({ body: "false" })
    )
    expect(fetch).toHaveBeenNthCalledWith(
      3,
      "/api/sessions",
      expect.objectContaining({ body: '""' })
    )
    expect(fetch).toHaveBeenNthCalledWith(
      4,
      "/api/sessions",
      expect.objectContaining({ body: "null" })
    )
  })

  it("maps aborted requests to a timeout message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        const error = new Error("aborted")
        error.name = "AbortError"
        throw error
      })
    )

    await expect(requestJson("GET", "/api/sessions")).rejects.toThrow(
      "Request timed out"
    )
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})
