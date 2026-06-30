import { describe, expect, it } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"

import App from "./App"
import { REQUEST_TIMEOUT_MS } from "./lib/api"

describe("REQUEST_TIMEOUT_MS", () => {
  it("gives stop requests time to return backend timeout payloads", () => {
    expect(REQUEST_TIMEOUT_MS).toBeGreaterThan(10000)
  })
})

describe("App", () => {
  it("renders an interval input that allows fractional seconds", () => {
    const previousWindow = globalThis.window
    globalThis.window = { location: { origin: "http://127.0.0.1:8765" } }

    try {
      const markup = renderToStaticMarkup(<App />)

      expect(markup).toMatch(/<form(?=[^>]*novalidate="")[^>]*>.*?Interval \(sec\)/s)
      expect(markup).toMatch(
        /Interval \(sec\).*?<input(?=[^>]*min="0.001")(?=[^>]*step="any")[^>]*>/s
      )
    } finally {
      globalThis.window = previousWindow
    }
  })

  it("renders manual refresh with opt-in auto refresh controls", () => {
    const previousWindow = globalThis.window
    globalThis.window = { location: { origin: "http://127.0.0.1:8765" } }

    try {
      const markup = renderToStaticMarkup(<App />)

      expect(markup).toContain("Refresh Now")
      expect(markup).toMatch(
        /<input(?=[^>]*type="checkbox")(?=[^>]*aria-label="Auto refresh")[^>]*>/s
      )
      expect(markup).toContain("manual refresh")
      expect(markup).not.toContain("refresh 3s")
    } finally {
      globalThis.window = previousWindow
    }
  })
})
