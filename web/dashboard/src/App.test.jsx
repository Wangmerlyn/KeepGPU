import { describe, expect, it } from "vitest"

import { REQUEST_TIMEOUT_MS } from "./lib/api"

describe("REQUEST_TIMEOUT_MS", () => {
  it("gives stop requests time to return backend timeout payloads", () => {
    expect(REQUEST_TIMEOUT_MS).toBeGreaterThan(10000)
  })
})
