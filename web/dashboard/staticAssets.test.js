import { readFileSync } from "node:fs"

import { describe, expect, it } from "vitest"

const cssAssets = [
  ["source dashboard CSS", new URL("./src/index.css", import.meta.url)],
  [
    "packaged dashboard CSS",
    new URL("../../src/keep_gpu/mcp/static/assets/index.css", import.meta.url)
  ]
]

const remoteRuntimeCssPattern =
  /(?:@import\s*(?:url\(\s*)?["']?|url\(\s*["']?)(?:https?:)?\/\//gi

function findRemoteRuntimeAssets(label, css) {
  const matches = css.match(remoteRuntimeCssPattern) ?? []
  return matches.map((match) => `${label}: ${match}`)
}

describe("remote runtime CSS detection", () => {
  it("detects protocol-relative imports and URLs", () => {
    const css = '@import "//fonts.example/css"; .icon { background: url(//cdn.example/icon.svg); }'

    expect(findRemoteRuntimeAssets("fixture", css)).toEqual([
      'fixture: @import "//',
      "fixture: url(//"
    ])
  })
})

describe("dashboard CSS assets", () => {
  it("do not load remote runtime assets", () => {
    const offenders = cssAssets.flatMap(([label, url]) => {
      const css = readFileSync(url, "utf8")
      return findRemoteRuntimeAssets(label, css)
    })

    expect(offenders).toEqual([])
  })
})
