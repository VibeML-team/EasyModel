# Embedded Fonts

Place the following files in this folder to enable embedded typography:

- `AvenirNextLTPro-Regular.woff2`
- `AvenirNextLTPro-Regular.woff`

`index.html` already references these files via `@font-face`.

If files are missing, the browser will fallback to its default font for missing glyphs (Chinese remains system default by design).
