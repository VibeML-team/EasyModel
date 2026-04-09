# VibeML Web

React migration workspace for VibeML frontend.

## Stack

- React + TypeScript + Vite
- Zustand for local app state
- TanStack Query for server state
- React Router for route management

## Commands

```bash
pnpm --dir web dev
pnpm --dir web typecheck
pnpm --dir web build
```

## Runtime Paths

- Dev: http://localhost:5173/
- API base: /api
- Production mount path: /app-next/

## Notes

- Legacy pages are kept under the existing FastAPI routes.
- This app is the gradual migration target for v2, dashboard, and landing/detail pages.
