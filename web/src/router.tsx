import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./ui/AppShell";
import { DashboardPage } from "./views/DashboardPage";
import { HomePage } from "./views/HomePage";
import { NotFoundPage } from "./views/NotFoundPage";
import { V2Page } from "./views/V2Page";

export function AppRouter() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<HomePage />} />
          <Route path="v2" element={<V2Page />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="home" element={<Navigate to="/" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
