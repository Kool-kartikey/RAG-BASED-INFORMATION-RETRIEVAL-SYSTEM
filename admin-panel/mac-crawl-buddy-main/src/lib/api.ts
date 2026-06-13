// API_BASE:
//   Local dev  → set VITE_API_URL="" in .env  (Vite proxy handles forwarding to :8000)
//   ngrok      → set VITE_API_URL="https://your-tunnel.ngrok-free.app" in .env
const API_BASE = import.meta.env.VITE_API_URL ?? "";

const auth = "Basic " + btoa("admin:mac2024");

// Only send the ngrok browser-warning bypass header when actually using ngrok
const isNgrok = API_BASE.includes("ngrok");

const headers = (): HeadersInit => ({
  Authorization: auth,
  "Content-Type": "application/json",
  ...(isNgrok ? { "ngrok-skip-browser-warning": "true" } : {}),
});

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...headers(), ...options?.headers },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.message || "Request failed");
  }
  return res.json();
}

export const api = {
  getStatus: () =>
    request<any>("/admin/status"),

  // ── Unwrap "crawls" array from response ──
  getCrawlHistory: () =>
    request<any>("/admin/crawl/history").then((r) => r.crawls || []),

  rebuildIndex: () =>
    request<any>("/admin/index/rebuild", { method: "POST" }),

  startCrawl: (subdomain: string, max_pages: number) =>
    request<any>("/admin/crawl/start", {
      method: "POST",
      body: JSON.stringify({ subdomain, max_pages }),
    }),

  deleteCrawl: (crawl_id: string) =>
    request<any>("/admin/crawl/delete", {
      method: "DELETE",
      body: JSON.stringify({ crawl_id }),
    }),

  scheduleCrawl: (subdomain: string, frequency: string, max_pages: number) =>
    request<any>("/admin/crawl/schedule", {
      method: "POST",
      body: JSON.stringify({ subdomain, frequency, max_pages }),
    }),

  // ── Unwrap "scheduled_jobs" array from response ──
  getSchedules: () =>
    request<any>("/admin/schedule/list").then((r) => r.scheduled_jobs || []),

  removeSchedule: (job_id: string) =>
    request<any>(`/admin/schedule/remove/${job_id}`, { method: "DELETE" }),
};