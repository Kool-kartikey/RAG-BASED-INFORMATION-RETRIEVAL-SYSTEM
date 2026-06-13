import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { Database, Globe, CalendarClock, Activity, RefreshCw, Loader2, Trash2, AlertTriangle } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";

// Admin credentials — same as the rest of the app
const ADMIN_USER = "admin";
const ADMIN_PASS = "mac2024";

// Build Basic Auth header
const basicAuth = "Basic " + btoa(`${ADMIN_USER}:${ADMIN_PASS}`);

const Dashboard = () => {
  const [status, setStatus] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [rebuilding, setRebuilding] = useState(false);
  const [error, setError] = useState("");

  // Reset state — two-step confirm
  const [resetConfirm, setResetConfirm] = useState(false);   // first click
  const [resetting, setResetting]       = useState(false);   // API call in progress

  const { toast } = useToast();

  const fetchData = async () => {
    setLoading(true);
    setError("");
    try {
      const [s, h] = await Promise.all([api.getStatus(), api.getCrawlHistory()]);
      setStatus(s);
      setHistory(Array.isArray(h) ? h : (h as any)?.history ?? []);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const handleRebuild = async () => {
    setRebuilding(true);
    try {
      const res = await api.rebuildIndex();
      toast({ title: "Index Rebuilt", description: `${res.chunks ?? "?"} chunks indexed in ${res.time ?? "?"}s` });
      fetchData();
    } catch (e: any) {
      toast({ title: "Error", description: e.message, variant: "destructive" });
    } finally {
      setRebuilding(false);
    }
  };

  // Step 1 — first click just shows confirm button
  const handleResetClick = () => {
    setResetConfirm(true);
  };

  // Step 2 — confirmed, call the API
  const handleResetConfirm = async () => {
    setResetting(true);
    setResetConfirm(false);
    try {
      // Direct fetch — DELETE /admin/data/reset with Basic Auth
      const baseUrl = (window as any).__API_BASE__ ?? "http://localhost:8000";
      const response = await fetch(`${baseUrl}/admin/data/reset`, {
        method : "DELETE",
        headers: {
          "Authorization"              : basicAuth,
          "ngrok-skip-browser-warning" : "true",
        },
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail ?? `HTTP ${response.status}`);
      }

      const data = await response.json();
      toast({
        title      : "⚠️ Full Reset Complete",
        description: "All chunks and index cleared. Run a new crawl + Rebuild Index to restore.",
        variant    : "destructive",
      });
      fetchData();  // refresh stats — should now show 0 chunks
    } catch (e: any) {
      toast({ title: "Reset Failed", description: e.message, variant: "destructive" });
    } finally {
      setResetting(false);
    }
  };

  // Cancel confirm
  const handleResetCancel = () => {
    setResetConfirm(false);
  };

  if (loading) return <div className="flex justify-center py-20"><Loader2 className="h-8 w-8 animate-spin text-primary" /></div>;
  if (error) return <div className="bg-destructive/10 text-destructive p-4 rounded-md">{error}</div>;

  const stats = [
    { label: "Total Chunks", value: status?.total_chunks ?? 0, icon: Database, color: "text-blue-600 bg-blue-50" },
    { label: "Total Crawls", value: status?.total_crawls ?? 0, icon: Globe, color: "text-emerald-600 bg-emerald-50" },
    { label: "Scheduled Jobs", value: status?.scheduled_jobs ?? 0, icon: CalendarClock, color: "text-violet-600 bg-violet-50" },
    { label: "Index Status", value: status?.index_exists ? "Active" : "Missing", icon: Activity, color: status?.index_exists ? "text-emerald-600 bg-emerald-50" : "text-red-600 bg-red-50" },
  ];

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
        <Button onClick={handleRebuild} disabled={rebuilding}>
          {rebuilding ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <RefreshCw className="h-4 w-4 mr-2" />}
          Rebuild Index
        </Button>
      </div>

      {/* ── Stat Cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <Card key={s.label} className="border-0 shadow-sm">
            <CardContent className="flex items-center gap-4 p-5">
              <div className={`p-3 rounded-lg ${s.color}`}>
                <s.icon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{s.label}</p>
                <p className="text-xl font-bold">{s.value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* ── Recent Crawls Table ── */}
      <Card className="border-0 shadow-sm">
        <CardHeader><CardTitle className="text-lg">Recent Crawls</CardTitle></CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">No crawls yet. Start one from the Crawl Manager.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Crawl ID</TableHead>
                  <TableHead>Subdomain</TableHead>
                  <TableHead>Pages</TableHead>
                  <TableHead>Chunks</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Date</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.map((c: any) => (
                  <TableRow key={c.crawl_id}>
                    <TableCell className="font-mono text-xs">{c.crawl_id}</TableCell>
                    <TableCell>{c.subdomain}</TableCell>
                    <TableCell>{c.pages_crawled}</TableCell>
                    <TableCell>{c.chunks_stored}</TableCell>
                    <TableCell><StatusBadge status={c.status} /></TableCell>
                    <TableCell className="text-muted-foreground text-sm">{c.started_at}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* ── Danger Zone ── */}
      <Card className="border-2 border-red-200 shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2 text-red-600">
            <AlertTriangle className="h-5 w-5" />
            Danger Zone
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <p className="font-medium text-foreground">Reset All Data</p>
              <p className="text-sm text-muted-foreground mt-1">
                Permanently deletes all chunks, FAISS index, and raw pages. 
                MongoDB chunks and embeddings are also cleared.
                Use only when the system needs a complete fresh start.
                <span className="font-semibold text-red-500"> This cannot be undone.</span>
              </p>
              {/* Post-reset instructions shown inline */}
              {status?.total_chunks === 0 && (
                <p className="text-sm text-amber-600 mt-2 font-medium">
                  ⚠️ Index is empty — go to Crawl Manager to start a new crawl, then Rebuild Index.
                </p>
              )}
            </div>

            <div className="flex items-center gap-2 shrink-0">
              {/* Initial state — single red button */}
              {!resetConfirm && !resetting && (
                <Button
                  variant   = "destructive"
                  onClick   = {handleResetClick}
                  className = "whitespace-nowrap"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Reset All Data
                </Button>
              )}

              {/* Confirm state — cancel + confirm buttons */}
              {resetConfirm && !resetting && (
                <>
                  <Button variant="outline" onClick={handleResetCancel}>
                    Cancel
                  </Button>
                  <Button
                    variant   = "destructive"
                    onClick   = {handleResetConfirm}
                    className = "whitespace-nowrap"
                  >
                    <AlertTriangle className="h-4 w-4 mr-2" />
                    Yes, Delete Everything
                  </Button>
                </>
              )}

              {/* In-progress state */}
              {resetting && (
                <Button variant="destructive" disabled>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Resetting...
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;
