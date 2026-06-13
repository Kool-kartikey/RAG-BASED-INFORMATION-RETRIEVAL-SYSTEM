import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Trash2, RefreshCw } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";

const CrawlManager = () => {
  const [url, setUrl] = useState("");
  const [maxPages, setMaxPages] = useState(100);
  const [starting, setStarting] = useState(false);
  const [banner, setBanner] = useState<{ type: "success" | "error"; msg: string } | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await api.getCrawlHistory();
      setHistory(Array.isArray(data) ? data : (data as any)?.history ?? []);
    } catch (e: any) {
      toast({ title: "Error", description: e.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchHistory(); }, []);

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    setStarting(true);
    setBanner(null);
    try {
      const res = await api.startCrawl(url, maxPages);
      setBanner({ type: "success", msg: `Crawl started! ID: ${res.crawl_id}` });
      setUrl("");
      fetchHistory();
    } catch (e: any) {
      setBanner({ type: "error", msg: e.message });
    } finally {
      setStarting(false);
    }
  };

  const handleDelete = async (crawl_id: string) => {
    setDeleting(crawl_id);
    try {
      await api.deleteCrawl(crawl_id);
      toast({ title: "Deleted", description: "Crawl data removed successfully" });
      fetchHistory();
    } catch (e: any) {
      toast({ title: "Error", description: e.message, variant: "destructive" });
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-foreground">Crawl Manager</h1>

      <Card className="border-0 shadow-sm">
        <CardHeader><CardTitle className="text-lg">Start New Crawl</CardTitle></CardHeader>
        <CardContent>
          {banner && (
            <div className={`mb-4 p-3 rounded-md text-sm ${banner.type === "success" ? "bg-emerald-50 text-emerald-800" : "bg-red-50 text-red-800"}`}>
              {banner.msg}
            </div>
          )}
          <form onSubmit={handleStart} className="flex flex-col sm:flex-row gap-3 items-end">
            <div className="flex-1 space-y-1">
              <Label>Subdomain URL</Label>
              <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://mac.du.ac.in" required />
            </div>
            <div className="w-32 space-y-1">
              <Label>Max Pages</Label>
              <Input type="number" value={maxPages} onChange={(e) => setMaxPages(Number(e.target.value))} min={1} />
            </div>
            <Button type="submit" disabled={starting}>
              {starting && <Loader2 className="h-4 w-4 animate-spin mr-2" />} Start Crawl
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card className="border-0 shadow-sm">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg">Crawl History</CardTitle>
          <Button variant="ghost" size="icon" onClick={fetchHistory}><RefreshCw className="h-4 w-4" /></Button>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-primary" /></div>
          ) : history.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">No crawl history yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Crawl ID</TableHead>
                  <TableHead>Subdomain</TableHead>
                  <TableHead>Pages</TableHead>
                  <TableHead>Chunks</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Started At</TableHead>
                  <TableHead className="w-16">Actions</TableHead>
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
                    <TableCell>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="ghost" size="icon" className="text-destructive hover:text-destructive">
                            {deleting === c.crawl_id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete crawl data?</AlertDialogTitle>
                            <AlertDialogDescription>This will permanently remove all data for crawl {c.crawl_id}.</AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction onClick={() => handleDelete(c.crawl_id)} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">Delete</AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default CrawlManager;
