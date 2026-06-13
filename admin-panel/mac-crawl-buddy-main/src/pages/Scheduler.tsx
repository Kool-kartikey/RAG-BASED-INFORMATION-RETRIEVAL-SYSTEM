import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Trash2 } from "lucide-react";

const Scheduler = () => {
  const [url, setUrl] = useState("");
  const [frequency, setFrequency] = useState("daily");
  const [maxPages, setMaxPages] = useState(100);
  const [submitting, setSubmitting] = useState(false);
  const [schedules, setSchedules] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [removing, setRemoving] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchSchedules = async () => {
    setLoading(true);
    try {
      setSchedules(await api.getSchedules());
    } catch (e: any) {
      toast({ title: "Error", description: e.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchSchedules(); }, []);

  const handleSchedule = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const res = await api.scheduleCrawl(url, frequency, maxPages);
      toast({ title: "Scheduled!", description: `Job ID: ${res.job_id}` });
      setUrl("");
      fetchSchedules();
    } catch (e: any) {
      toast({ title: "Error", description: e.message, variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  const handleRemove = async (job_id: string) => {
    setRemoving(job_id);
    try {
      await api.removeSchedule(job_id);
      toast({ title: "Removed", description: "Schedule removed" });
      fetchSchedules();
    } catch (e: any) {
      toast({ title: "Error", description: e.message, variant: "destructive" });
    } finally {
      setRemoving(null);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-foreground">Scheduler</h1>

      <Card className="border-0 shadow-sm">
        <CardHeader><CardTitle className="text-lg">Add Scheduled Crawl</CardTitle></CardHeader>
        <CardContent>
          <form onSubmit={handleSchedule} className="flex flex-col sm:flex-row gap-3 items-end">
            <div className="flex-1 space-y-1">
              <Label>Subdomain URL</Label>
              <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://mac.du.ac.in" required />
            </div>
            <div className="w-36 space-y-1">
              <Label>Frequency</Label>
              <Select value={frequency} onValueChange={setFrequency}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="daily">Daily</SelectItem>
                  <SelectItem value="weekly">Weekly</SelectItem>
                  <SelectItem value="monthly">Monthly</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="w-32 space-y-1">
              <Label>Max Pages</Label>
              <Input type="number" value={maxPages} onChange={(e) => setMaxPages(Number(e.target.value))} min={1} />
            </div>
            <Button type="submit" disabled={submitting}>
              {submitting && <Loader2 className="h-4 w-4 animate-spin mr-2" />} Schedule Crawl
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card className="border-0 shadow-sm">
        <CardHeader><CardTitle className="text-lg">Active Schedules</CardTitle></CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-primary" /></div>
          ) : schedules.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">No scheduled jobs yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Job ID</TableHead>
                  <TableHead>Subdomain</TableHead>
                  <TableHead>Frequency</TableHead>
                  <TableHead>Next Run</TableHead>
                  <TableHead>Created At</TableHead>
                  <TableHead className="w-16">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {schedules.map((s: any) => (
                  <TableRow key={s.job_id}>
                    <TableCell className="font-mono text-xs">{s.job_id}</TableCell>
                    <TableCell>{s.subdomain}</TableCell>
                    <TableCell className="capitalize">{s.frequency}</TableCell>
                    <TableCell className="text-muted-foreground text-sm">{s.next_run}</TableCell>
                    <TableCell className="text-muted-foreground text-sm">{s.created_at}</TableCell>
                    <TableCell>
                      <Button variant="ghost" size="icon" className="text-destructive hover:text-destructive" onClick={() => handleRemove(s.job_id)} disabled={removing === s.job_id}>
                        {removing === s.job_id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                      </Button>
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

export default Scheduler;
