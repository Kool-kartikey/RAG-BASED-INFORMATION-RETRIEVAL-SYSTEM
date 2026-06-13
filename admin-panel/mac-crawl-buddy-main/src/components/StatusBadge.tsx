import { cn } from "@/lib/utils";

const colors: Record<string, string> = {
  done: "bg-emerald-100 text-emerald-800",
  running: "bg-amber-100 text-amber-800",
  failed: "bg-red-100 text-red-800",
};

const StatusBadge = ({ status }: { status: string }) => (
  <span className={cn("px-2 py-0.5 rounded-full text-xs font-medium capitalize", colors[status] || "bg-muted text-muted-foreground")}>
    {status}
  </span>
);

export default StatusBadge;
