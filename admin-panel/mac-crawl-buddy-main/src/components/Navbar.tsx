import { useNavigate, useLocation } from "react-router-dom";
import { logout, getUsername } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { LogOut } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { label: "Dashboard", path: "/dashboard" },
  { label: "Crawl Manager", path: "/crawl" },
  { label: "Scheduler", path: "/scheduler" },
];

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  return (
    <nav className="bg-primary text-primary-foreground shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <div className="flex items-center gap-6">
            <span className="text-lg font-bold tracking-tight flex items-center gap-2">
              🎓 MAC Admin
            </span>
            <div className="hidden sm:flex items-center gap-1">
              {links.map((l) => (
                <button
                  key={l.path}
                  onClick={() => navigate(l.path)}
                  className={cn(
                    "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                    location.pathname === l.path
                      ? "bg-primary-foreground/20"
                      : "hover:bg-primary-foreground/10"
                  )}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm opacity-80 hidden sm:inline">{getUsername()}</span>
            <Button variant="ghost" size="sm" onClick={handleLogout} className="text-primary-foreground hover:bg-primary-foreground/10">
              <LogOut className="h-4 w-4 mr-1" /> Logout
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
