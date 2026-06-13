import { Navigate, Outlet } from "react-router-dom";
import { isAuthenticated } from "@/lib/auth";
import Navbar from "./Navbar";

const AppLayout = () => {
  if (!isAuthenticated()) return <Navigate to="/" replace />;
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Outlet />
      </main>
    </div>
  );
};

export default AppLayout;
