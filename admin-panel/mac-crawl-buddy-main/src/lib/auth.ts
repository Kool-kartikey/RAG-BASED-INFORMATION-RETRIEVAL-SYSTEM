const AUTH_KEY = "mac_admin_auth";

export function login(username: string, password: string): boolean {
  if (username === "admin" && password === "mac2024") {
    localStorage.setItem(AUTH_KEY, JSON.stringify({ username }));
    return true;
  }
  return false;
}

export function logout() {
  localStorage.removeItem(AUTH_KEY);
}

export function isAuthenticated(): boolean {
  return localStorage.getItem(AUTH_KEY) !== null;
}

export function getUsername(): string {
  const data = localStorage.getItem(AUTH_KEY);
  if (data) return JSON.parse(data).username;
  return "";
}
