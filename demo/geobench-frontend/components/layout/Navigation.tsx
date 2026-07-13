"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sun, Moon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { STATIC_DEMO_MODE } from "@/lib/constants";

const navLinks = [
  { href: "/", label: "Home" },
  { href: "/test-council", label: "Run the Council" },
  { href: "/council", label: "The Council" },
];

export function Navigation() {
  const pathname = usePathname();
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const dark = stored === "dark" || (!stored && prefersDark);
    setIsDark(dark);
    document.documentElement.classList.toggle("dark", dark);
  }, []);

  function toggleTheme() {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/60 bg-background">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 select-none">
          <span
            className="flex h-8 w-8 items-center justify-center rounded-lg font-black text-sm"
            style={{ background: "#c6ef38", color: "#111" }}
          >
            G
          </span>
          <span className="font-extrabold text-base tracking-tight">GeoBench</span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden sm:flex items-center gap-0.5">
          {navLinks.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "px-4 py-1.5 rounded-full text-sm font-semibold transition-all",
                pathname === href
                  ? "text-[#111] dark:text-[#111]"
                  : "text-foreground/60 hover:text-foreground"
              )}
              style={pathname === href ? { background: "#c6ef38" } : undefined}
            >
              {label}
            </Link>
          ))}
        </nav>

        {/* Theme toggle */}
        <div className="flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="flex h-8 w-8 items-center justify-center rounded-full text-foreground/50 hover:text-foreground hover:bg-muted transition-all"
            aria-label="Toggle theme"
          >
            {isDark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>

      {/* Mobile nav */}
      <nav className="flex sm:hidden border-t border-border/60">
        {navLinks.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex flex-1 items-center justify-center py-2.5 text-xs font-semibold transition-colors",
              pathname === href
                ? "text-[#111] dark:text-[#111]"
                : "text-foreground/50 hover:text-foreground"
            )}
            style={pathname === href ? { background: "#c6ef38" } : undefined}
          >
            {label}
          </Link>
        ))}
      </nav>

      {/* Static-mode banner — visible only on the GitHub Pages build. */}
      {STATIC_DEMO_MODE && (
        <div
          className="border-t border-border/60 px-4 py-1.5 text-center text-xs"
          style={{ background: "rgba(198,239,56,0.12)", color: "var(--foreground)" }}
        >
          <strong>Static replay mode.</strong> You are watching a pre-recorded
          Progressive Narrowing run from a live cluster. Upload and Random
          are disabled — see the{" "}
          <a
            href="https://github.com/"
            style={{ textDecoration: "underline" }}
          >
            repository README
          </a>{" "}
          to run it against your own vLLM endpoint.
        </div>
      )}
    </header>
  );
}
