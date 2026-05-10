"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Command } from "cmdk";
import { Search, BarChart3, FileCog2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/search",       label: "Search",       icon: Search   },
  { href: "/eval",         label: "Eval",         icon: BarChart3 },
  { href: "/architecture", label: "Architecture", icon: FileCog2 },
] as const;

export function TopNav() {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <header className="border-border bg-background/70 supports-[backdrop-filter]:bg-background/50 sticky top-0 z-40 border-b backdrop-blur">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-6 px-4">
        <Link href="/" className="group flex items-center gap-2">
          <Sparkles className="text-primary h-5 w-5" />
          <span className="font-mono text-sm font-semibold tracking-tight">
            pathfinder
          </span>
        </Link>

        <nav className="flex items-center gap-1 text-sm">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname === href || pathname?.startsWith(`${href}/`);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "hover:bg-accent hover:text-accent-foreground inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 transition-colors",
                  active && "bg-accent text-accent-foreground",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </Link>
            );
          })}
        </nav>

        <button
          type="button"
          onClick={() => setPaletteOpen(true)}
          className="border-border bg-muted/40 hover:bg-muted text-muted-foreground ml-auto inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs"
          aria-label="Open command palette"
        >
          <Search className="h-3.5 w-3.5" />
          Search…
          <kbd className="border-border bg-background ml-2 rounded border px-1.5 py-0.5 font-mono text-[10px]">
            ⌘K
          </kbd>
        </button>
      </div>

      {paletteOpen ? (
        <div
          className="bg-background/70 fixed inset-0 z-50 flex items-start justify-center p-4 pt-32 backdrop-blur-sm"
          onClick={(e) => e.target === e.currentTarget && setPaletteOpen(false)}
        >
          <Command
            label="Command palette"
            className="border-border bg-popover w-full max-w-lg rounded-lg border shadow-2xl"
          >
            <Command.Input
              placeholder="Type a query, e.g. ‘Python engineers in Bangalore, 5+ years, fintech’"
              className="bg-transparent text-foreground placeholder:text-muted-foreground border-border w-full rounded-t-lg border-b px-4 py-3 outline-none"
              autoFocus
            />
            <Command.List className="max-h-80 overflow-y-auto p-2 text-sm">
              <Command.Empty className="text-muted-foreground px-3 py-6 text-center">
                No matching pages.
              </Command.Empty>
              <Command.Group heading="Pages">
                {NAV.map(({ href, label, icon: Icon }) => (
                  <Command.Item
                    key={href}
                    className="data-[selected=true]:bg-accent data-[selected=true]:text-accent-foreground flex cursor-pointer items-center gap-2 rounded-md px-3 py-2"
                    onSelect={() => {
                      setPaletteOpen(false);
                      window.location.assign(href);
                    }}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </Command.Item>
                ))}
              </Command.Group>
            </Command.List>
          </Command>
        </div>
      ) : null}
    </header>
  );
}
