import Link from "next/link";
import { MapPinOff } from "lucide-react";

export default function NotFound() {
  return (
    <div
      className="flex min-h-screen flex-col items-center justify-center px-4"
      style={{ background: "#111111", color: "#f5f5f5" }}
    >
      <MapPinOff size={80} className="mb-6" style={{ color: "#737373" }} />

      <h1 className="mb-3 text-4xl font-bold tracking-tight sm:text-5xl">
        Lost in the world?
      </h1>

      <p className="mb-8 text-lg" style={{ color: "#737373" }}>
        Even the Council can&apos;t find this page.
      </p>

      <Link href="/game" className="geo-btn geo-btn-primary">
        Back to Game
      </Link>

      <div
        className="mt-12 flex gap-4 text-2xl select-none"
        style={{ color: "#737373", opacity: 0.4 }}
        aria-hidden="true"
      >
        <span>&#x1F5FA;&#xFE0F;</span>
        <span>&#x1F50D;</span>
        <span>&#x1F52E;</span>
        <span>&#x1F9ED;</span>
        <span>&#x1F4DC;</span>
      </div>
    </div>
  );
}
