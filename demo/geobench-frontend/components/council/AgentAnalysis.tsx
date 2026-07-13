"use client";

import { useState, useEffect, useRef } from "react";
import { useInView } from "framer-motion";

interface AgentAnalysisProps {
  text: string;
  speed?: number;
}

export function AgentAnalysis({ text, speed = 30 }: AgentAnalysisProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true });
  const [charIndex, setCharIndex] = useState(0);

  useEffect(() => {
    if (!isInView) return;

    if (charIndex >= text.length) return;

    const timer = setInterval(() => {
      setCharIndex((prev) => {
        if (prev >= text.length) {
          clearInterval(timer);
          return prev;
        }
        return prev + 1;
      });
    }, speed);

    return () => clearInterval(timer);
  }, [isInView, text, speed, charIndex]);

  useEffect(() => {
    setCharIndex(0);
  }, [text]);

  const displayedText = text.slice(0, charIndex);
  const isTyping = charIndex < text.length;

  return (
    <div
      ref={ref}
      className="mt-3 rounded-lg px-3 py-2.5 font-[family-name:var(--font-geist-mono)] text-xs leading-relaxed text-muted-foreground/80"
      style={{ background: "var(--muted)" }}
    >
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground/50 block mb-1.5">
        Example Analysis
      </span>
      <span>{displayedText}</span>
      {isTyping && (
        <span className="inline-block w-[2px] h-3.5 bg-current align-text-bottom ml-0.5 animate-pulse" />
      )}
    </div>
  );
}
