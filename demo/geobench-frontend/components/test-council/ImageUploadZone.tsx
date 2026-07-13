"use client";

import { useRef, useState } from "react";
import { Upload, ImageIcon } from "lucide-react";

interface ImageUploadZoneProps {
  onImage: (file: File, previewUrl: string) => void;
}

export function ImageUploadZone({ onImage }: ImageUploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleFile(file: File) {
    if (!file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    onImage(file, url);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragging ? "#c6ef38" : "var(--border)"}`,
        borderRadius: 16,
        padding: "32px 16px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        cursor: "pointer",
        background: dragging ? "rgba(198,239,56,0.05)" : "var(--background)",
        transition: "border-color 0.15s, background 0.15s",
        minHeight: 160,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />
      <Upload size={28} style={{ color: dragging ? "#c6ef38" : "var(--muted-foreground)" }} />
      <div style={{ textAlign: "center" }}>
        <div style={{ fontWeight: 600, fontSize: 14, color: "var(--foreground)" }}>
          Drop an image or click to upload
        </div>
        <div style={{ fontSize: 12, color: "var(--muted-foreground)", marginTop: 4 }}>
          JPG, PNG, WebP — any street-level photo
        </div>
      </div>
    </div>
  );
}

export function ImagePreview({ url, onReset }: { url: string; onReset?: () => void }) {
  return (
    <div style={{ position: "relative", borderRadius: 16, overflow: "hidden", border: "1px solid var(--border)" }}>
      <img src={url} alt="Uploaded" style={{ width: "100%", maxHeight: 220, objectFit: "cover", display: "block" }} />
      {onReset && (
        <button
          onClick={onReset}
          style={{
            position: "absolute", top: 8, right: 8,
            background: "rgba(0,0,0,0.6)", border: "none", borderRadius: 8,
            color: "#fff", fontSize: 11, fontWeight: 600, padding: "4px 10px",
            cursor: "pointer",
          }}
        >
          Change
        </button>
      )}
    </div>
  );
}
