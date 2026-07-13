"use client";

const BG_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Empire_State_Building_%28aerial_view%29.jpg/1280px-Empire_State_Building_%28aerial_view%29.jpg";

export function CouncilHero() {
  return (
    <div
      className="relative rounded-2xl overflow-hidden px-8 py-14 sm:px-14"
      style={{ minHeight: "280px", background: "#111" }}
    >
      <img
        src={BG_IMAGE}
        alt=""
        className="absolute inset-0 w-full h-full object-cover"
        style={{ pointerEvents: "none" }}
      />

      <div
        className="absolute inset-0"
        style={{ background: "linear-gradient(to right, rgba(0,0,0,0.82) 0%, rgba(0,0,0,0.55) 60%, rgba(0,0,0,0.2) 100%)" }}
      />

      <div className="relative z-10 max-w-2xl">
        <span
          className="inline-block text-xs font-bold tracking-[0.15em] uppercase mb-5 px-3 py-1 rounded-full"
          style={{ background: "rgba(198,239,56,0.15)", color: "#c6ef38" }}
        >
          Progressive Narrowing Council
        </span>
        <h1 className="text-4xl sm:text-5xl font-black tracking-tight mb-4 text-white">
          Meet the Council
        </h1>
        <p className="text-base leading-relaxed max-w-lg" style={{ color: "rgba(255,255,255,0.6)" }}>
          Five specialist vision-language agents work in isolation, then a judge funnels their
          evidence through a region-then-country narrowing pipeline to pin down where the image
          was taken.
        </p>
      </div>
    </div>
  );
}
