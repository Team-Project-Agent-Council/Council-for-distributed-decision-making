"use client";

import { Skeleton } from "@/components/ui/skeleton";

export function CouncilSkeleton() {
  return (
    <div className="space-y-8 p-6 max-w-7xl mx-auto">
      {/* Hero skeleton */}
      <Skeleton className="h-64 w-full rounded-xl" />
      {/* Agent grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-48 rounded-xl" />
        ))}
      </div>
      {/* Collaboration flow */}
      <Skeleton className="h-40 w-full rounded-xl" />
    </div>
  );
}

export function StatisticsSkeleton() {
  return (
    <div className="space-y-6 p-6 max-w-7xl mx-auto">
      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-28 rounded-xl" />
        ))}
      </div>
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Skeleton className="h-72 rounded-xl" />
        <Skeleton className="h-72 rounded-xl" />
      </div>
      {/* Hall of Fame */}
      <Skeleton className="h-64 rounded-xl" />
    </div>
  );
}

export function ProfileSkeleton() {
  return (
    <div className="space-y-6 p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Skeleton className="h-20 w-20 rounded-full" />
        <div className="space-y-2 flex-1">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-full max-w-xs" />
        </div>
      </div>
      {/* Achievements grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-32 rounded-xl" />
        ))}
      </div>
      {/* Game history */}
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-16 rounded-lg" />
        ))}
      </div>
    </div>
  );
}
