"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { EmbeddedLens } from "@/components/lens/EmbeddedLens";

export default function LensPage() {
  const { accessToken, isAuthorized, isLoading } = useAuthorized();
  if (!accessToken || !isAuthorized || isLoading) return null;
  return <EmbeddedLens />;
}
