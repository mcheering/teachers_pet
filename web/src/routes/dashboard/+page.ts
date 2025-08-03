// src/routes/dashboard/+page.ts
import type { Submission } from '$lib/types';

export async function load(): Promise<{ classes: Submission[] }> {
  // Use the global fetch (SvelteKit injects the correct one), which TS already knows
  const res = await fetch('/classes');
  if (!res.ok) throw new Error('Failed to load classes');
  const classes: Submission[] = await res.json();
  return { classes };
}