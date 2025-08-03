// src/lib/stores/user.ts
import { writable } from 'svelte/store';

export interface User {
  sub: string;
  role: string;
  exp: number;
}

// JWT string or null
export const token = writable<string | null>(null);

// Decoded user claims or null
export const user = writable<User | null>(null);