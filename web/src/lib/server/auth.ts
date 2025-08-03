import { Lucia } from "lucia";
import { postgres as postgresAdapter } from "@lucia-auth/adapter-postgresql";
import { pool } from "$lib/server/db"; // your existing pool

export const lucia = new Lucia(
  postgresAdapter(pool, {
    user: "users",          // your user table name
    session: "sessions"     // your session table name
  }),
  {
    sessionCookie: {
      name: "session_token", // this is the cookie name
      attributes: {
        secure: true,
        sameSite: "lax",
        httpOnly: true,
        path: "/"
      }
    }
  }
);