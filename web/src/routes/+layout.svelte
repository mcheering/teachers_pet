<script lang="ts">
  import { onMount } from 'svelte';
  import { token, user } from '$lib/stores/user';
  import { goto } from '$app/navigation';
  import Header from '$lib/components/Header.svelte';

  onMount(async () => {
    const path = window.location.pathname;
    if (path === '/login') return;       // ← allow public

    const jwt = localStorage.getItem('jwt');
    if (!jwt) {                          // ← no token? kick to login
      goto('/login');
      return;
    }
    token.set(jwt);

    // verify on the server
    const res = await fetch('/auth/me', {
      headers: { Authorization: `Bearer ${jwt}` }
    });
    if (!res.ok) {
      localStorage.removeItem('jwt');
      token.set(null);
      user.set(null);
      goto('/login');
      return;
    }
    user.set(await res.json());
  });
</script>

<Header />
<slot />