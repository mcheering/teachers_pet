<script lang="ts">
  import { token, user } from '$lib/stores/user';
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';

  let email = '';
  let password = '';
  let error = '';

  // If already logged-in, send straight to dashboard
  onMount(() => {
    const jwt = localStorage.getItem('jwt');
    if (jwt) goto('/');
  });

  async function handleSubmit() {
    error = '';
    try {
      const res = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      if (!res.ok) {
        error = await res.text() || 'Login failed';
        return;
      }
      const { token: jwt } = await res.json();
      // persist & update store
      localStorage.setItem('jwt', jwt);
      token.set(jwt);

      // fetch the user info
      const me = await fetch('/auth/me', {
        headers: { Authorization: `Bearer ${jwt}` }
      }).then((r) => r.json());
      user.set(me);

      goto('/');
    } catch {
      error = 'Network error';
    }
  }
</script>

<form on:submit|preventDefault={handleSubmit} class="max-w-sm mx-auto p-4">
  {#if error}
    <div class="mb-2 text-red-600">{error}</div>
  {/if}
  <div class="mb-4">
    <label class="block mb-1">Email</label>
    <input
      type="email"
      bind:value={email}
      required
      class="w-full border rounded p-2"
    />
  </div>
  <div class="mb-4">
    <label class="block mb-1">Password</label>
    <input
      type="password"
      bind:value={password}
      required
      class="w-full border rounded p-2"
    />
  </div>
  <button type="submit" class="w-full bg-blue-600 text-white p-2 rounded">
    Login
  </button>
</form>