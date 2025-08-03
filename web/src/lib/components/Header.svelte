<script lang="ts">
  import { token, user } from '$lib/stores/user';
  import { goto } from '$app/navigation';
  $: $user; // subscribe

  function logout() {
    localStorage.removeItem('jwt');
    token.set(null);
    user.set(null);
    goto('/login');
  }
</script>

<nav class="flex items-center justify-between p-4 bg-gray-100">
  <div class="text-xl font-semibold">Teachers’ Pet</div>
  {#if $user}
    <div class="flex items-center space-x-4">
      <span class="text-gray-700">{$user.sub} ({$user.role})</span>
      <button
        on:click={logout}
        class="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
      >
        Logout
      </button>
    </div>
  {/if}
</nav>