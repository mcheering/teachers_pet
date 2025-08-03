<script lang="ts">
  import type { Class } from '$lib/types';
  export let data: { classes: Class[] };
</script>

<h1 class="text-2xl font-bold mb-4">Your Classes</h1>

{#if data.classes.length === 0}
  <p>You’re not teaching or enrolled in any classes yet.</p>
{:else}
  <ul class="space-y-2">
    {#each data.classes as cls}
      <li>
        <a
          href={`/classes/${cls.id}`}
          class="block p-4 border rounded hover:bg-gray-50"
        >
          <h2 class="text-xl">{cls.name}</h2>
          <small class="text-gray-500">Join code: {cls.join_code}</small>
        </a>
      </li>
    {/each}
  </ul>
{/if}