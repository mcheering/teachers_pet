<script lang="ts">
  export let data: { classes: any[] };
</script>

<h1>Teacher Dashboard</h1>
<ul>
  {#each data.classes as cls}
    <li><a href="/classes/{cls.id}">{cls.name}</a></li>
  {/each}
</ul>