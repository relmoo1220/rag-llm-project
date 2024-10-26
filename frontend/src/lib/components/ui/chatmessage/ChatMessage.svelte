<script>
  import * as Avatar from "$lib/components/ui/avatar";

  export let message = "";
  export let align = "left"; // 'left' or 'right'
</script>

<div class={`flex ${align === "right" ? "justify-end" : "justify-start"} my-2`}>
  {#if align === "left"}
    <Avatar.Root class="my-2 mx-4 p-1 bg-secondary">
      <Avatar.Image src="./bot.svg" alt="bot" />
    </Avatar.Root>
  {/if}
  <div
    class={`max-w-xs p-4 rounded-lg ${align === "right" ? "bg-blue-500 text-white" : "bg-gray-200 text-black"}`}
  >
    {message}
  </div>
  {#if align === "right"}
    <Avatar.Root class="my-2 mx-4 p-1 bg-secondary">
      <Avatar.Image src="./user.svg" alt="user" />
    </Avatar.Root>
  {/if}
</div>
