<script lang="ts">
  import Button from "../button/button.svelte";
  import ChatMessage from "../chatmessage/ChatMessage.svelte";
  import Input from "../input/input.svelte";
  import ScrollArea from "../scroll-area/scroll-area.svelte";

  export let messageArray: { text: string; sender: "user" | "bot" }[] = [];
  let newMessage: string = ""; // Variable to capture the input message
  let loading: boolean = false; // Loading state to track the send process

  // Function to send a message
  async function sendMessage() {
    if (newMessage.trim() !== "") {
      messageArray = [...messageArray, { text: newMessage, sender: "user" }];
      loading = true; // Set loading state to true
      const query = newMessage;
      newMessage = "";

      // Try to get the bot's reply from the backend
      try {
        // Send POST request to the FastAPI backend
        const response = await fetch("http://localhost:8000/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json", // Specify JSON in request
          },
          body: JSON.stringify({ query: query }), // Send the user input as the query
        });

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }

        const data = await response.json();
        messageArray = [...messageArray, { text: data.answer, sender: "bot" }];
      } catch (err) {
        console.error("Failed to fetch bot reply:", (err as Error).message);
        messageArray = [
          ...messageArray,
          {
            text: `Failed to fetch bot reply: ${(err as Error).message}`,
            sender: "bot",
          },
        ];
        throw err; // Rethrow the error to be handled elsewhere if needed
      } finally {
        loading = false; // Reset loading state
      }
    }
  }

  // Event handler for keydown to send message on Enter key
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevents the default behavior (e.g., adding a newline)
      sendMessage(); // Call sendMessage function
    }
  }
</script>

<div
  class="flex flex-col border-4 border-secondary rounded-2xl w-4/5 h-4/5 m-3 p-5"
>
  <ScrollArea class="flex-grow overflow-auto">
    {#each messageArray as { text, sender }, index}
      <ChatMessage
        align={sender === "user" ? "right" : "left"}
        message={text}
      />
    {/each}
  </ScrollArea>

  <div
    class="flex justify-between items-center border-2 border-secondary rounded-2xl mt-4 p-2"
  >
    <Input
      class="py-5 border-0"
      placeholder="Type your message..."
      bind:value={newMessage}
      on:keydown={handleKeydown}
    />
    <Button
      class="border-2 bg-primary text-white rounded-md px-4 py-2 transition duration-300 ease-in-out hover:bg-zinc-600 flex items-center justify-center"
      on:click={sendMessage}
      disabled={loading || !newMessage.trim()}
    >
      {#if loading}
        <img src="/bouncing-circles.svg" alt="Loading..." class="h-6 w-6" />
      {:else}
        Send
      {/if}
    </Button>
  </div>
</div>
