const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

function appendMessage(text, sender) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}`;
  const bubble = document.createElement("div");
  bubble.className = `bubble ${sender}`;
  bubble.textContent = text;
  msgDiv.appendChild(bubble);
  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;
  appendMessage(text, "user");
  userInput.value = "";
  appendMessage("Thinking...", "bot");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text }),
    });
    const data = await res.json();
    chatWindow.removeChild(chatWindow.lastChild); // remove 'Thinking...'
    appendMessage(data.answer || "[No response]", "bot");
  } catch (err) {
    chatWindow.removeChild(chatWindow.lastChild);
    appendMessage("Error: " + err.message, "bot");
  }
});
