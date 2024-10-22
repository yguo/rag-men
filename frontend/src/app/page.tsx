"use client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useState } from "react"

interface Message {
  text: string
  sender: "user" | "bot"
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")

  const handleSend = async () => {
    if (input.trim() === "") return

    const newMessage: Message = { text: input, sender: "user" }
    setMessages([...messages, newMessage])
    setInput("")

    // TODO: Send message to backend and get response
    // For now, we'll just simulate a response
    setTimeout(() => {
      const botResponse: Message = { text: "This is a placeholder response.", sender: "bot" }
      setMessages(prevMessages => [...prevMessages, botResponse])
    }, 1000)
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8">RAG-MEN Chat</h1>
        <div className="border rounded-lg p-4 h-[400px] overflow-y-auto mb-4">
          {messages.map((message, index) => (
            <div key={index} className={`mb-2 ${message.sender === "user" ? "text-right" : "text-left"}`}>
              <span className={`inline-block p-2 rounded-lg ${message.sender === "user" ? "bg-blue-500 text-white" : "bg-gray-200"}`}>
                {message.text}
              </span>
            </div>
          ))}
        </div>
        <div className="flex">
          <Input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
            className="flex-grow mr-2"
          />
          <Button onClick={handleSend}>Send</Button>
        </div>
      </div>
    </main>
  )
}